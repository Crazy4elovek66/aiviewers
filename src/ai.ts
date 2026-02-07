import axios from 'axios';
import { EventEmitter } from 'events';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';
import { unlinkSync, existsSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { FfmpegCommand } from 'fluent-ffmpeg';
import { spawn } from 'child_process';
import crypto from 'crypto';

import { logger } from './logger';
import { analyzePcm16le } from './audioDebug';
import { MemoryStore, MemoryEvent, MemoryFact } from './memory';

ffmpeg.setFfmpegPath(ffmpegInstaller.path);

interface TwitchChannelInfo {
  title: string;
  description: string;
  gameName: string;
  viewerCount: number;
  isLive: boolean;
}

// ===== Helpers =====
function mergeFactsSimple(existing: MemoryFact[], incoming: Omit<MemoryFact, 'id' | 'ts'>[]): MemoryFact[] {
  const now = Date.now();
  const out = [...existing];

  const normKey = (s: string) => normalizeMessage(s).toLowerCase();

  const existingKeys = new Map<string, MemoryFact>();
  for (const f of out) existingKeys.set(normKey(f.text), f);

  for (const inc of incoming) {
    const key = normKey(inc.text);
    if (!key) continue;

    const prev = existingKeys.get(key);
    if (prev) {
      // refresh timestamp + keep max importance
      prev.ts = now;
      prev.importance = clamp(Math.max(prev.importance || 1, inc.importance || 1), 1, 5);
      prev.tags = Array.from(new Set([...(prev.tags || []), ...(inc.tags || [])]));
    } else {
      const created: MemoryFact = {
        id: `${now}-${crypto.randomBytes(6).toString('hex')}`,
        ts: now,
        text: inc.text,
        importance: clamp(inc.importance, 1, 5),
        tags: inc.tags || []
      };
      out.push(created);
      existingKeys.set(key, created);
    }
  }

  // cap size
  out.sort((a, b) => (b.importance - a.importance) || (b.ts - a.ts));
  return out.slice(0, 120);
}

function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

function stripCodeFences(s: string): string {
  const t = (s || '').trim();
  if (!t) return t;
  // ```json ... ``` or ``` ... ```
  return t.replace(/^```(?:json)?\s*/i, '').replace(/```$/i, '').trim();
}

function extractJsonObject(text: string): any | null {
  const t = stripCodeFences(text);
  // try direct
  try {
    return JSON.parse(t);
  } catch { /* noop */ }

  // find first '{' and last '}' to salvage JSON
  const start = t.indexOf('{');
  const end = t.lastIndexOf('}');
  if (start >= 0 && end > start) {
    const slice = t.slice(start, end + 1);
    try {
      return JSON.parse(slice);
    } catch { /* noop */ }
  }

  return null;
}

function normalizeMessage(text: string): string {
  return (text || '')
    .replace(/\s+/g, ' ')
    .replace(/[“”]/g, '"')
    .replace(/[’]/g, "'")
    .trim();
}

function tooManyEmojis(s: string): boolean {
  // rough emoji check (covers most emoji ranges)
  const m = s.match(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu);
  return (m?.length || 0) > 1;
}

function reduceEmojiSpam(s: string): string {
  let t = s;
  // if more than 1 emoji – keep only the first one
  const arr = t.match(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu);
  if (arr && arr.length > 1) {
    let kept = false;
    t = t.replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, (ch) => {
      if (!kept) { kept = true; return ch; }
      return '';
    });
  }
  return t.trim();
}

// Русский Twitch-системный промт (живой чатер, без обяз. смайликов/«чел»)
const RU_SYSTEM_PROMPT = `
Ты — обычный зритель Twitch, который пишет в чат по происходящему на стриме.

Правила:
- Пиши естественно, коротко (обычно 1 фраза).
- НЕ используй слово «чел» как обязательное. Можешь иногда, но редко.
- Эмодзи/смайлы не обязательны; используй редко, только когда уместно.
- Не пересказывай дословно речь стримера. Реагируй по смыслу.
- Без ссылок, команд, капса, спама и одинаковых сообщений подряд.
- Если нет повода (вопрос/обращение/яркая эмоция/активный чат) — лучше молчи (верни пустую строку).

Верни только текст сообщения для чата, без пояснений.
`.trim();

export class AIService extends EventEmitter {
  private isCapturing = false;
  private isProcessing = false;
  private currentProcess: FfmpegCommand | null = null;
  private tempAudioFile: string | null = null;

  private accessToken: string | null = null;
  private _currentChannelInfo: TwitchChannelInfo | null = null;
  private currentHlsUrl: string | null = null;

  private yandexApiKey: string;
  private yandexFolderId: string;

  // ===== pacing / anti-spam =====
  private aiMinIntervalMs: number;
  private aiJitter: number;
  private nextAllowedAiTs = 0;

  // ===== VAD =====
  private aiRmsMin: number;

  // ===== dedupe =====
  private lastNonEmptyTranscription = '';

  // ===== Memory =====
  private memoryStore = new MemoryStore();
  private channelIdForMemory = '';
  private facts: MemoryFact[] = [];
  private lastFactsUpdateTs = 0;

  // short-term context
  private shortHistory: { role: 'user' | 'assistant'; text: string; ts: number }[] = [];
  private shortEvents: MemoryEvent[] = [];

  // ===== bot identity / mentions =====
  private botUsername: string;

  // ===== screen OCR =====
  private ocrEnabled: boolean;
  private ocrIntervalMs: number;
  private lastOcrTs = 0;
  private lastScreenText = '';

  constructor() {
    super();
    logger.info('AIService initialized');

    this.yandexApiKey = process.env.YANDEX_API_KEY || '';
    this.yandexFolderId = process.env.YANDEX_FOLDER_ID || '';

    if (!this.yandexApiKey || !this.yandexFolderId) {
      logger.warn('Yandex credentials are missing: set YANDEX_API_KEY and YANDEX_FOLDER_ID');
    } else {
      logger.info('Yandex client configured');
    }

    this.aiMinIntervalMs = parseInt(process.env.AI_MIN_INTERVAL_MS || process.env.MESSAGE_INTERVAL || '90000', 10);
    this.aiJitter = clamp(Number(process.env.AI_JITTER || 0.25), 0, 0.9);
    this.aiRmsMin = clamp(Number(process.env.AI_RMS_MIN || 0.0025), 0, 1);

    this.botUsername = (process.env.BOT_USERNAME || process.env.TWITCH_BOT_USERNAME || '').trim().toLowerCase();

    this.ocrEnabled = (process.env.OCR_ENABLED || '').trim() === '1';
    this.ocrIntervalMs = parseInt(process.env.OCR_INTERVAL_MS || '30000', 10);

    // Подхватываем входящий чат-контекст (его шлёт bot.ts)
    this.on('chatMessage', (payload: string) => {
      try {
        const ctx = JSON.parse(payload);
        const user = String(ctx.username || 'viewer');
        const msg = String(ctx.chatMessage || '').trim();
        if (!msg) return;

        // store chat event
        this.recordEvent({
          ts: Date.now(),
          type: 'chat',
          text: `${user}: ${msg}`,
          meta: { user, raw: msg }
        });
      } catch (e) {
        logger.error('Failed to parse chatMessage payload for memory:', e);
      }
    });
  }

  public get currentChannelInfo(): TwitchChannelInfo | null {
    return this._currentChannelInfo;
  }

  private set currentChannelInfo(info: TwitchChannelInfo | null) {
    this._currentChannelInfo = info;
  }

  // ===== Screen context hook (public manual) =====
  public recordScreenContext(text: string, meta?: Record<string, any>) {
    const t = (text || '').trim();
    if (!t) return;
    this.lastScreenText = t;
    this.recordEvent({ ts: Date.now(), type: 'screen', text: t, meta });
  }

  private recordEvent(ev: MemoryEvent) {
    this.shortEvents.push(ev);

    const prefix =
      ev.type === 'speech' ? 'Стример: ' :
        ev.type === 'chat' ? 'Чат: ' :
          'Экран: ';

    this.shortHistory.push({ role: 'user', text: `${prefix}${ev.text}`, ts: ev.ts });

    if (this.shortHistory.length > 30) this.shortHistory = this.shortHistory.slice(-30);
    if (this.shortEvents.length > 60) this.shortEvents = this.shortEvents.slice(-60);
  }

  private pushAssistantTurn(text: string) {
    const t = (text || '').trim();
    if (!t) return;
    this.shortHistory.push({ role: 'assistant', text: t, ts: Date.now() });
    if (this.shortHistory.length > 30) this.shortHistory = this.shortHistory.slice(-30);
  }

  private nextAllowedAi(now: number): number {
    const mult = (1 - this.aiJitter) + Math.random() * (2 * this.aiJitter);
    return now + Math.floor(this.aiMinIntervalMs * mult);
  }

  private canCallAi(now: number): boolean {
    return now >= this.nextAllowedAiTs;
  }

  private isChatActive(now: number, windowMs = 25000, minMsgs = 2): boolean {
    const msgs = this.shortEvents.filter(e => e.type === 'chat' && now - e.ts <= windowMs);
    return msgs.length >= minMsgs;
  }

  private lastChatMessage(): { user: string; text: string } | null {
    for (let i = this.shortEvents.length - 1; i >= 0; i--) {
      const e = this.shortEvents[i];
      if (e.type !== 'chat') continue;
      const raw = (e.meta?.raw ? String(e.meta.raw) : e.text) || '';
      const user = (e.meta?.user ? String(e.meta.user) : '').trim();
      const msg = raw.includes(':') ? raw.split(':').slice(1).join(':').trim() : raw.trim();
      return { user, text: msg };
    }
    return null;
  }

  private wasMentionedInChat(now: number, windowMs = 45000): boolean {
    if (!this.botUsername) return false;
    const name = this.botUsername;
    const re = new RegExp(`(^|\\s|@)${name}(\\s|$)`, 'i');

    return this.shortEvents.some(e => {
      if (e.type !== 'chat') return false;
      if (now - e.ts > windowMs) return false;
      const raw = (e.meta?.raw ? String(e.meta.raw) : e.text) || '';
      return re.test(raw);
    });
  }

  private hasQuestionInChat(now: number, windowMs = 45000): boolean {
    return this.shortEvents.some(e => {
      if (e.type !== 'chat') return false;
      if (now - e.ts > windowMs) return false;
      const raw = (e.meta?.raw ? String(e.meta.raw) : e.text) || '';
      return raw.includes('?');
    });
  }

  private hasBrightEmotion(transcription: string, rms: number): boolean {
    const t = (transcription || '').toLowerCase();
    if (rms >= 0.02) return true; // громкая/эмоциональная речь
    if (/[!？?]{2,}/.test(transcription)) return true;
    if (/(ахах|хаха|лол|жесть|капец|блин|ура|офиг|пизд|смешно|страшно)/i.test(t)) return true;
    return false;
  }

  private shouldRespond(now: number, transcription: string, rms: number): { ok: boolean; reason: string } {
    // 1) direct reasons from chat
    if (this.wasMentionedInChat(now)) return { ok: true, reason: 'mention' };
    if (this.hasQuestionInChat(now)) return { ok: true, reason: 'question' };
    if (this.isChatActive(now)) return { ok: true, reason: 'chat_active' };

    // 2) speech-only: react only when it sounds emotional or clearly asks something
    if (this.hasBrightEmotion(transcription, rms)) return { ok: true, reason: 'emotion' };

    return { ok: false, reason: 'no_trigger' };
  }

  // ===== Twitch auth / info =====
  private async generateAccessToken(): Promise<string> {
    logger.info('Generating new access token...');
    const response = await axios.post('https://id.twitch.tv/oauth2/token', null, {
      params: {
        client_id: process.env.TWITCH_CLIENT_ID,
        client_secret: process.env.TWITCH_CLIENT_SECRET,
        grant_type: 'client_credentials'
      }
    });

    const { access_token } = response.data;
    logger.info('New access token generated successfully');
    return access_token;
  }

  private async getChannelInfo(channelName: string): Promise<TwitchChannelInfo> {
    logger.info('Fetching channel info for:', channelName);

    if (!this.accessToken) this.accessToken = await this.generateAccessToken();

    const userResponse = await axios.get(`https://api.twitch.tv/helix/users?login=${channelName}`, {
      headers: {
        'Client-ID': process.env.TWITCH_CLIENT_ID,
        Authorization: `Bearer ${this.accessToken}`
      }
    });

    const userData = userResponse.data.data[0];
    if (!userData) throw new Error(`Channel not found: ${channelName}`);

    const channelId = userData.id;
    this.channelIdForMemory = channelId;

    const streamResponse = await axios.get(`https://api.twitch.tv/helix/streams?user_id=${channelId}`, {
      headers: {
        'Client-ID': process.env.TWITCH_CLIENT_ID,
        Authorization: `Bearer ${this.accessToken}`
      }
    });

    const streamData = streamResponse.data.data[0];
    const isLive = !!streamData;

    const title = streamData?.title || userData?.description || 'Stream';
    const gameName = streamData?.game_name || 'Just Chatting';
    const viewerCount = streamData?.viewer_count || 0;

    const info: TwitchChannelInfo = {
      title,
      description: userData?.description || '',
      gameName,
      viewerCount,
      isLive
    };

    logger.info('Channel info retrieved:', {
      title: info.title,
      game: info.gameName,
      viewers: info.viewerCount,
      isLive: info.isLive
    });

    return info;
  }

  private async getTwitchHlsUrl(channel: string): Promise<string> {
    const TWITCH_WEB_CLIENT_ID = process.env.TWITCH_WEB_CLIENT_ID || 'kimne78kx3ncx6brgo4mv6wki5h1ko';
    if (!this.accessToken) this.accessToken = await this.generateAccessToken();

    // IMPORTANT: variables must match the query usage, иначе Twitch ругается "never used"
    const body = [
      {
        operationName: 'PlaybackAccessToken',
        variables: { login: channel, playerType: 'site' },
        query: `
          query PlaybackAccessToken($login: String!, $playerType: String!) {
            streamPlaybackAccessToken(
              channelName: $login,
              params: { platform: "web", playerBackend: "mediaplayer", playerType: $playerType }
            ) {
              value
              signature
            }
          }
        `
      }
    ];

    const r = await axios.post('https://gql.twitch.tv/gql', body, {
      headers: {
        'Client-ID': TWITCH_WEB_CLIENT_ID,
        Authorization: `Bearer ${this.accessToken}`,
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0',
        Accept: 'application/json'
      },
      timeout: 15000
    });

    const payload = Array.isArray(r.data) ? r.data[0] : r.data;

    if (payload?.errors?.length) {
      logger.error('Twitch GQL errors:', JSON.stringify(payload.errors, null, 2));
    }

    const tokenObj = payload?.data?.streamPlaybackAccessToken;
    const sig = tokenObj?.signature;
    const token = tokenObj?.value;

    if (!sig || !token) {
      logger.error('Twitch GQL payload:', JSON.stringify(payload, null, 2));
      throw new Error(`Failed to get playback token for channel ${channel}`);
    }

    const encodedToken = encodeURIComponent(token);

    return (
      `https://usher.ttvnw.net/api/channel/hls/${channel}.m3u8` +
      `?sig=${sig}` +
      `&token=${encodedToken}` +
      `&allow_source=true` +
      `&allow_audio_only=true`
    );
  }

  // ===== Voice capture (chunked WAV, then STT) =====
  public async startVoiceCapture(channel: string): Promise<void> {
    if (this.isCapturing) return;

    this.isCapturing = true;
    this.currentChannelInfo = await this.getChannelInfo(channel);

    // Load memory facts
    try {
      const mem = await this.memoryStore.load(this.channelIdForMemory);
      this.facts = mem.facts || [];
      logger.info(`Loaded ${this.facts.length} memory facts for channel ${channel}`);
    } catch (e) {
      logger.warn('Failed to load memory facts:', e);
      this.facts = [];
    }

    const durationMs = parseInt(process.env.TRANSCRIPT_DURATION || '20000', 10);
    const output = join(tmpdir(), `twitch_audio_${Date.now()}.wav`);

    this.currentHlsUrl = await this.getTwitchHlsUrl(channel);

    logger.info(`Voice capture chunk: 640000 bytes (~${(durationMs / 1000).toFixed(1)}s)`);

    while (this.isCapturing) {
      try {
        this.tempAudioFile = output;

        await new Promise<void>((resolve) => {
          let isProcessing = false;

          this.currentProcess = ffmpeg()
            .input(this.currentHlsUrl!)
            .inputOptions(
              '-reconnect', '1',
              '-reconnect_streamed', '1',
              '-reconnect_delay_max', '5',
              '-user_agent', 'Mozilla/5.0'
            )
            .audioCodec('pcm_s16le')
            .audioChannels(1)
            .audioFrequency(16000)
            .format('wav')
            .duration(durationMs / 1000)
            .on('start', () => logger.info('Spawning ffmpeg for live PCM capture...'))
            .on('end', async () => {
              if (!isProcessing) {
                isProcessing = true;
                try {
                  await this.processAudioChunk();
                } finally {
                  isProcessing = false;
                }
                resolve();
              } else {
                resolve();
              }
            })
            .on('error', async (err) => {
              logger.error('ffmpeg error:', err);
              await new Promise(r => setTimeout(r, 2000));
              resolve();
            })
            .save(this.tempAudioFile!);
        });

        await new Promise(resolve => setTimeout(resolve, 200));
      } catch (error) {
        logger.error('Error in capture loop:', error);
        this.stopVoiceCapture();
      }
    }
  }

  public stopVoiceCapture(): void {
    if (!this.isCapturing) return;

    if (this.currentProcess) {
      this.currentProcess.kill('SIGKILL');
      this.currentProcess = null;
    }

    if (this.tempAudioFile) {
      try { unlinkSync(this.tempAudioFile); } catch { /* noop */ }
      this.tempAudioFile = null;
    }

    this.isCapturing = false;
    logger.info('Voice capture stopped');
  }

  private readAudioFile(filePath: string): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const fs = require('fs');
      fs.readFile(filePath, (err: Error | null, data: Buffer) => {
        if (err) reject(err);
        else resolve(data);
      });
    });
  }

  private wavToLpcm(wav: Buffer): Buffer {
    // SpeechKit sync: format=lpcm expects raw PCM without WAV header
    if (wav.length < 12) return wav;

    const riff = wav.toString('ascii', 0, 4);
    const wave = wav.toString('ascii', 8, 12);
    if (riff !== 'RIFF' || wave !== 'WAVE') return wav;

    let offset = 12;
    while (offset + 8 <= wav.length) {
      const chunkId = wav.toString('ascii', offset, offset + 4);
      const chunkSize = wav.readUInt32LE(offset + 4);
      const dataStart = offset + 8;

      if (chunkId === 'data') {
        const dataEnd = Math.min(dataStart + chunkSize, wav.length);
        return wav.slice(dataStart, dataEnd);
      }

      offset = dataStart + chunkSize + (chunkSize % 2);
    }

    return wav;
  }

  public async processAudioToText(audioData: Buffer): Promise<string> {
    const lang = process.env.YANDEX_STT_LANG || (process.env.ORIGINAL_STREAM_LANGUAGE === 'ru' ? 'ru-RU' : 'en-US');

    if (!this.yandexApiKey) throw new Error('YANDEX_API_KEY is missing');
    if (!this.yandexFolderId) throw new Error('YANDEX_FOLDER_ID is missing');

    const lpcm = this.wavToLpcm(audioData);

    const url = 'https://stt.api.cloud.yandex.net/speech/v1/stt:recognize';
    const r = await axios.post(url, lpcm, {
      params: {
        folderId: this.yandexFolderId,
        lang,
        format: 'lpcm',
        sampleRateHertz: 16000
      },
      headers: {
        Authorization: `Api-Key ${this.yandexApiKey}`,
        'Content-Type': 'application/octet-stream'
      },
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      timeout: 30000
    });

    return (r.data?.result || '').toString().trim();
  }

  // ===== Screen OCR (frame from HLS + Yandex Vision OCR) =====
  private async captureFramePng(hlsUrl: string, timeoutMs = 15000): Promise<Buffer> {
    return await new Promise<Buffer>((resolve, reject) => {
      const args = [
        '-user_agent', 'Mozilla/5.0',
        '-loglevel', 'error',
        '-y',
        '-i', hlsUrl,
        '-frames:v', '1',
        '-vf', 'scale=1280:-1',
        '-f', 'image2pipe',
        '-vcodec', 'png',
        'pipe:1'
      ];

      const proc = spawn(ffmpegInstaller.path, args, { stdio: ['ignore', 'pipe', 'pipe'] });

      const chunks: Buffer[] = [];
      const errChunks: Buffer[] = [];

      const timer = setTimeout(() => {
        try { proc.kill('SIGKILL'); } catch { /* noop */ }
        reject(new Error(`ffmpeg screenshot timeout after ${timeoutMs}ms`));
      }, timeoutMs);

      proc.stdout.on('data', (d: Buffer) => chunks.push(d));
      proc.stderr.on('data', (d: Buffer) => errChunks.push(d));

      proc.on('error', (e) => {
        clearTimeout(timer);
        reject(e);
      });

      proc.on('close', (code) => {
        clearTimeout(timer);
        if (code !== 0) {
          const stderr = Buffer.concat(errChunks).toString('utf8').trim();
          return reject(new Error(`ffmpeg screenshot failed (code=${code}): ${stderr}`));
        }
        resolve(Buffer.concat(chunks));
      });
    });
  }

  private async ocrRecognize(png: Buffer): Promise<string> {
    if (!this.yandexApiKey) throw new Error('YANDEX_API_KEY is missing');
    if (!this.yandexFolderId) throw new Error('YANDEX_FOLDER_ID is missing');

    const url = 'https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText';
    const body = {
      mimeType: 'PNG',
      languageCodes: ['ru', 'en'],
      model: 'page',
      content: png.toString('base64')
    };

    const r = await axios.post(url, body, {
      headers: {
        Authorization: `Api-Key ${this.yandexApiKey}`,
        'x-folder-id': this.yandexFolderId,
        'Content-Type': 'application/json'
      },
      timeout: 30000
    });

    const fullText = r.data?.textAnnotation?.fullText;
    return String(fullText || '').trim();
  }

  private async maybeUpdateScreenFromStream(triggerHint: string): Promise<void> {
    if (!this.ocrEnabled) return;
    if (!this.currentHlsUrl) return;

    const now = Date.now();
    if (now - this.lastOcrTs < this.ocrIntervalMs) return;

    this.lastOcrTs = now;

    try {
      const png = await this.captureFramePng(this.currentHlsUrl);
      const text = await this.ocrRecognize(png);
      const cleaned = normalizeMessage(text);

      if (cleaned && cleaned !== this.lastScreenText) {
        this.lastScreenText = cleaned;
        this.recordEvent({ ts: now, type: 'screen', text: cleaned, meta: { source: 'ocr', triggerHint } });
        await this.maybeUpdateFacts('обновление экрана');
      }
    } catch (e) {
      logger.debug('Screen OCR failed (non-fatal):', e);
    }
  }

  // ===== AI prompt building =====
  private buildUserPrompt(parsedContext: any): string {
    const channelContext = `
Название стрима: ${this.currentChannelInfo?.title}
Категория: ${this.currentChannelInfo?.gameName}
Зрителей: ${this.currentChannelInfo?.viewerCount}
`.trim();

    const lastTranscription = parsedContext.lastTranscription
      ? `Стример только что сказал: "${parsedContext.lastTranscription}"`
      : '';

    const screenBlock = this.lastScreenText
      ? `Текст/подсказки на экране (OCR): ${this.lastScreenText.slice(0, 500)}`
      : '';

    const recentChat = (this.shortEvents || [])
      .filter(e => e.type === 'chat')
      .slice(-8)
      .map(e => `- ${e.text}`)
      .join('\n');

    const recentSpeech = (this.shortEvents || [])
      .filter(e => e.type === 'speech')
      .slice(-6)
      .map(e => `- ${e.text}`)
      .join('\n');

    const lastChatLine = parsedContext.chatMessage ? `Последнее сообщение чата: "${parsedContext.chatMessage}"` : '';

    return `
${channelContext}

${screenBlock}

Последние сообщения чата:
${recentChat || '- (пока нет)'}

Последние фразы стримера:
${recentSpeech || '- (пока нет)'}

${lastChatLine}
${lastTranscription}

Напиши ОДНО сообщение в чат, максимально естественно.
Если есть вопрос/обращение в чате — ответь на него.
Иначе коротко отреагируй на контекст.
Если реально нечего сказать — верни пустую строку.
`.trim();
  }

  public async generateMessage(context?: string): Promise<string> {
    try {
      if (!this.currentChannelInfo) return '';

      let parsedContext: any = {};
      try {
        if (context) parsedContext = JSON.parse(context);
      } catch {
        parsedContext = { rawText: context };
      }

      const prompt = this.buildUserPrompt(parsedContext);

      const queryText =
        (parsedContext.lastTranscription || '') + ' ' +
        (parsedContext.chatMessage || '') + ' ' +
        (this.lastScreenText || '') + ' ' +
        (this.currentChannelInfo.gameName || '');

      const relevantFacts = this.memoryStore.pickRelevantFacts(this.facts || [], queryText, 8);
      const factsBlock = relevantFacts.length
        ? `Память (важные факты о стриме/чате):\n- ${relevantFacts.map(f => f.text).join('\n- ')}`
        : '';

      const url = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion';
      const modelName = process.env.YANDEX_GPT_MODEL || 'yandexgpt-lite';
      const modelUri = `gpt://${this.yandexFolderId}/${modelName}`;

      const messages: { role: 'system' | 'user' | 'assistant'; text: string }[] = [
        { role: 'system', text: RU_SYSTEM_PROMPT },
      ];

      if (factsBlock) messages.push({ role: 'system', text: factsBlock });

      const recentTurns = this.shortHistory.slice(-16).map(t => ({ role: t.role, text: t.text }));
      messages.push(...recentTurns);

      messages.push({ role: 'user', text: prompt });

      const body = {
        modelUri,
        completionOptions: {
          stream: false,
          temperature: 0.7,
          maxTokens: 90
        },
        messages
      };

      const r = await axios.post(url, body, {
        headers: {
          Authorization: `Api-Key ${this.yandexApiKey}`,
          'Content-Type': 'application/json'
        },
        timeout: 30000
      });

      let out = r.data?.result?.alternatives?.[0]?.message?.text?.trim?.() || '';
      out = normalizeMessage(out);

      // If model ignored instruction and returned markdown / something else, keep only first line
      out = out.split('\n').map((x: string) => x.trim()).filter(Boolean)[0] || '';


      // soften emoji spam
      if (tooManyEmojis(out)) out = reduceEmojiSpam(out);

      // hard guard: prevent accidental long messages
      if (out.length > 160) out = out.slice(0, 160).trim();

      // guard against "чел" spam
      const forceNoChel = (process.env.NO_CHEL || '').trim() === '1';
      if (forceNoChel) out = out.replace(/\bчел\b/gi, '').replace(/\s+/g, ' ').trim();

      return out;
    } catch (error) {
      logger.error('Ошибка генерации сообщения:', error);
      return '';
    }
  }

  // ===== Memory update =====
  private async maybeUpdateFacts(triggerHint: string) {
    const now = Date.now();
    const should =
      (now - this.lastFactsUpdateTs > 120_000 && this.shortEvents.length >= 8) ||
      (this.shortEvents.length >= 18);

    if (!should) return;
    if (!this.channelIdForMemory) return;

    const events = this.shortEvents.splice(0);
    this.lastFactsUpdateTs = now;

    const modelName = process.env.YANDEX_GPT_MODEL || 'yandexgpt-lite';
    const modelUri = `gpt://${this.yandexFolderId}/${modelName}`;
    const url = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion';

    const eventsText = events
      .slice(-24)
      .map(e => {
        const tag = e.type === 'speech' ? 'SPEECH' : e.type === 'chat' ? 'CHAT' : 'SCREEN';
        return `[${tag}] ${e.text}`;
      })
      .join('\n');

    const existing = (this.facts || [])
      .slice(-40)
      .map(f => `- (${f.importance}) ${f.text}`)
      .join('\n');

    const system = `
Ты — модуль памяти для Twitch-стрима.
Из новых событий извлеки КОРОТКИЕ факты, которые пригодятся для будущих реплик.
Факт — 1 строка, без домыслов.
Выход СТРОГО JSON (без markdown и без пояснений):
{"facts":[{"text":"...", "importance":1-5, "tags":["..."]}]}
`.trim();

    const body = {
      modelUri,
      completionOptions: { stream: false, temperature: 0.2, maxTokens: 260 },
      messages: [
        { role: 'system', text: system },
        { role: 'user', text: `Существующие факты:\n${existing || '(нет)'}\n\nНовые события (${triggerHint}):\n${eventsText}` }
      ]
    };

    try {
      const r = await axios.post(url, body, {
        headers: { Authorization: `Api-Key ${this.yandexApiKey}`, 'Content-Type': 'application/json' },
        timeout: 30000
      });

      const raw = r.data?.result?.alternatives?.[0]?.message?.text || '';
      const json = extractJsonObject(raw);
      if (!json) throw new Error(`Invalid JSON from model: ${String(raw).slice(0, 200)}`);

      const facts = Array.isArray(json?.facts) ? json.facts : [];

      const normalized: Omit<MemoryFact, 'id' | 'ts'>[] = facts
        .map((f: any) => ({
          text: String(f?.text || '').trim(),
          importance: clamp(Number(f?.importance || 3), 1, 5),
          tags: Array.isArray(f?.tags) ? f.tags.map((x: any) => String(x)) : []
        }))
        .filter((f: { text: string }) => f.text.length >= 3);

      if (normalized.length) {
        this.facts = mergeFactsSimple(this.facts, normalized);
        await this.memoryStore.save({
          channel: this.channelIdForMemory,
          updatedAt: Date.now(),
          facts: this.facts
        });

        logger.info(`Memory updated: +${normalized.length} facts`);
      }
    } catch (e) {
      logger.warn('Failed to update facts:', e);
    }
  }

  // ===== Main audio chunk processing =====
  private async processAudioChunk(): Promise<void> {
    if (!this.tempAudioFile || this.isProcessing) return;

    this.isProcessing = true;

    try {
      if (!existsSync(this.tempAudioFile)) return;

      const wav = await this.readAudioFile(this.tempAudioFile);
      if (!Buffer.isBuffer(wav) || wav.length === 0) return;

      logger.info('Audio bytes:', wav.length);

      // stats for VAD & emotion
      const stats = analyzePcm16le(wav);
      logger.info(`PCM stats: samples=${stats.samples} rms=${stats.rms.toFixed(6)} peak=${stats.peak.toFixed(6)}`);

      if (stats.rms < this.aiRmsMin) {
        logger.info(`RMS below threshold (${this.aiRmsMin}), skip chunk`);
        return;
      }

      const md5 = crypto.createHash('md5').update(wav).digest('hex');
      logger.info(`PCM md5: ${md5}`);

      const transcribedText = await this.processAudioToText(wav);
      const t = normalizeMessage(transcribedText);

      logger.info('STT raw:', JSON.stringify(t));

      // empty STT must NOT poison duplicate logic and must not stop loop
      if (!t) return;

      // dedupe only for non-empty
      if (t === this.lastNonEmptyTranscription) {
        logger.info('Skipping duplicate transcription');
        return;
      }
      this.lastNonEmptyTranscription = t;

      // memory: streamer speech event
      this.recordEvent({ ts: Date.now(), type: 'speech', text: t });
      await this.maybeUpdateFacts('новая речь стримера');

      // update screen context sometimes (OCR)
      await this.maybeUpdateScreenFromStream('audio_chunk');

      const now = Date.now();

      // trigger logic (only respond when there is a reason)
      const trig = this.shouldRespond(now, t, stats.rms);
      if (!trig.ok) {
        return;
      }

      // rate limit
      if (!this.canCallAi(now)) {
        const wait = Math.max(0, this.nextAllowedAiTs - now);
        logger.info(`AI rate limit active, next allowed in ${Math.ceil(wait / 1000)}s`);
        return;
      }

      // include last chat line if any
      const lastChat = this.lastChatMessage();
      const ctx = {
        lastTranscription: t,
        chatMessage: lastChat ? `${lastChat.user}: ${lastChat.text}` : '',
        trigger: trig.reason,
        isStreamerMessage: true
      };

      const message = await this.generateMessage(JSON.stringify(ctx));
      const out = normalizeMessage(message);

      if (!out) return;

      logger.info('Generated message:', out);

      // update next allowed immediately, before emitting
      this.nextAllowedAiTs = this.nextAllowedAi(now);

      this.emit('message', out);
      this.pushAssistantTurn(out);
    } catch (error) {
      logger.error('Error processing audio chunk:', error);
    } finally {
      try {
        if (this.tempAudioFile && existsSync(this.tempAudioFile)) unlinkSync(this.tempAudioFile);
      } catch (error) {
        logger.error('Error cleaning up temporary file:', error);
      }
      this.tempAudioFile = null;
      this.isProcessing = false;
    }
  }
}
