import axios from 'axios';
import { EventEmitter } from 'events';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';
import { unlinkSync, existsSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { FfmpegCommand } from 'fluent-ffmpeg';
import { logger } from './logger';
import { analyzePcm16le } from "./audioDebug";
import crypto from "crypto";

import { MemoryStore, MemoryEvent, MemoryFact } from './memory';

ffmpeg.setFfmpegPath(ffmpegInstaller.path);

interface TwitchChannelInfo {
  title: string;
  description: string;
  gameName: string;
  viewerCount: number;
  isLive: boolean;
}

// Более “человеческий” системный промпт (без навязанного "чел")
const RU_SYSTEM_PROMPT = `
Ты — обычный зритель Twitch в чате.

Пиши коротко и живо, как человек.
Главное — попадать в контекст (вопросы чата и что сказал стример).

Ограничения:
- НЕ упоминай ИИ/модели/алгоритмы
- Обычно 15–70 символов
- Без спама и повторов
- "чел" НЕ используй (только если прям уместно, но лучше избегать)
- Эмодзи редко (10–15%)
- Без ссылок/команд/капса

Формат: выдай ОДНО сообщение для чата, без пояснений.
`.trim();

function makeFactId() {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function mergeFacts(existing: MemoryFact[], incoming: MemoryFact[]): MemoryFact[] {
  const map = new Map<string, MemoryFact>();

  for (const f of existing || []) {
    const key = (f.text || '').trim().toLowerCase();
    if (!key) continue;
    map.set(key, f);
  }

  for (const f of incoming || []) {
    const key = (f.text || '').trim().toLowerCase();
    if (!key) continue;

    const prev = map.get(key);
    if (!prev) {
      map.set(key, {
        ...f,
        id: f.id || makeFactId(),
        ts: f.ts || Date.now()
      });
    } else {
      // обновляем “подтверждение” факта
      map.set(key, {
        ...prev,
        ts: Date.now(),
        importance: Math.max(prev.importance || 1, f.importance || 1),
        tags: Array.from(new Set([...(prev.tags || []), ...(f.tags || [])]))
      });
    }
  }

  // можно ограничить размер памяти
  return Array.from(map.values()).sort((a, b) => (b.importance - a.importance) || (b.ts - a.ts)).slice(0, 200);
}


export class AIService extends EventEmitter {
  private isCapturing: boolean = false;
  private currentProcess: FfmpegCommand | null = null;
  private tempAudioFile: string | null = null;
  private accessToken: string | null = null;
  private _currentChannelInfo: TwitchChannelInfo | null = null;

  // старое MESSAGE_INTERVAL больше не используем для частоты отправки
  private isProcessing: boolean = false;
  private processingLock: boolean = false;

  // дедуп речи
  private lastTranscriptionHash: string = '';
  private lastProcessedTime: number = 0;

  // rate limit (AI)
  private aiMinIntervalMs: number;
  private aiJitter: number;
  private aiRmsMin: number;
  private nextAllowedMessageTs: number = 0;

  private yandexApiKey: string;
  private yandexFolderId: string;

  // ===== Memory =====
  private memoryStore = new MemoryStore();
  private channelIdForMemory: string = '';
  private facts: MemoryFact[] = [];
  private lastFactsUpdateTs = 0;

  // short-term timeline (speech/chat/screen + bot replies)
  private shortHistory: { role: 'user' | 'assistant'; text: string; ts: number }[] = [];
  private shortEvents: MemoryEvent[] = [];

  private wavToLpcm(wav: Buffer): Buffer {
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

  constructor() {
    super();
    logger.info('AIService initialized');

    this.yandexApiKey = process.env.YANDEX_API_KEY || '';
    this.yandexFolderId = process.env.YANDEX_FOLDER_ID || '';

    // антиспам по умолчанию
    this.aiMinIntervalMs = parseInt(process.env.AI_MIN_INTERVAL_MS || '90000', 10); // 90s
    this.aiJitter = parseFloat(process.env.AI_JITTER || '0.25'); // 25%
    this.aiRmsMin = parseFloat(process.env.AI_RMS_MIN || '0.002'); // гейт по громкости

    if (!this.yandexApiKey || !this.yandexFolderId) {
      logger.warn('Yandex credentials are missing: set YANDEX_API_KEY and YANDEX_FOLDER_ID');
    } else {
      logger.info('Yandex client configured');
    }

    // Входящий чат-контекст (его шлёт bot.ts)
    this.on('chatMessage', (payload: string) => {
      try {
        const ctx = JSON.parse(payload);
        const user = (ctx.username || 'зритель').toString();
        const msg = (ctx.chatMessage || '').toString().trim();
        if (!msg) return;

        this.recordEvent({
          ts: Date.now(),
          type: 'chat',
          text: `${user}: ${msg}`,
          meta: { user }
        });
      } catch (e) {
        logger.error('Failed to parse chatMessage payload:', e);
      }
    });
  }

  public get currentChannelInfo(): TwitchChannelInfo | null {
    return this._currentChannelInfo;
  }

  private set currentChannelInfo(info: TwitchChannelInfo | null) {
    this._currentChannelInfo = info;
  }

  public get isVoiceCaptureActive(): boolean {
    return this.isCapturing;
  }

  // ===== Screen context hook (later OCR / scene description) =====
  public recordScreenContext(text: string, meta?: Record<string, any>) {
    const t = (text || '').trim();
    if (!t) return;
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
    if (this.shortEvents.length > 50) this.shortEvents = this.shortEvents.slice(-50);
  }

  private pushAssistantTurn(text: string) {
    const t = (text || '').trim();
    if (!t) return;
    this.shortHistory.push({ role: 'assistant', text: t, ts: Date.now() });
    if (this.shortHistory.length > 30) this.shortHistory = this.shortHistory.slice(-30);
  }

  private calcNextAllowedTs(now: number): number {
    const j = this.aiJitter;
    const base = this.aiMinIntervalMs;
    const mult = (1 - j) + Math.random() * (2 * j); // 0.75..1.25
    return now + Math.floor(base * mult);
  }

  private async generateAccessToken(): Promise<string> {
    try {
      logger.info('Generating new access token...');
      const response = await axios.post('https://id.twitch.tv/oauth2/token', null, {
        params: {
          client_id: process.env.TWITCH_CLIENT_ID,
          client_secret: process.env.TWITCH_CLIENT_SECRET,
          grant_type: 'client_credentials',
          scope: 'user:read:email channel:read:stream_key channel:manage:broadcast'
        }
      });

      const { access_token } = response.data;
      logger.info('New access token generated successfully');
      return access_token;
    } catch (error) {
      logger.error('Error generating access token:', error);
      throw error;
    }
  }

  private async getChannelInfo(channelName: string): Promise<TwitchChannelInfo> {
    try {
      logger.info('Fetching channel info for:', channelName);

      if (!this.accessToken) {
        this.accessToken = await this.generateAccessToken();
      }

      const userResponse = await axios.get(`https://api.twitch.tv/helix/users?login=${channelName}`, {
        headers: {
          'Client-ID': process.env.TWITCH_CLIENT_ID,
          Authorization: `Bearer ${this.accessToken}`
        }
      });

      const userData = userResponse.data.data[0];
      if (!userData) {
        throw new Error(`Channel not found: ${channelName}`);
      }

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
    } catch (error) {
      logger.error('Error fetching channel info:', error);
      throw error;
    }
  }

  private async getTwitchHlsUrl(channel: string): Promise<string> {
    // Web client-id (публичный, используется Twitch web-плеером)
    const TWITCH_WEB_CLIENT_ID =
      process.env.TWITCH_WEB_CLIENT_ID || 'kimne78kx3ncx6brgo4mv6wki5h1ko';

    // Часто без Bearer токена Twitch возвращает null — поэтому используем app access token
    if (!this.accessToken) {
      this.accessToken = await this.generateAccessToken();
    }

    const body = [
      {
        operationName: 'PlaybackAccessToken',
        variables: {
          login: channel,
          playerType: 'site'
        },
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
        // ВАЖНО: добавляем Bearer
        Authorization: `Bearer ${this.accessToken}`,
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0',
        Accept: 'application/json'
      },
      timeout: 15000
    });

    // Twitch может вернуть массив (обычно) или объект — делаем устойчиво
    const payload = Array.isArray(r.data) ? r.data[0] : r.data;

    if (payload?.errors?.length) {
      // покажем реальную причину
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
    const hlsUrl = await this.getTwitchHlsUrl(channel);

    logger.info(`Voice capture chunk: 640000 bytes (~${(durationMs / 1000).toFixed(1)}s)`);

    while (this.isCapturing) {
      try {
        this.tempAudioFile = output;

        await new Promise<void>((resolve) => {
          let isProcessing = false;

          this.currentProcess = ffmpeg()
            .input(hlsUrl)
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
            .on('start', () => {
              logger.info('Spawning ffmpeg for live PCM capture...');
            })
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
                isProcessing = false;
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

        await new Promise(resolve => setTimeout(resolve, 200)); // маленькая пауза
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
      try { unlinkSync(this.tempAudioFile); } catch { }
      this.tempAudioFile = null;
    }

    this.isCapturing = false;
    logger.info('Voice capture stopped');
  }

  private async processAudioChunk(): Promise<void> {
    if (!this.tempAudioFile || this.isProcessing || this.processingLock) {
      return;
    }

    this.processingLock = true;
    this.isProcessing = true;

    try {
      if (!existsSync(this.tempAudioFile)) return;

      const audioData = await this.readAudioFile(this.tempAudioFile);
      if (!Buffer.isBuffer(audioData) || audioData.length === 0) return;

      logger.info('Audio bytes:', audioData.length);

      // stats/диагностика
      const stats = analyzePcm16le(audioData);
      logger.info(`PCM stats: samples=${stats.samples} rms=${stats.rms.toFixed(6)} peak=${stats.peak.toFixed(6)}`);

      // VAD-гейт: если тишина/шум — не реагируем
      if (stats.rms < this.aiRmsMin) {
        logger.info(`RMS below threshold (${this.aiRmsMin}), skip chunk`);
        return;
      }

      const md5 = crypto.createHash("md5").update(audioData).digest("hex");
      logger.info(`PCM md5: ${md5}`);

      const transcribedText = await this.processAudioToText(audioData);
      const t = (transcribedText || '').trim();

      logger.info('STT raw:', JSON.stringify(t));

      // пустота — НЕ считается дублем и НЕ должна “глушить цикл”
      if (!t) {
        return;
      }

      const now = Date.now();

      // дедуп речи (чтобы не молотить одно и то же)
      if (t === this.lastTranscriptionHash && (now - this.lastProcessedTime) < 15000) {
        logger.info('Skipping duplicate transcription (non-empty)');
        return;
      }

      this.lastTranscriptionHash = t;
      this.lastProcessedTime = now;

      // память: речь стримера
      this.recordEvent({ ts: Date.now(), type: 'speech', text: t });
      await this.maybeUpdateFacts('новая речь стримера');

      // rate-limit: не чаще, чем AI_MIN_INTERVAL_MS (+ jitter)
      if (now < this.nextAllowedMessageTs) {
        logger.info(`AI rate limit active, next allowed in ${Math.ceil((this.nextAllowedMessageTs - now) / 1000)}s`);
        return;
      }

      // вытаскиваем последний чат (особенно вопрос)
      const recentChat = [...this.shortEvents]
        .reverse()
        .find(e => e.type === 'chat' && (now - e.ts) < 35000); // 35 сек

      const recentChatText = recentChat?.text || '';
      const hasQuestion = recentChatText.includes('?');

      // если чата нет и речь слишком “ни о чём” — лучше промолчать
      if (!hasQuestion && t.length < 16) {
        logger.info('Transcription too short and no chat question -> skip');
        return;
      }

      const msg = await this.generateMessage(JSON.stringify({
        lastTranscription: t,
        chatMessage: recentChatText || undefined
      }));

      const message = (msg || '').trim();
      if (message) {
        logger.info('Generated message:', message);
        this.emit('message', message);
        this.pushAssistantTurn(message);

        // после отправки — назначаем следующий “слот”
        this.nextAllowedMessageTs = this.calcNextAllowedTs(Date.now());
      }
    } catch (error) {
      logger.error('Error processing audio chunk:', error);
    } finally {
      try {
        if (this.tempAudioFile && existsSync(this.tempAudioFile)) unlinkSync(this.tempAudioFile);
      } catch { }
      this.tempAudioFile = null;
      this.isProcessing = false;
      this.processingLock = false;
    }
  }

  private readAudioFile(filePath: string): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      const fs = require('fs');
      fs.readFile(filePath, (err: Error | null, data: Buffer) => {
        if (err) reject(err);
        else resolve(data);
      });
    });
  }

  public async processAudioToText(audioData: Buffer): Promise<string> {
    try {
      const lang = process.env.YANDEX_STT_LANG || (process.env.ORIGINAL_STREAM_LANGUAGE === 'ru' ? 'ru-RU' : 'en-US');

      if (!this.yandexApiKey) throw new Error('YANDEX_API_KEY is missing');
      if (!this.yandexFolderId) throw new Error('YANDEX_FOLDER_ID is missing');

      const lpcm = this.wavToLpcm(audioData);
      const url = 'https://stt.api.cloud.yandex.net/speech/v1/stt:recognize';

      const r = await axios.post(url, lpcm, {
        params: {
          lang,
          format: 'lpcm',
          sampleRateHertz: 16000
        },
        headers: {
          Authorization: `Api-Key ${this.yandexApiKey}`,
          'Content-Type': 'application/octet-stream'
        },
        maxBodyLength: Infinity,
        maxContentLength: Infinity
      });

      return (r.data?.result || '').toString().trim();
    } catch (error) {
      logger.error('Error processing audio to text with Yandex SpeechKit:', error);
      throw error;
    }
  }

  private buildUserPrompt(parsedContext: any): string {
    const channelContext = `
Название стрима: ${this.currentChannelInfo?.title}
Игра: ${this.currentChannelInfo?.gameName}
Зрителей: ${this.currentChannelInfo?.viewerCount}
`.trim();

    const lastTranscription = parsedContext.lastTranscription
      ? `Стример только что сказал: "${parsedContext.lastTranscription}"`
      : '';

    const recentChat = (this.shortEvents || [])
      .filter(e => e.type === 'chat' && !String(e.text || '').startsWith('bot:'))
      .slice(-6)
      .map(e => `- ${e.text}`)
      .join('\n');

    const recentSpeech = (this.shortEvents || [])
      .filter(e => e.type === 'speech')
      .slice(-4)
      .map(e => `- ${e.text}`)
      .join('\n');

    const lastChatLine = parsedContext.chatMessage ? `Последнее сообщение чата: "${parsedContext.chatMessage}"` : '';

    return `
${channelContext}

Последние сообщения чата:
${recentChat || '- (пока нет)'}

Последняя речь стримера:
${recentSpeech || '- (пока нет)'}

${lastChatLine}
${lastTranscription}

Задача:
- Если в чате есть вопрос (в последних строках есть '?') — ответь на него коротко и по делу.
- Иначе коротко отреагируй на последнюю фразу стримера (естественно, без цитирования).
- Одно сообщение. Без слова "чел".
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

      // если совсем нет повода — молчим
      if (!parsedContext.lastTranscription && !parsedContext.chatMessage) return '';

      const prompt = this.buildUserPrompt(parsedContext);

      const queryText =
        (parsedContext.lastTranscription || '') + ' ' +
        (parsedContext.chatMessage || '') + ' ' +
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

      // short history (последние события + ответы)
      const recentTurns = this.shortHistory.slice(-12).map(t => ({ role: t.role, text: t.text }));
      messages.push(...recentTurns);

      messages.push({ role: 'user', text: prompt });

      const body = {
        modelUri,
        completionOptions: {
          stream: false,
          temperature: 0.75,
          maxTokens: 80
        },
        messages
      };

      const r = await axios.post(url, body, {
        headers: {
          Authorization: `Api-Key ${this.yandexApiKey}`,
          'Content-Type': 'application/json'
        }
      });

      return r.data?.result?.alternatives?.[0]?.message?.text?.trim?.() || '';
    } catch (error) {
      logger.error('Ошибка генерации сообщения:', error);
      return '';
    }
  }

  private async maybeUpdateFacts(triggerHint: string) {
    const now = Date.now();
    const should =
      (now - this.lastFactsUpdateTs > 120_000 && this.shortEvents.length >= 6) ||
      (this.shortEvents.length >= 15);

    if (!should) return;
    if (!this.channelIdForMemory) return;

    const events = this.shortEvents.splice(0);
    this.lastFactsUpdateTs = now;

    const modelName = process.env.YANDEX_GPT_MODEL || 'yandexgpt-lite';
    const modelUri = `gpt://${this.yandexFolderId}/${modelName}`;
    const url = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion';

    const eventsText = events
      .slice(-20)
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
Выход СТРОГО JSON без markdown:
{"facts":[{"text":"...", "importance":1-5, "tags":["..."]}]}
`.trim();

    const body = {
      modelUri,
      completionOptions: { stream: false, temperature: 0.2, maxTokens: 250 },
      messages: [
        { role: 'system', text: system },
        { role: 'user', text: `Существующие факты:\n${existing || '(нет)'}\n\nНовые события (${triggerHint}):\n${eventsText}` }
      ]
    };

    try {
      const r = await axios.post(url, body, {
        headers: { Authorization: `Api-Key ${this.yandexApiKey}`, 'Content-Type': 'application/json' }
      });

      const raw = r.data?.result?.alternatives?.[0]?.message?.text || '';
      const json = JSON.parse(raw);
      const facts = Array.isArray(json?.facts) ? json.facts : [];

      const normalized: MemoryFact[] = facts
        .map((f: any) => ({
          text: String(f.text || '').trim(),
          importance: Math.max(1, Math.min(5, Number(f.importance || 3))),
          tags: Array.isArray(f.tags) ? f.tags.map((x: any) => String(x)) : []
        }))
        .filter((f: MemoryFact) => f.text.length >= 3);

      if (normalized.length) {
        this.facts = mergeFacts(this.facts, normalized);
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
}
