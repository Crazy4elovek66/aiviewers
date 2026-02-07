import tmi from 'tmi.js';
import { AIService } from './ai';
import { logger } from './logger';

interface BotConfig {
  username: string;
  oauth: string;
  channel: string;
  aiService: AIService;
  shouldHandleVoiceCapture?: boolean;
}

function normalizeChannel(input: string): string {
  let c = (input || '').trim();
  if (c.includes('twitch.tv/')) {
    c = (c.split('twitch.tv/')[1] || c).split('/')[0].split('?')[0];
  }
  if (c.startsWith('#')) c = c.slice(1);
  return c.trim();
}

export class Bot {
  private client: tmi.Client;
  private aiService: AIService;

  private readonly channelName: string;
  private isConnected = false;

  private readonly minSendIntervalMs = parseInt(process.env.BOT_MIN_SEND_INTERVAL_MS || '90000', 10); // 90s
  private readonly jitter = parseFloat(process.env.BOT_JITTER || '0.25'); // 25%
  private lastSendTs = 0;

  private sending = false;
  private queue: string[] = [];

  private pendingEcho:
    | { msg: string; timeout: NodeJS.Timeout; resolve: () => void; reject: (e: Error) => void }
    | null = null;

  constructor(config: BotConfig) {
    this.aiService = config.aiService;
    this.channelName = normalizeChannel(config.channel);

    if (!config.oauth.startsWith('oauth:')) {
      throw new Error(`Invalid OAuth token format. Must start with oauth:`);
    }

    this.client = new tmi.Client({
      options: { debug: false, messagesLogLevel: 'info' },
      identity: { username: config.username, password: config.oauth },
      channels: [this.channelName],
      connection: { reconnect: true, secure: true }
    });

    this.setupHandlers();

    if (config.shouldHandleVoiceCapture) {
      this.aiService.startVoiceCapture(this.channelName).catch(err => logger.error('Voice capture error:', err));
    }
  }

  private setupHandlers(): void {
    this.client.on('message', (channel, tags, message, self) => {
      if (self) {
        if (this.pendingEcho && message === this.pendingEcho.msg) {
          clearTimeout(this.pendingEcho.timeout);
          this.pendingEcho.resolve();
          this.pendingEcho = null;
        }
        return;
      }

      // ВАЖНО: передаём в AIService только username + текст
      // (AIService сам это пишет в shortEvents для контекста)
      try {
        this.aiService.emit('chatMessage', JSON.stringify({
          username: tags['display-name'] || tags.username,
          chatMessage: message
        }));
      } catch (e) {
        logger.error('chatMessage emit failed:', e);
      }
    });

    this.client.on('connected', (addr, port) => {
      this.isConnected = true;
      logger.info(`Bot connected to ${addr}:${port}`);
    });

    this.client.on('disconnected', (reason) => {
      this.isConnected = false;
      logger.warn(`Bot disconnected: ${reason}`);

      if (this.pendingEcho) {
        clearTimeout(this.pendingEcho.timeout);
        this.pendingEcho.reject(new Error('Disconnected before echo'));
        this.pendingEcho = null;
      }

      this.sending = false;
    });

    this.client.on('join', (channel, username, self) => {
      if (self) logger.info(`Joined ${channel} as ${username}`);
    });

    // AI → очередь
    this.aiService.on('message', (msg: string) => {
      const m = (msg || '').trim();
      if (!m) return;

      // защита от дублей подряд
      const last = this.queue.length ? this.queue[this.queue.length - 1] : '';
      if (m === last) return;

      this.queue.push(m);
      this.flushQueue().catch(e => logger.error('flushQueue error:', e));
    });
  }

  private calcMinIntervalWithJitter(): number {
    const base = this.minSendIntervalMs;
    const j = this.jitter;
    return Math.floor(base * ((1 - j) + Math.random() * (2 * j)));
  }

  private async flushQueue(): Promise<void> {
    if (this.sending) return;
    this.sending = true;

    try {
      while (this.queue.length) {
        if (!this.isConnected) return;

        const now = Date.now();
        const minGap = this.calcMinIntervalWithJitter();
        const wait = this.lastSendTs ? Math.max(0, minGap - (now - this.lastSendTs)) : 0;
        if (wait > 0) await new Promise(r => setTimeout(r, wait));

        const message = this.queue.shift()!;
        await this.safeSend(message);
        this.lastSendTs = Date.now();
      }
    } finally {
      this.sending = false;
    }
  }

  private async safeSend(message: string): Promise<void> {
    if (!this.isConnected) return;

    const chan = `#${this.channelName}`;

    const echoPromise = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingEcho = null;
        reject(new Error('No echo received'));
      }, 4000);
      this.pendingEcho = { msg: message, timeout, resolve, reject };
    });

    try {
      await this.client.say(chan, message);
      logger.info(`say() -> ${chan}: ${message}`);
    } catch (e) {
      if (this.pendingEcho) {
        clearTimeout(this.pendingEcho.timeout);
        this.pendingEcho = null;
      }
      throw e;
    }

    try {
      await echoPromise;
      logger.info(`Echo confirmed: ${message}`);
    } catch (e) {
      logger.warn(`Send not confirmed: ${(e as Error).message}`);
    }
  }

  public connect(): void {
    logger.info(`Connecting bot... (channel=${this.channelName})`);
    this.client.connect().catch(e => logger.error('connect error:', e));
  }

  public disconnect(): void {
    logger.info('Disconnecting bot...');
    this.client.disconnect();
    this.aiService.stopVoiceCapture();
  }
}
