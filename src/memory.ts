import { promises as fs } from 'fs';
import { join } from 'path';

export type MemoryEventType = 'speech' | 'chat' | 'screen';

export interface MemoryEvent {
    ts: number;              // epoch ms
    type: MemoryEventType;
    text: string;            // уже “чистый” текст (без мусора)
    meta?: Record<string, any>;
}

export interface MemoryFact {
    id: string;              // стабильный id (можно timestamp+rand)
    text: string;            // короткое утверждение
    ts: number;              // когда в последний раз подтверждали/встречали
    importance: number;      // 1..5
    tags?: string[];
}

export interface ChannelMemoryFile {
    channel: string;
    updatedAt: number;
    facts: MemoryFact[];
}

function safeChannelId(channel: string) {
    return channel.toLowerCase().replace(/[^a-z0-9_-]/gi, '_');
}

export class MemoryStore {
    private baseDir: string;

    constructor(baseDir = join(process.cwd(), 'data', 'memory')) {
        this.baseDir = baseDir;
    }

    private async ensureDir() {
        await fs.mkdir(this.baseDir, { recursive: true });
    }

    private filePath(channel: string) {
        return join(this.baseDir, `${safeChannelId(channel)}.json`);
    }

    async load(channel: string): Promise<ChannelMemoryFile> {
        await this.ensureDir();
        const fp = this.filePath(channel);

        try {
            const raw = await fs.readFile(fp, 'utf-8');
            const parsed = JSON.parse(raw) as ChannelMemoryFile;
            if (!parsed.facts) parsed.facts = [];
            return parsed;
        } catch {
            return { channel, updatedAt: Date.now(), facts: [] };
        }
    }

    async save(mem: ChannelMemoryFile): Promise<void> {
        await this.ensureDir();
        mem.updatedAt = Date.now();
        const fp = this.filePath(mem.channel);
        await fs.writeFile(fp, JSON.stringify(mem, null, 2), 'utf-8');
    }

    /**
     * Простой поиск релевантных фактов: по пересечению слов.
     * На старте этого хватит. Потом можно заменить на embeddings/RAG.
     */
    pickRelevantFacts(allFacts: MemoryFact[], queryText: string, limit = 8): MemoryFact[] {
        const q = (queryText || '').toLowerCase();
        const qWords = new Set(q.split(/[^a-zа-я0-9_]+/i).filter(w => w.length >= 3));

        const scored = allFacts.map(f => {
            const t = f.text.toLowerCase();
            const words = t.split(/[^a-zа-я0-9_]+/i).filter(w => w.length >= 3);
            let hit = 0;
            for (const w of words) if (qWords.has(w)) hit++;
            const recency = Math.max(0, 1_000_000_000 - (Date.now() - f.ts)); // грубо
            const score = hit * 10 + (f.importance || 1) * 3 + recency / 1_000_000_000;
            return { f, score };
        });

        return scored
            .sort((a, b) => b.score - a.score)
            .slice(0, limit)
            .map(x => x.f);
    }
}
