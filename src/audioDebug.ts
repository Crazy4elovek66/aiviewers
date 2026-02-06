// src/audioDebug.ts
// Безопасная диагностика PCM16LE (int16) — считает RMS и Peak.
// Никаких побочных эффектов, только анализ буфера.

export function analyzePcm16le(buf: Buffer): { rms: number; peak: number; samples: number } {
    try {
        if (!buf || buf.length < 2) return { rms: 0, peak: 0, samples: 0 };

        const sampleCount = Math.floor(buf.length / 2); // int16 => 2 bytes
        const samples = new Int16Array(buf.buffer, buf.byteOffset, sampleCount);

        let sumSq = 0;
        let peak = 0;

        for (let i = 0; i < samples.length; i++) {
            const v = samples[i] / 32768; // -1..1
            const av = Math.abs(v);
            if (av > peak) peak = av;
            sumSq += v * v;
        }

        const rms = samples.length ? Math.sqrt(sumSq / samples.length) : 0;
        return { rms, peak, samples: samples.length };
    } catch {
        // Если вдруг что-то не так с буфером — не ломаем поток
        return { rms: 0, peak: 0, samples: 0 };
    }
}
