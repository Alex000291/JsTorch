// Download CUDA/cuDNN DLLs on npm install
import { existsSync, mkdirSync, createWriteStream, unlinkSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';
import https from 'https';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dllsDir = join(root, 'build', 'win', 'dlls');
const zipPath = join(root, 'build', 'win', 'dlls.zip');
const url = 'https://github.com/Alex000291/JsTorch/releases/download/v0.7.0/dlls.zip';

if (process.platform !== 'win32') process.exit(0);
if (existsSync(join(dllsDir, '.ok'))) { console.log('[jstorch] DLLs ready.'); process.exit(0); }

console.log('[jstorch] Downloading CUDA/cuDNN DLLs (~1.9GB)...');

const follow = (u) => new Promise((res, rej) => {
    https.get(u, { headers: { 'User-Agent': 'jstorch' } }, r => {
        if (r.statusCode >= 300 && r.headers.location) return follow(r.headers.location).then(res, rej);
        if (r.statusCode !== 200) return rej(new Error(`HTTP ${r.statusCode}`));
        const f = createWriteStream(zipPath);
        let dl = 0; const total = +r.headers['content-length'] || 0;
        r.on('data', c => { dl += c.length; if (total) process.stdout.write(`\r[jstorch] ${(dl/total*100|0)}%`); });
        r.pipe(f); f.on('finish', () => { console.log(''); res(); });
    }).on('error', rej);
});

mkdirSync(join(root, 'build', 'win'), { recursive: true });
await follow(url);
console.log('[jstorch] Extracting...');
mkdirSync(dllsDir, { recursive: true });
execSync(`powershell -Command "Expand-Archive -Path '${zipPath}' -DestinationPath '${dllsDir}' -Force"`, { stdio: 'inherit' });
unlinkSync(zipPath);
writeFileSync(join(dllsDir, '.ok'), '');
console.log('[jstorch] Done.');
