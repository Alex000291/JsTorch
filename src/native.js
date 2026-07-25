// native.js - Direct access to native Tensor module
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
const require = createRequire(import.meta.url);

const __dirname = dirname(fileURLToPath(import.meta.url));
const dllDirs = ['build/win/dlls/cuda', 'build/win/dlls/cudnn'].map(d => join(__dirname, '..', d));
process.env.PATH = dllDirs.join(';') + ';' + process.env.PATH;

export default require('../build/win/jstorch.node');
