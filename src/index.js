import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

const addon = require(path.join(__dirname, '../build/win/jstorch.node'));

console.log(addon.hello())