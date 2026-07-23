// loader.js - #28 Load .bin model files
import fs from 'fs';
import { Tensor } from './tensor.js';

/**
 * .bin format:
 *   Magic: "JSTORCH\0" (8 bytes)
 *   Version: uint32 LE
 *   TensorCount: uint32 LE
 *   HeaderSize: uint64 LE (BigInt)
 *   Header: JSON string (utf-8, HeaderSize bytes)
 *     { tensors: { "key": { shape: [...], dtype: "float32", offset: N, size: N }, ... },
 *       config: { ... } }
 *   Data: raw float32 tensor data (concatenated)
 */

const MAGIC = 'JSTORCH\0';

export function loadModel(path) {
    const buf = fs.readFileSync(path);
    
    // Validate magic
    const magic = buf.toString('ascii', 0, 8);
    if (magic !== MAGIC) throw new Error(`Invalid .bin file: bad magic "${magic}"`);
    
    const version = buf.readUInt32LE(8);
    const tensorCount = buf.readUInt32LE(12);
    const headerSize = Number(buf.readBigUInt64LE(16));
    
    // Parse header JSON
    const headerJson = buf.toString('utf-8', 24, 24 + headerSize);
    const header = JSON.parse(headerJson);
    
    const dataOffset = 24 + headerSize;
    const stateDict = {};
    
    for (const [key, info] of Object.entries(header.tensors)) {
        const { shape, offset, size } = info;
        const start = dataOffset + offset;
        const f32 = new Float32Array(buf.buffer, buf.byteOffset + start, size);
        stateDict[key] = Tensor.fromBuffer(f32, shape);
    }
    
    return { stateDict, config: header.config || {} };
}

export function saveModel(path, stateDict, config = {}) {
    const tensors = {};
    const dataBuffers = [];
    let offset = 0;
    
    for (const [key, tensor] of Object.entries(stateDict)) {
        const arr = tensor.toArray().flat(Infinity);
        const f32 = new Float32Array(arr);
        tensors[key] = {
            shape: [...tensor.shape],
            dtype: 'float32',
            offset,
            size: f32.length,
        };
        dataBuffers.push(Buffer.from(f32.buffer));
        offset += f32.length * 4;
    }
    
    const header = JSON.stringify({ tensors, config });
    const headerBuf = Buffer.from(header, 'utf-8');
    
    const count = Object.keys(tensors).length;
    const metaBuf = Buffer.alloc(24);
    metaBuf.write(MAGIC, 0, 8, 'ascii');
    metaBuf.writeUInt32LE(1, 8); // version
    metaBuf.writeUInt32LE(count, 12);
    metaBuf.writeBigUInt64LE(BigInt(headerBuf.length), 16);
    
    const finalBuf = Buffer.concat([metaBuf, headerBuf, ...dataBuffers]);
    fs.writeFileSync(path, finalBuf);
}
