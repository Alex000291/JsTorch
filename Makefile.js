// Makefile.js - 完美架构构建系统
import { execSync } from 'child_process';
import fs from 'fs';

import { dirname } from 'path';

const CUDA_PATH = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.4';
const MSVC_PATH = 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community';
const NODE_PATH = dirname(process.execPath);

console.log('=== JsTorch Perfect Build ===\n');

// Load MSVC environment
const vcvars = `"${MSVC_PATH}\\VC\\Auxiliary\\Build\\vcvars64.bat"`;
try {
    const envOutput = execSync(`cmd /c "${vcvars} > nul && set"`, { encoding: 'utf8', stdio: 'pipe' });
    envOutput.split('\n').forEach(line => {
        const m = line.trim().match(/^([^=]+)=(.*)$/);
        if (m && m[1] && m[2]) {
            process.env[m[1]] = m[2];
        }
    });
    console.log('MSVC loaded\n');
} catch(e) {
    console.error('Failed to load MSVC environment');
    process.exit(1);
}

fs.mkdirSync('build/obj', { recursive: true });

const sources = {
    cpp: ['native/core/tensor.cpp', 'native/binding/napi.cpp'],
    cuda: ['native/ops/unary.cu', 'native/ops/binary.cu', 'native/ops/reduce.cu', 'native/ops/misc.cu', 'native/ops/matmul.cu', 'native/ops/conv.cu', 'native/audio/stft.cu']
};

const includes = `-Inode_modules\\node-addon-api -Inode_modules\\node-api-headers\\include -I"${CUDA_PATH}\\include" -Ibuild\\win\\include -Inative/core -Inative/ops`;

// Compile CUDA
console.log('Compiling CUDA...');
for (const src of sources.cuda) {
    const obj = `build/obj/${src.replace(/\//g, '_').replace('.cu', '.obj')}`;
    console.log(`  ${src}`);
    try {
        execSync(`nvcc -std=c++17 -O3 --use_fast_math -Xcompiler /MD -Xcompiler /EHsc -Xcompiler /wd4819 -DNAPI_VERSION=8 --generate-code=arch=compute_89,code=sm_89 --generate-code=arch=compute_90,code=sm_90 --generate-code=arch=compute_90,code=compute_90 ${includes} -c ${src} -o ${obj}`, { stdio: 'inherit' });
    } catch(e) {
        console.error(`Failed: ${src}`);
        process.exit(1);
    }
}

// Compile C++
console.log('\nCompiling C++...');
for (const src of sources.cpp) {
    const obj = `build/obj/${src.replace(/\//g, '_').replace('.cpp', '.obj')}`;
    console.log(`  ${src}`);
    try {
        const output = execSync(`cl /std:c++17 /O2 /MD /EHsc /utf-8 /wd4819 /DNAPI_VERSION=8 ${includes} /c ${src} /Fo${obj}`, { encoding: 'utf8', stdio: 'pipe' });
        if (output) console.log(output);
    } catch(e) {
        console.error(e.stdout || e.stderr || e.message);
        process.exit(1);
    }
}

// Link
console.log('\nLinking...');
const objs = [...sources.cuda, ...sources.cpp].map(s => 
    `build/obj/${s.replace(/\//g, '_').replace(/\.(cu|cpp)$/, '.obj')}`
).join(' ');

fs.mkdirSync('build/win', { recursive: true });
execSync(`link /DLL /OUT:build/win/jstorch.node ${objs} /LIBPATH:"build\\win\\libs\\node" /LIBPATH:"${CUDA_PATH}\\lib\\x64" /LIBPATH:"build\\win\\libs\\cudnn" node.lib cudart.lib cublas.lib cufft.lib curand.lib cudnn.lib`, { stdio: 'inherit' });

console.log('\n✓ Build complete: build/win/jstorch.node\n');
