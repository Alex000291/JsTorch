import fs from 'fs';
import { exec as execCallback, execSync } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import { fileURLToPath } from 'url';


const exec = promisify(execCallback);
const obj_files = [];

async function check_installed(compiler) {
    try {
        // cl.exe 不支持 --version，需要特殊处理
        if (compiler === 'cl') {
            await exec('cl');  // cl.exe 无参数会输出版本信息（返回错误码但有输出）
        } else {
            await exec(`${compiler} --version`);
        }
    } catch (error) {
        // cl.exe 无参数会返回非零退出码，但如果输出包含 "Microsoft" 说明安装了
        if (compiler === 'cl' && error.stderr && error.stderr.includes('Microsoft')) {
            return true;
        }
        console.warn(`${compiler} not installed`);
        return false;
    }
    return true;
}

async function compile(vars) {
    const allFiles = fs.readdirSync('./native', { recursive: true });

    // 编译 C++ 文件
    const cpp_src_files = allFiles
        .filter(f => f.endsWith('.cpp'))
        .map(f => f.replace(/\.[^.]+$/, ''));
    
    for (const element of cpp_src_files) {
        const srcPath = path.join('native', element + '.cpp');
        const objFile = path.join('native', 'cpp_obj', path.basename(element) + '.obj');
        console.log(`Compiling ${srcPath}...`);
        
        // Windows 使用 cl.exe (MSVC)，需要 C++17 标准（/std:c++17）
        const cudaPath = process.env.CUDA_PATH || 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.4';
        const cudaInclude = path.join(cudaPath, 'include');
        
        await exec(`cl /c "${srcPath}" /Fo"${objFile}" /std:c++20 /EHsc /MD /utf-8 /I"node_modules\\node-addon-api" /I"node_modules\\node-api-headers\\include" /I"${cudaInclude}"`);
        obj_files.push(objFile);
    }

    // 编译 CUDA 文件
    const cu_src_files = allFiles
        .filter(f => f.endsWith('.cu'))
        .map(f => f.replace(/\.[^.]+$/, ''));
    
    for (const element of cu_src_files) {
        const srcPath = path.join('native', element + '.cu');
        const objFile = path.join('native', 'cu_obj', path.basename(element) + '.obj');
        console.log(`Compiling ${srcPath} for all architectures...`);
        
        // Windows 使用 nvcc + MSVC
        const cleanEnv = {
            ...process.env,
            ...vars,
            TEMP: process.env.TEMP || 'C:\\Temp',
            TMP: process.env.TMP || 'C:\\Temp'
        };
        
        // 编译所有架构到同一个 .obj 文件（类似 PyTorch）
        // CUDA 13.4 支持的架构：sm_75 及以上
        await exec(`nvcc -c "${srcPath}" -o "${objFile}" -Xcompiler /MD `
            + `-gencode arch=compute_75,code=sm_75 `  // Turing: RTX 20xx, GTX 16xx
            + `-gencode arch=compute_80,code=sm_80 `  // Ampere: A100
            + `-gencode arch=compute_86,code=sm_86 `  // Ampere: RTX 30xx
            + `-gencode arch=compute_89,code=sm_89 `  // Ada: RTX 40xx
            + `-gencode arch=compute_90,code=sm_90 `  // Hopper: H100
            + `-gencode arch=compute_90,code=compute_90`, { // PTX for future
            env: cleanEnv
        });
        obj_files.push(objFile);
    }
}

async function link() {
    console.log('Linking...');
    const outPath = path.join('build', 'win', 'jstorch.node');
    const nodeLibPath = path.join('build', 'win', 'libs', 'node');
    const cudaLibPath = path.join('build', 'win', 'libs', 'cuda');
    
    // Windows 使用 link.exe (MSVC linker)
    // 链接 node.lib（N-API）和 CUDA 库
    await exec(`link /DLL /OUT:"${outPath}" ${obj_files.map(f => `"${f}"`).join(' ')} /LIBPATH:"${nodeLibPath}" /LIBPATH:"${cudaLibPath}" node.lib cudart.lib cublas.lib`);
    
    // 清理中间产物
    const expFile = path.join('build', 'win', 'jstorch.exp');
    const libFile = path.join('build', 'win', 'jstorch.lib');
    if (fs.existsSync(expFile)) fs.unlinkSync(expFile);
    if (fs.existsSync(libFile)) fs.unlinkSync(libFile);
    console.log('Cleaned up intermediate files (.exp, .lib)');
}

async function setupMSVCEnvironment() {
    if (process.platform !== 'win32') return {};
    
    console.log('Setting up MSVC environment...');
    try {
        const env = execSync(
            `"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" && set`,
            { encoding: 'utf8' }
        );

        const vars = {};
        env.split('\n').forEach(line => {
            const i = line.indexOf('=');
            if (i > 0) {
                const key = line.slice(0, i);
                const value = line.slice(i + 1).replace(/\r/g, '').trim();  // 移除 \r
                vars[key] = value;
            }
        });

        // 更新当前进程环境变量
        Object.assign(process.env, vars);
        console.log('MSVC environment loaded');
        return vars;
    } catch (error) {
        console.error('Failed to load MSVC environment. Please run in "x64 Native Tools Command Prompt for VS 2022"');
        throw error;
    }
}

async function build() {
    // Windows: 先加载 MSVC 环境
    let vars = {};
    if (process.platform === 'win32') {
        vars = await setupMSVCEnvironment();
    }
    
    // 检查编译器
    if (process.platform === 'win32') {
        if (!(await check_installed('cl'))) {
            console.error('MSVC (cl.exe) not found after environment setup');
            return;
        }
    } else {
        if (!(await check_installed('g++'))) return;
    }
    
    if (!(await check_installed('nvcc'))) return;
    
    // 编译单个包含所有架构的 .node 文件（类似 PyTorch）
    console.log('\n=== Building universal binary with all GPU architectures ===');
    await compile(vars);
    await link();
    console.log(`\n✓ Build complete: ${path.join('build', 'win', 'jstorch.node')}`);
    console.log('\nSupported GPU architectures:');
    console.log('  - sm_75: Turing (RTX 20xx, GTX 16xx, Tesla T4, Quadro RTX)');
    console.log('  - sm_80: Ampere (A100, A30, A40)');
    console.log('  - sm_86: Ampere (RTX 30xx, A10, A16, A2)');
    console.log('  - sm_89: Ada Lovelace (RTX 40xx, L4, L40)');
    console.log('  - sm_90: Hopper (H100, H800)');
    console.log('  - PTX: Future/unknown architectures (JIT compiled at runtime)');
}

build();