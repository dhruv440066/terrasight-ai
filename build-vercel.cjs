const fs = require('fs');
const cp = require('child_process');
const path = require('path');

const rootDir = process.cwd();
const hasFrontendFolder = fs.existsSync(path.join(rootDir, 'frontend', 'package.json'));
const frontendDir = hasFrontendFolder ? path.join(rootDir, 'frontend') : rootDir;

console.log('[TerraSight Build] Target frontend directory:', frontendDir);

// 1. Install dependencies in frontend directory
console.log('[TerraSight Build] Step 1: Installing dependencies...');
try {
  cp.execSync('npm install --include=dev', { cwd: frontendDir, stdio: 'inherit' });
} catch (e) {
  console.warn('[TerraSight Build] npm install warning:', e.message);
  cp.execSync('npm install', { cwd: frontendDir, stdio: 'inherit' });
}

// 2. Locate vite binary and compile
console.log('[TerraSight Build] Step 2: Compiling production bundle with Vite...');
const viteBin = path.join(frontendDir, 'node_modules', 'vite', 'bin', 'vite.js');
if (fs.existsSync(viteBin)) {
  cp.execSync(`node "${viteBin}" build`, { cwd: frontendDir, stdio: 'inherit' });
} else {
  cp.execSync('npx vite build', { cwd: frontendDir, stdio: 'inherit' });
}

// 3. Mirror dist to root and frontend so outputDirectory matches wherever Vercel checks
console.log('[TerraSight Build] Step 3: Synchronizing output dist folders...');
const sourceDist = path.join(frontendDir, 'dist');
if (!fs.existsSync(sourceDist)) {
  throw new Error(`Build failed: dist folder was not generated at ${sourceDist}`);
}

const rootDist = path.join(rootDir, 'dist');
if (rootDist !== sourceDist) {
  fs.cpSync(sourceDist, rootDist, { recursive: true });
}

const nestedDist = path.join(rootDir, 'frontend', 'dist');
if (nestedDist !== sourceDist) {
  fs.cpSync(sourceDist, nestedDist, { recursive: true });
}

console.log('[TerraSight Build] Build complete! Production bundle verified.');
