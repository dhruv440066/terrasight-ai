const fs = require('fs');
const cp = require('child_process');
const path = require('path');

// Check if we are running from the repository root (frontend/package.json exists)
const hasFrontendDir = fs.existsSync(path.join(process.cwd(), 'frontend', 'package.json'));
const projectDir = hasFrontendDir ? path.resolve('frontend') : process.cwd();

console.log(`[TerraSight Build] Building frontend from directory: ${projectDir}`);

// Ensure dependencies are installed in projectDir
console.log('[TerraSight Build] Installing frontend dependencies...');
cp.execSync('npm install --include=dev', {
  cwd: projectDir,
  stdio: 'inherit',
});

// Run vite build directly using npx
console.log('[TerraSight Build] Compiling frontend production bundle with Vite...');
cp.execSync('npx vite build', {
  cwd: projectDir,
  stdio: 'inherit',
});

// If in root, copy frontend/dist -> dist so outputDirectory "dist" matches in both root and frontend
const sourceDist = path.join(projectDir, 'dist');
if (hasFrontendDir) {
  const rootDist = path.resolve('dist');
  fs.cpSync(sourceDist, rootDist, { recursive: true });
}

console.log('[TerraSight Build] Frontend built successfully.');
