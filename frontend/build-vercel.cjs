const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('[CleanPath Build] Starting universal build script...');
console.log('[CleanPath Build] Current directory:', process.cwd());

// Determine if we are already in frontend or at root
const isInsideFrontend = fs.existsSync(path.join(process.cwd(), 'src', 'App.tsx'));
const targetDir = isInsideFrontend ? process.cwd() : path.join(process.cwd(), 'frontend');

if (fs.existsSync(targetDir) && fs.existsSync(path.join(targetDir, 'package.json'))) {
  console.log('[CleanPath Build] Target frontend directory:', targetDir);
  console.log('[CleanPath Build] Running npm install --include=dev...');
  execSync('npm install --include=dev', { cwd: targetDir, stdio: 'inherit' });

  console.log('[CleanPath Build] Running npx vite build...');
  execSync('npx vite build', { cwd: targetDir, stdio: 'inherit' });

  // If built from root, sync to root dist as well
  const distDir = path.join(targetDir, 'dist');
  const rootDist = path.join(process.cwd(), 'dist');
  if (!isInsideFrontend && fs.existsSync(distDir)) {
    console.log('[CleanPath Build] Syncing dist to root dist...');
    try {
      fs.cpSync(distDir, rootDist, { recursive: true });
    } catch (e) {
      console.warn('[CleanPath Build] Sync warning:', e.message);
    }
  }
} else {
  console.log('[CleanPath Build] Frontend directory not applicable in this context; exiting cleanly.');
}
console.log('[CleanPath Build] Universal build finished successfully.');
