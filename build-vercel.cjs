const fs = require('fs');
const cp = require('child_process');
const path = require('path');

const inRoot = fs.existsSync('frontend');
const projectDir = inRoot ? path.resolve('frontend') : process.cwd();

console.log(`[TerraSight Build] Building frontend from directory: ${projectDir}`);

// Run npm run build
cp.execSync('npm run build', {
  cwd: projectDir,
  stdio: 'inherit',
});

// Sync dist so both ./dist and ./frontend/dist exist regardless of where Vercel looks
const sourceDist = path.join(projectDir, 'dist');
if (inRoot) {
  // If in root, copy frontend/dist -> dist
  fs.cpSync(sourceDist, path.resolve('dist'), { recursive: true });
} else {
  // If in frontend, copy dist -> frontend/dist
  const nestedFrontend = path.resolve('frontend');
  if (!fs.existsSync(nestedFrontend)) {
    fs.mkdirSync(nestedFrontend, { recursive: true });
  }
  fs.cpSync(sourceDist, path.join(nestedFrontend, 'dist'), { recursive: true });
}

console.log('[TerraSight Build] Frontend built and synced successfully to dist & frontend/dist.');
