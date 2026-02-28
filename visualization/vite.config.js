import { defineConfig } from 'vite';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const runsDir = path.resolve(__dirname, '..', 'experimentation', 'runs');

export default defineConfig({
  root: '.',
  publicDir: 'public',
  server: {
    open: true,
  },
  plugins: [
    {
      name: 'serve-experiment-runs',
      configureServer(server) {
        // Return a function so this runs BEFORE Vite's internal middleware
        server.middlewares.use((req, res, next) => {
          if (!req.url.startsWith('/data/')) return next();

          // Strip /data prefix and query params
          const urlPath = req.url.slice('/data'.length).split('?')[0];
          const filePath = path.join(runsDir, decodeURIComponent(urlPath));

          // Prevent path traversal
          if (!filePath.startsWith(runsDir)) return next();

          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            res.setHeader('Content-Type', 'application/json');
            res.setHeader('Cache-Control', 'no-cache');
            fs.createReadStream(filePath).pipe(res);
          } else {
            next();
          }
        });
      },
    },
  ],
});
