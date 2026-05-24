import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import apiRouter from './src/api.ts';

// Load environment config
dotenv.config();

const app = express();
const PORT = 3000;

// Derive directories for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Request parsing middleware
app.use(express.json());

// Log incoming API calls for diagnostic convenience
app.use((req, res, next) => {
  if (req.url.startsWith('/api')) {
    console.log(`[RAG Server] ${req.method} ${req.url}`);
  }
  next();
});

// Bind API routing sub-system
app.use('/api', apiRouter);

// Serve static assets from build output folder
const distPath = path.join(__dirname, 'dist');
app.use(express.static(distPath));

// Direct client fallback routing of HTML5 routes to SPA entry
app.get('*', (req, res, next) => {
  if (req.url.startsWith('/api')) {
    return next();
  }
  res.sendFile(path.join(distPath, 'index.html'));
});

// Launch server listener
app.listen(PORT, '0.0.0.0', () => {
  console.log(`=================================================`);
  console.log(`🚀 GenAI RAG Reference Guide Server running offline`);
  console.log(`🌐 Address: http://localhost:${PORT}`);
  console.log(`🔑 Gemini Key Present: ${!!process.env.GEMINI_API_KEY}`);
  console.log(`=================================================`);
});
