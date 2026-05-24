import express from 'express';
import { GoogleGenAI } from '@google/genai';

// Initialize the Express router
const router = express.Router();

// Types for RAG Store
export interface Document {
  id: string;
  title: string;
  content: string;
}

export interface Chunk {
  id: string;
  docId: string;
  docTitle: string;
  text: string;
  charCount: number;
  wordCount: number;
}

export interface EmbeddedChunk {
  chunk: Chunk;
  embedding: number[];
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  timestamp: string;
}

export interface ChatSession {
  sessionId: string;
  history: ChatMessage[];
  lastActive: string;
}

// In-Memory RAG Database
let dbDocs: Document[] = [
  {
    id: 'doc-1',
    title: 'Reset Password Options',
    content: 'Users can reset their password from Settings > Security. They must verify their identity via their registered email address. This option is available globally across both web and mobile client interfaces.'
  },
  {
    id: 'doc-2',
    title: 'Account Deletion Policy',
    content: 'Users can delete their account permanently from Account Settings. Once deleted, all personal transactions, history, and workspace files are archived for 30 days before visual purge.'
  },
  {
    id: 'doc-3',
    title: 'Refund Claims Period',
    content: 'Refund claims are processed within 5-7 business days. Customers must submit their claim within 14 days of purchase. No refunds are permitted for enterprise tiers after workspace setup.'
  },
  {
    id: 'doc-4',
    title: 'API Rate Limits details',
    content: 'The API rate limit is 60 requests per minute for the free tier, and 1000 requests per minute for the premium subscription tier. Exceeding limits triggers an HTTP 429 error response.'
  },
  {
    id: 'doc-5',
    title: 'Recommended Gemini Models',
    content: 'For standard text tasks including Q&A, formatting, and summarization, use gemini-3.5-flash. For high-accuracy text embedding tasks, use the gemini-embedding-2-preview model.'
  },
  {
    id: 'doc-6',
    title: 'Retrieval RAG Threshold config',
    content: 'We enforce a similarity threshold of 0.70 to 0.75 (cosine similarity metrics) to isolate high-quality context and reject secondary noise. Refuse requests with max score < threshold.'
  }
];

let dbChunks: Chunk[] = [];
let dbEmbeddings: EmbeddedChunk[] = [];
const dbSessions: Map<string, ChatSession> = new Map();

// Helper to calculate Cosine Similarity
function dotProduct(a: number[], b: number[]): number {
  let product = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    product += a[i] * b[i];
  }
  return product;
}

function magnitude(a: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * a[i];
  }
  return Math.sqrt(sum);
}

function cosineSimilarity(a: number[], b: number[]): number {
  const m1 = magnitude(a);
  const m2 = magnitude(b);
  if (m1 === 0 || m2 === 0) return 0;
  return dotProduct(a, b) / (m1 * m2);
}

// Generate deterministic/mock embeddings for fallback when key is not defined
function generateMockEmbedding(text: string, dimensions = 768): number[] {
  // Simple hashing algorithm to make embedding deterministic based on input text
  const values: number[] = [];
  let hash1 = 5381;
  let hash2 = 89;
  
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash1 = ((hash1 << 5) + hash1) + char;
    hash2 = ((hash2 << 4) + hash2) ^ char;
  }

  for (let i = 0; i < dimensions; i++) {
    const angle = ((hash1 * i + hash2) % 360) * (Math.PI / 180);
    values.push(Math.sin(angle) * (0.5 + 0.5 * Math.cos(i * 0.1)));
  }

  // Normalize vector to magnitude 1.0 (so dot product equals cosine similarity!)
  const mag = magnitude(values);
  if (mag > 0) {
    for (let i = 0; i < values.length; i++) {
      values[i] = values[i] / mag;
    }
  }
  return values;
}

// Instantiate Gemini SDK cleanly and safely (lazy getter to prevent startup crash)
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): { ai: GoogleGenAI | null; hasKey: boolean } {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY' || apiKey.trim() === '') {
    return { ai: null, hasKey: false };
  }
  
  if (!aiClient) {
    aiClient = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return { ai: aiClient, hasKey: true };
}

// Initial Chunking logic (run once on start)
function performChunkingLogic(
  docs: Document[],
  chunkSize: number = 150,
  overlap: number = 30,
  strategy: 'char' | 'word' | 'sentence' = 'char'
): Chunk[] {
  const chunks: Chunk[] = [];
  
  for (const doc of docs) {
    const text = doc.content;
    
    if (strategy === 'sentence') {
      // Split on standard punctuation
      const sentences = text.match(/[^.!?]+[.!?]+(\s|$)/g) || [text];
      let currentChunkText = '';
      let chunkIdx = 1;
      
      for (const sent of sentences) {
        if ((currentChunkText + sent).length > chunkSize && currentChunkText.trim()) {
          chunks.push({
            id: `chunk-${doc.id}-${chunkIdx++}`,
            docId: doc.id,
            docTitle: doc.title,
            text: currentChunkText.trim(),
            charCount: currentChunkText.trim().length,
            wordCount: currentChunkText.trim().split(/\s+/).length
          });
          // Simple overlap: keep some characters
          currentChunkText = currentChunkText.substring(Math.max(0, currentChunkText.length - overlap)) + sent;
        } else {
          currentChunkText += (currentChunkText ? ' ' : '') + sent;
        }
      }
      
      if (currentChunkText.trim()) {
        chunks.push({
          id: `chunk-${doc.id}-${chunkIdx++}`,
          docId: doc.id,
          docTitle: doc.title,
          text: currentChunkText.trim(),
          charCount: currentChunkText.trim().length,
          wordCount: currentChunkText.trim().split(/\s+/).length
        });
      }
    } else if (strategy === 'word') {
      const words = text.split(/\s+/);
      let i = 0;
      let chunkIdx = 1;
      
      while (i < words.length) {
        const slice = words.slice(i, i + chunkSize);
        const chunkText = slice.join(' ');
        chunks.push({
          id: `chunk-${doc.id}-${chunkIdx++}`,
          docId: doc.id,
          docTitle: doc.title,
          text: chunkText,
          charCount: chunkText.length,
          wordCount: slice.length
        });
        i += Math.max(1, chunkSize - overlap);
      }
    } else {
      // default: 'char' strategy (chunks by characters)
      let i = 0;
      let chunkIdx = 1;
      
      while (i < text.length) {
        const chunkText = text.substring(i, i + chunkSize);
        chunks.push({
          id: `chunk-${doc.id}-${chunkIdx++}`,
          docId: doc.id,
          docTitle: doc.title,
          text: chunkText,
          charCount: chunkText.length,
          wordCount: chunkText.trim().split(/\s+/).filter(Boolean).length
        });
        
        i += Math.max(1, chunkSize - overlap);
      }
    }
  }
  
  return chunks;
}

// Initial hydration
dbChunks = performChunkingLogic(dbDocs, 120, 25, 'char');
dbEmbeddings = dbChunks.map(ch => ({
  chunk: ch,
  embedding: generateMockEmbedding(ch.text)
}));

// Define API Endpoints
// 1. GET Current State Information
router.get('/state', (req, res) => {
  const { hasKey } = getGeminiClient();
  res.json({
    hasRealApiKey: hasKey,
    documentCount: dbDocs.length,
    chunkCount: dbChunks.length,
    embeddingCount: dbEmbeddings.length,
    sessionCount: dbSessions.size,
  });
});

// 2. GET Documents List
router.get('/docs', (req, res) => {
  res.json({ docs: dbDocs });
});

// 3. POST Update Documents List
router.post('/docs', (req, res) => {
  const { docs } = req.body;
  if (!Array.isArray(docs)) {
    return res.status(400).json({ error: 'Docs format must be an array of documents' });
  }

  // Map and validate docs
  dbDocs = docs.map((doc: any, idx: number) => ({
    id: doc.id || `doc-${idx + 1}`,
    title: String(doc.title || `Untitled Doc ${idx + 1}`).trim(),
    content: String(doc.content || '').trim(),
  })).filter(d => d.content.length > 0);

  // Clear stale chunks and embeddings
  dbChunks = [];
  dbEmbeddings = [];

  res.json({ message: 'Documents updated successfully', count: dbDocs.length });
});

// 4. POST Reset Documents
router.post('/docs/reset', (req, res) => {
  dbDocs = [
    {
      id: 'doc-1',
      title: 'Reset Password Options',
      content: 'Users can reset their password from Settings > Security. They must verify their identity via their registered email address. This option is available globally across both web and mobile client interfaces.'
    },
    {
      id: 'doc-2',
      title: 'Account Deletion Policy',
      content: 'Users can delete their account permanently from Account Settings. Once deleted, all personal transactions, history, and workspace files are archived for 30 days before visual purge.'
    },
    {
      id: 'doc-3',
      title: 'Refund Claims Period',
      content: 'Refund claims are processed within 5-7 business days. Customers must submit their claim within 14 days of purchase. No refunds are permitted for enterprise tiers after workspace setup.'
    },
    {
      id: 'doc-4',
      title: 'API Rate Limits details',
      content: 'The API rate limit is 60 requests per minute for the free tier, and 1000 requests per minute for the premium subscription tier. Exceeding limits triggers an HTTP 429 error response.'
    },
    {
      id: 'doc-5',
      title: 'Recommended Gemini Models',
      content: 'For standard text tasks including Q&A, formatting, and summarization, use gemini-3.5-flash. For high-accuracy text embedding tasks, use the gemini-embedding-2-preview model.'
    },
    {
      id: 'doc-6',
      title: 'Retrieval RAG Threshold config',
      content: 'We enforce a similarity threshold of 0.70 to 0.75 (cosine similarity metrics) to isolate high-quality context and reject secondary noise. Refuse requests with max score < threshold.'
    }
  ];
  
  dbChunks = performChunkingLogic(dbDocs, 120, 25, 'char');
  dbEmbeddings = dbChunks.map(ch => ({
    chunk: ch,
    embedding: generateMockEmbedding(ch.text)
  }));
  dbSessions.clear();

  res.json({ message: 'Documents and context database reset successfully', docs: dbDocs });
});

// 5. POST Chunking trigger
router.post('/chunks/make', (req, res) => {
  const { chunkSize = 150, chunkOverlap = 30, strategy = 'char' } = req.body;
  
  dbChunks = performChunkingLogic(
    dbDocs,
    Math.max(20, Number(chunkSize)),
    Math.max(0, Number(chunkOverlap)),
    strategy
  );

  // Clearing out old embeddings immediately as vectors correspond to old chunks
  dbEmbeddings = [];

  res.json({
    message: `Regenerated ${dbChunks.length} chunks`,
    chunks: dbChunks
  });
});

// 6. GET visual chunks list
router.get('/chunks', (req, res) => {
  res.json({ chunks: dbChunks });
});

// 7. POST Embeddings trigger
router.post('/embeddings/generate', async (req, res) => {
  if (dbChunks.length === 0) {
    return res.status(400).json({ error: 'No chunks available. Please configure docs or trigger chunking first.' });
  }

  const { ai, hasKey } = getGeminiClient();
  const startTime = Date.now();
  const embeddingsList: EmbeddedChunk[] = [];
  
  try {
    if (hasKey && ai) {
      // Real API calls utilizing gemini-embedding-2-preview
      for (const ch of dbChunks) {
        try {
          const response = await ai.models.embedContent({
            model: 'gemini-embedding-2-preview',
            contents: ch.text
          });
          const embeddingObj = (response as any).embedding || (response as any).embeddings;
          const values = embeddingObj?.values;
          if (values) {
            embeddingsList.push({
              chunk: ch,
              embedding: values
            });
          } else {
            throw new Error('No embedding values returned');
          }
        } catch (singleErr: any) {
          console.error(`Embedding generation failed for chunk ID ${ch.id}`, singleErr);
          // Fallback to mock embedding values for this specific node
          embeddingsList.push({
            chunk: ch,
            embedding: generateMockEmbedding(ch.text)
          });
        }
      }
      
      dbEmbeddings = embeddingsList;
      return res.json({
        success: true,
        isSimulated: false,
        dimension: dbEmbeddings[0]?.embedding.length || 768,
        count: dbEmbeddings.length,
        durationMs: Date.now() - startTime
      });
    } else {
      // Simulated Local Mode
      for (const ch of dbChunks) {
        embeddingsList.push({
          chunk: ch,
          embedding: generateMockEmbedding(ch.text)
        });
      }
      
      dbEmbeddings = embeddingsList;
      return res.json({
        success: true,
        isSimulated: true,
        message: 'Running in Simulated Mode: To create real AI vectors, insert a GEMINI_API_KEY in Settings > Secrets.',
        dimension: 768,
        count: dbEmbeddings.length,
        durationMs: Date.now() - startTime
      });
    }
  } catch (err: any) {
    console.error('Fatal error in embeddings endpoint', err);
    return res.status(500).json({
      error: 'Failed to process embedding matrix calculations',
      details: err.message
    });
  }
});

// 8. GET current Embeddings preview
router.get('/embeddings', (req, res) => {
  // Mapping embeddings to exclude the massive vector matrix for general scannable view, or pass truncated preview
  const summaries = dbEmbeddings.map((emb) => ({
    chunkId: emb.chunk.id,
    docTitle: emb.chunk.docTitle,
    text: emb.chunk.text,
    vectorPreview: emb.embedding.slice(0, 5).map(v => Number(v.toFixed(4))),
    dimension: emb.embedding.length
  }));
  res.json({
    totalCount: dbEmbeddings.length,
    embeddings: summaries
  });
});

// 9. POST Calculate Cosine Similarity for search
router.post('/similarity/search', async (req, res) => {
  const { query = '', threshold = 0.70, k = 3 } = req.body;
  if (!query.trim()) {
    return res.status(400).json({ error: 'Query text cannot be empty' });
  }

  if (dbEmbeddings.length === 0) {
    return res.status(400).json({ error: 'Vector store empty. Compute embedding matrices before search.' });
  }

  const { ai, hasKey } = getGeminiClient();
  let queryVector: number[];

  try {
    if (hasKey && ai) {
      try {
        const response = await ai.models.embedContent({
          model: 'gemini-embedding-2-preview',
          contents: query
        });
        const embeddingObj = (response as any).embedding || (response as any).embeddings;
        queryVector = embeddingObj?.values;
        if (!queryVector) throw new Error('Failed to retrieve vector values');
      } catch (embErr) {
        console.error('Query embedding failed via API, falling back to mock.', embErr);
        queryVector = generateMockEmbedding(query);
      }
    } else {
      queryVector = generateMockEmbedding(query);
    }

    // Compute similarities
    const results = dbEmbeddings.map((emb) => {
      const score = cosineSimilarity(queryVector, emb.embedding);
      return {
        chunk: emb.chunk,
        score: Number(score.toFixed(4)),
        isGrounded: score >= threshold
      };
    });

    // Sort results by score (descending)
    results.sort((a, b) => b.score - a.score);

    // Filter by Top K
    const slicedResults = results.slice(0, Number(k));

    res.json({
      query,
      queryVectorPreview: queryVector.slice(0, 8).map(v => Number(v.toFixed(4))),
      isSimulated: !hasKey,
      threshold,
      k,
      results: slicedResults,
      hasMatchAboveThreshold: slicedResults.some(r => r.score >= threshold)
    });
  } catch (err: any) {
    res.status(500).json({ error: 'Failed similarity matrix processing', details: err.message });
  }
});

// 10. POST Chat interaction API endpoints
router.post('/chat', async (req, res) => {
  const { sessionId = 'default_student_session', message = '', threshold = 0.70, k = 3, temperature = 0.2 } = req.body;
  
  if (!message.trim()) {
    return res.status(400).json({ error: 'User message input must not be empty' });
  }

  // Get or create session
  let session = dbSessions.get(sessionId);
  if (!session) {
    session = {
      sessionId,
      history: [],
      lastActive: new Date().toISOString()
    };
    dbSessions.set(sessionId, session);
  }

  const { ai, hasKey } = getGeminiClient();
  
  // RAG Workflow:
  // Step 1: Query embedding generation & similarity search
  let queryVector: number[] = [];
  try {
    if (hasKey && ai) {
      const response = await ai.models.embedContent({
        model: 'gemini-embedding-2-preview',
        contents: message
      });
      const embeddingObj = (response as any).embedding || (response as any).embeddings;
      queryVector = embeddingObj?.values || generateMockEmbedding(message);
    } else {
      queryVector = generateMockEmbedding(message);
    }
  } catch (err) {
    queryVector = generateMockEmbedding(message);
  }

  // Step 2: Retrieve relevant chunks
  let retrievedChunks: { text: string; score: number; docTitle: string; chunkId: string }[] = [];
  let isGrounded = false;

  if (dbEmbeddings.length > 0) {
    const scores = dbEmbeddings.map(emb => ({
      chunkId: emb.chunk.id,
      docTitle: emb.chunk.docTitle,
      text: emb.chunk.text,
      score: Number(cosineSimilarity(queryVector, emb.embedding).toFixed(4))
    }));
    
    // Filter and sort
    scores.sort((a, b) => b.score - a.score);
    const candidateChunks = scores.slice(0, Number(k));
    
    // Threshold application - at least one must meet threshold
    const topScore = candidateChunks[0]?.score || 0;
    if (topScore >= threshold) {
      // Use only the chunks that are >= threshold to form context
      retrievedChunks = candidateChunks.filter(c => c.score >= threshold);
      isGrounded = true;
    } else {
      retrievedChunks = candidateChunks; // Return candidate checks for student analysis but will NOT ground
      isGrounded = false;
    }
  }

  // Step 3: Build grounded or empty context prompt
  let contextBlock = '';
  if (isGrounded && retrievedChunks.length > 0) {
    contextBlock = retrievedChunks.map((c, i) => `[Context Chunk #${i + 1} - From doc: ${c.docTitle}]\n${c.text}`).join('\n\n');
  } else {
    contextBlock = 'NO RELEVANT CONTEXT FOUND IN THE INTERNAL KNOWLEDGE BASE.';
  }

  // Build conversational message history block
  const historyLimit = session.history.slice(-6); // Last 3 turns (3 pairs)
  const historyText = historyLimit.map(msg => `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.text}`).join('\n');

  // Format the comprehensive prompt to mirror Assignment specs:
  const systemInstruction = 
    `You are a grounded RAG QA chat assistant for our student assignments.
Guidelines:
1. You MUST answer the user question strictly using ONLY the provided "Context Block" below.
2. If the "Context Block" matches "NO RELEVANT CONTEXT FOUND", or is insufficient to answer, you must respond EXACTLY with the literal sentence: "I do not have enough information to answer that question based on the knowledge base."
3. Keep your answers brief, factual, precise, and completely faithful to the context block. Avoid extra remarks or speculation.
4. If the question is a generic greeting (e.g., 'hello', 'hi', 'hey') and has no contextual question, you can reply with a brief friendly greeting and list what topics you are authorized to discuss.`;

  const dynamicPrompt = `--- SYSTEM INSTRUCTION ---
${systemInstruction}

--- CONTEXT BLOCK ---
${contextBlock}

--- CONVERSTATION HISTORY ---
${historyText || 'No previous messages.'}

--- CURRENT QUESTION ---
User: ${message}

--- ASSISTANT ANSWER ---`;

  let assistantReply = '';
  let tokenCount = 0;
  
  try {
    if (hasKey && ai) {
      // Invoke gemini-3.5-flash
      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: dynamicPrompt,
        config: {
          temperature: Number(temperature),
          systemInstruction: systemInstruction,
        }
      });
      assistantReply = response.text || 'Error parsing generated response.';
      // estimate tokens roughly as ~4 chars per token
      tokenCount = Math.round(dynamicPrompt.length / 4) + Math.round(assistantReply.length / 4);
    } else {
      // Fallback response for playground testing
      tokenCount = Math.round(dynamicPrompt.length / 4);
      if (!isGrounded && !/^(hi|hello|hey|greetings)/i.test(message.trim())) {
        assistantReply = "I do not have enough information to answer that question based on the knowledge base.";
      } else if (/^(hi|hello|hey|greetings)/i.test(message.trim())) {
        assistantReply = "Hello! I am a grounded reference guide assistant. Ask me questions about Password Reset, Account Deletion, Refunds, Rate Limits, Gemini models, or RAG thresholds.";
      } else {
        // Simple local pattern matching to simulate dynamic RAG
        const bestMatchedChunk = retrievedChunks[0];
        assistantReply = `[Simulated Grounded Response] Based on the context item: "${bestMatchedChunk.docTitle}", I can report that: ${bestMatchedChunk.text}`;
      }
    }
  } catch (err: any) {
    console.error('Gemini generateContent error', err);
    assistantReply = `Error in LLM Generation: ${err.message}. Ensure your API key is correctly configured.`;
  }

  // Update conversation session history
  session.history.push({ role: 'user', text: message, timestamp: new Date().toISOString() });
  session.history.push({ role: 'model', text: assistantReply, timestamp: new Date().toISOString() });
  session.lastActive = new Date().toISOString();

  res.json({
    reply: assistantReply,
    isGrounded,
    tokensUsed: tokenCount,
    retrievedChunks: isGrounded ? retrievedChunks : [],
    allCandidateScores: retrievedChunks, // Included to let students analyze threshold metrics
    promptUsed: dynamicPrompt,
    sessionId: session.sessionId,
    history: session.history
  });
});

// Clear active session chat history
router.post('/chat/clear', (req, res) => {
  const { sessionId = 'default_student_session' } = req.body;
  const session = dbSessions.get(sessionId);
  if (session) {
    session.history = [];
    dbSessions.set(sessionId, session);
  }
  res.json({ message: 'Session chat history purged successfully', sessionId });
});

export default router;
