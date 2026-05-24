import React, { useState, useEffect, useRef } from 'react';
import {
  BookOpen,
  Database,
  Layers,
  Cpu,
  Search,
  MessageSquare,
  RefreshCw,
  Plus,
  Trash2,
  Sliders,
  CheckCircle,
  AlertTriangle,
  HelpCircle,
  Terminal,
  FileText,
  ChevronRight,
  Info,
  SlidersHorizontal,
  ThumbsUp,
  Sparkles,
  Play,
  Send,
  Code
} from 'lucide-react';

// Define Step list matching RAG implementation checklist
interface StepsItem {
  id: string;
  num: string;
  name: string;
  category: string;
}

const SIDEBAR_STEPS: StepsItem[] = [
  { id: 'logic', num: '01', name: 'RAG Architecture Workflow', category: 'Understanding' },
  { id: 'docs', num: '02', name: 'Document Manager Store', category: 'Phase 1' },
  { id: 'chunking', num: '03', name: 'Chunking Engine Logic', category: 'Phase 2' },
  { id: 'embeddings', num: '04', name: 'Vector Store & Embeddings', category: 'Phase 3' },
  { id: 'similarity', num: '05', name: 'Similarity Matrix Search', category: 'Phase 4' },
  { id: 'prompt', num: '06', name: 'dynamic Prompt Engineering', category: 'Phase 5' },
  { id: 'chat', num: '07', name: 'Playable Conversational Bot', category: 'Validation' }
];

interface Document {
  id: string;
  title: string;
  content: string;
}

interface Chunk {
  id: string;
  docId: string;
  docTitle: string;
  text: string;
  charCount: number;
  wordCount: number;
}

interface EmbeddedChunkSummary {
  chunkId: string;
  docTitle: string;
  text: string;
  vectorPreview: number[];
  dimension: number;
}

interface SearchResult {
  chunk: Chunk;
  score: number;
  isGrounded: boolean;
}

interface SessionMessage {
  role: 'user' | 'model';
  text: string;
  timestamp: string;
}

export default function App() {
  const [currentStep, setCurrentStep] = useState<string>('logic');
  const [hasRealApiKey, setHasRealApiKey] = useState<boolean>(false);
  const [appStateStatus, setAppStateStatus] = useState<any>({
    documentCount: 0,
    chunkCount: 0,
    embeddingCount: 0,
    sessionCount: 0
  });

  // Docs manager state
  const [docs, setDocs] = useState<Document[]>([]);
  const [newTitle, setNewTitle] = useState('');
  const [newContent, setNewContent] = useState('');
  const [isEditingDoc, setIsEditingDoc] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [editContent, setEditContent] = useState('');

  // Chunking form settings
  const [chunkSize, setChunkSize] = useState<number>(120);
  const [chunkOverlap, setChunkOverlap] = useState<number>(25);
  const [chunkStrategy, setChunkStrategy] = useState<'char' | 'word' | 'sentence'>('char');
  const [chunks, setChunks] = useState<Chunk[]>([]);

  // Embeddings generation responses
  const [embeddings, setEmbeddings] = useState<EmbeddedChunkSummary[]>([]);
  const [embMeta, setEmbMeta] = useState<{ isSimulated: boolean; dimension: number; count: number } | null>(null);

  // Similarity Search Query
  const [searchQuery, setSearchQuery] = useState<string>('How do I reset my password?');
  const [searchThreshold, setSearchThreshold] = useState<number>(0.70);
  const [searchK, setSearchK] = useState<number>(3);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [queryVectorPreview, setQueryVectorPreview] = useState<number[]>([]);
  const [searchIsSimulated, setSearchIsSimulated] = useState<boolean>(false);

  // Chat Play State
  const [sessionId] = useState<string>(() => 'session_' + Math.random().toString(36).substring(2, 9));
  const [chatMessage, setChatMessage] = useState<string>('');
  const [chatLog, setChatLog] = useState<SessionMessage[]>([]);
  const [chatThreshold, setChatThreshold] = useState<number>(0.70);
  const [chatK, setChatK] = useState<number>(3);
  const [chatTemp, setChatTemp] = useState<number>(0.2);
  const [lastPromptPreview, setLastPromptPreview] = useState<string>('');
  const [lastRetrievedChunks, setLastRetrievedChunks] = useState<any[]>([]);
  const [tokensUsed, setTokensUsed] = useState<number>(0);
  const [isGroundedChatState, setIsGroundedChatState] = useState<boolean | null>(null);

  // Async loaders
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [loadingEmbeddings, setLoadingEmbeddings] = useState(false);
  const [loadingSearch, setLoadingSearch] = useState(false);
  const [loadingChat, setLoadingChat] = useState(false);

  // Global Info Alert banner
  const [globalBannerText, setGlobalBannerText] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchGlobalState();
    loadAllDocuments();
    loadAllChunks();
    loadAllEmbeddingsSummary();
  }, []);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatLog]);

  const fetchGlobalState = async () => {
    try {
      const res = await fetch('/api/state');
      const data = await res.json();
      setHasRealApiKey(data.hasRealApiKey);
      setAppStateStatus({
        documentCount: data.documentCount,
        chunkCount: data.chunkCount,
        embeddingCount: data.embeddingCount,
        sessionCount: data.sessionCount
      });
      if (!data.hasRealApiKey) {
        setGlobalBannerText("⚠️ Running in Simulated Local Mode. Enter a GEMINI_API_KEY in the Settings > Secrets panel of your editor to perform live AI requests.");
      } else {
        setGlobalBannerText(null);
      }
    } catch (e) {
      console.error('Error fetching API status State', e);
    }
  };

  const loadAllDocuments = async () => {
    setLoadingDocs(true);
    try {
      const res = await fetch('/api/docs');
      const data = await res.json();
      setDocs(data.docs || []);
    } catch (e) {
      console.error('Failed to load documents list from backend', e);
    } finally {
      setLoadingDocs(false);
    }
  };

  const loadAllChunks = async () => {
    try {
      const res = await fetch('/api/chunks');
      const data = await res.json();
      setChunks(data.chunks || []);
    } catch (e) {
      console.error(e);
    }
  };

  const loadAllEmbeddingsSummary = async () => {
    try {
      const res = await fetch('/api/embeddings');
      const data = await res.json();
      setEmbeddings(data.embeddings || []);
      if (data.totalCount > 0) {
        setEmbMeta({
          isSimulated: !hasRealApiKey,
          dimension: data.embeddings[0]?.dimension || 768,
          count: data.totalCount
        });
      } else {
        setEmbMeta(null);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const resetAllDocumentsToDefaults = async () => {
    setLoadingDocs(true);
    try {
      const res = await fetch('/api/docs/reset', { method: 'POST' });
      const data = await res.json();
      setDocs(data.docs || []);
      setNewTitle('');
      setNewContent('');
      await loadAllChunks();
      await loadAllEmbeddingsSummary();
      await fetchGlobalState();
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingDocs(false);
    }
  };

  const handleAddDocument = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim() || !newContent.trim()) return;

    setLoadingDocs(true);
    try {
      const updatedDocs = [...docs, {
        id: 'doc_' + Date.now(),
        title: newTitle.trim(),
        content: newContent.trim()
      }];
      const res = await fetch('/api/docs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ docs: updatedDocs })
      });
      if (res.ok) {
        setNewTitle('');
        setNewContent('');
        await loadAllDocuments();
        await loadAllChunks();
        await loadAllEmbeddingsSummary();
        await fetchGlobalState();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingDocs(false);
    }
  };

  const handleDeleteDocument = async (id: string) => {
    setLoadingDocs(true);
    try {
      const updatedDocs = docs.filter(d => d.id !== id);
      await fetch('/api/docs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ docs: updatedDocs })
      });
      await loadAllDocuments();
      await loadAllChunks();
      await loadAllEmbeddingsSummary();
      await fetchGlobalState();
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingDocs(false);
    }
  };

  const handleTriggerChunking = async () => {
    setLoadingChunks(true);
    try {
      const res = await fetch('/api/chunks/make', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chunkSize, chunkOverlap, strategy: chunkStrategy })
      });
      const data = await res.json();
      setChunks(data.chunks || []);
      await loadAllEmbeddingsSummary(); // Clear out embeddings visualizer
      await fetchGlobalState();
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingChunks(false);
    }
  };

  const handleGenerateEmbeddings = async () => {
    setLoadingEmbeddings(true);
    try {
      const res = await fetch('/api/embeddings/generate', { method: 'POST' });
      await loadAllEmbeddingsSummary();
      await fetchGlobalState();
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingEmbeddings(false);
    }
  };

  const handleExecuteSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;

    setLoadingSearch(true);
    try {
      const res = await fetch('/api/similarity/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, threshold: searchThreshold, k: searchK })
      });
      const data = await res.json();
      setSearchResults(data.results || []);
      setQueryVectorPreview(data.queryVectorPreview || []);
      setSearchIsSimulated(data.isSimulated);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingSearch(false);
    }
  };

  const handleSendChatMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatMessage.trim()) return;

    setLoadingChat(true);
    // Optimistic user bubble append
    const userMessage: SessionMessage = {
      role: 'user',
      text: chatMessage,
      timestamp: new Date().toLocaleTimeString()
    };
    setChatLog(prev => [...prev, userMessage]);
    const originalQuery = chatMessage;
    setChatMessage('');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          message: originalQuery,
          threshold: chatThreshold,
          k: chatK,
          temperature: chatTemp
        })
      });
      const data = await res.json();
      
      setChatLog(data.history || []);
      setLastPromptPreview(data.promptUsed);
      setLastRetrievedChunks(data.retrievedChunksList || []);
      setTokensUsed(data.tokensUsed || 0);
      setIsGroundedChatState(data.isGrounded);
      await fetchGlobalState();
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingChat(false);
    }
  };

  const clearChatHistory = async () => {
    try {
      await fetch('/api/chat/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId })
      });
      setChatLog([]);
      setLastPromptPreview('');
      setLastRetrievedChunks([]);
      setTokensUsed(0);
      setIsGroundedChatState(null);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div id="app-viewport-container" className="min-h-screen bg-[#050505] text-[#e2e8f0] flex flex-col font-sans selection:bg-indigo-500/30 selection:text-white">
      {/* Top Header Navigation */}
      <header id="app-top-header" className="h-16 border-b border-white/10 flex items-center justify-between px-8 bg-[#0a0a0a] shrink-0 sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <div id="logo-badge" className="w-9 h-9 rounded bg-indigo-600 flex items-center justify-center font-black text-white text-sm tracking-wider shadow-[0_0_15px_rgba(79,70,229,0.3)]">
            RAG
          </div>
          <div>
            <h1 id="app-title-head" className="font-serif italic text-xl tracking-wide flex items-center gap-2">
              RAGify-AI
              <span className="text-xs font-sans not-italic font-bold text-white/40 border-l border-white/20 pl-2 uppercase tracking-[0.2em] hidden sm:inline">
                Assignment v1.0
              </span>
            </h1>
          </div>
        </div>

        {/* Dynamic State Metrics */}
        <div className="flex items-center gap-4 md:gap-6 text-xs text-white/50">
          <div className="hidden lg:flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="uppercase text-[10px] tracking-widest font-semibold text-white/40">Status: App Ready</span>
          </div>
          <div className="h-4 w-[1px] bg-white/10 hidden lg:block"></div>
          <div className="flex items-center gap-2 bg-white/5 border border-white/10 rounded-full py-1 px-3">
            <span className="text-white/30 text-[10px] uppercase font-mono">Active Key:</span>
            {hasRealApiKey ? (
              <span className="text-emerald-400 font-bold font-mono">Live Gemini AI</span>
            ) : (
              <span className="text-amber-400 font-bold font-mono">Simulated Local</span>
            )}
          </div>
        </div>
      </header>

      {/* Global Status Banner alert */}
      {globalBannerText && (
        <div id="notice-alert-banner" className="bg-amber-950/20 border-b border-amber-500/20 px-8 py-2.5 text-xs text-amber-300 flex items-center justify-between gap-4 transition-all animate-fade-in shrink-0">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>{globalBannerText}</span>
          </div>
          <button 
            id="dismiss-banner-btn"
            onClick={() => setGlobalBannerText(null)} 
            className="text-amber-400/60 hover:text-amber-200 transition-colors uppercase font-mono text-[9px] font-bold ml-auto"
          >
            Acknowledge
          </button>
        </div>
      )}

      {/* Main Sandbox Split Panel */}
      <div id="main-content-split" className="flex flex-1 overflow-hidden">
        
        {/* Step-by-Step Navigation Bar */}
        <nav id="sidebar-step-nav" className="w-72 border-r border-white/10 bg-[#080808] p-5 flex flex-col shrink-0 overflow-y-auto hidden md:flex">
          <div className="text-[10px] font-bold text-white/30 uppercase tracking-[0.2em] mb-4">
            Implementation steps
          </div>
          
          <ul id="steps-list" className="space-y-2 flex-1">
            {SIDEBAR_STEPS.map((step) => {
              const isActive = currentStep === step.id;
              return (
                <li key={step.id}>
                  <button
                    id={`sidebar-step-${step.id}`}
                    onClick={() => setCurrentStep(step.id)}
                    className={`w-full text-left flex gap-3 items-center p-2.5 rounded-lg border transition-all duration-150 ${
                      isActive
                        ? 'bg-indigo-950/40 border-indigo-500/40 text-white shadow-[0_2px_8px_rgba(79,70,229,0.08)]'
                        : 'border-transparent text-white/40 hover:text-white/80 hover:bg-white/5'
                    }`}
                  >
                    <span className={`text-[10px] w-6 h-6 flex items-center justify-center rounded font-mono border leading-none shrink-0 ${
                      isActive
                        ? 'bg-indigo-600 border-indigo-600 text-white font-bold'
                        : 'border-white/10 text-white/30'
                    }`}>
                      {step.num}
                    </span>
                    <div className="min-w-0">
                      <div className="text-[9px] uppercase font-bold text-white/20 tracking-wider">
                        {step.category}
                      </div>
                      <div className="text-xs font-semibold truncate leading-tight">
                        {step.name}
                      </div>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>

          {/* Quick reference guide info card */}
          <div id="sidebar-info-card" className="mt-8 pt-4 border-t border-white/10">
            <div className="p-4 rounded-lg bg-indigo-950/20 border border-indigo-500/20">
              <p className="text-[10px] text-indigo-400 font-black uppercase tracking-widest mb-1.5 flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5" /> Stack recommendation
              </p>
              <p className="text-[11px] text-indigo-200/70 leading-relaxed font-mono">
                FastAPI + Sentence-Transformers + SQLite
              </p>
              <div className="mt-2.5 pt-2 border-t border-indigo-500/10 text-[9px] text-white/40 flex justify-between items-center">
                <span>Ref model: gemini-3.5-flash</span>
                <span className="text-emerald-400 font-bold">docs.json</span>
              </div>
            </div>
          </div>
        </nav>

        {/* Content Area */}
        <main id="app-main-content" className="flex-1 overflow-y-auto bg-[#0a0a0a]/50 p-6 lg:p-10 flex flex-col gap-8 relative">
          
          {/* Mobile Step Selection Rail (Fall-back) */}
          <div id="mobile-navigation-selection" className="block md:hidden bg-[#121212] p-4 rounded-xl border border-white/10 shrink-0">
            <label className="block text-[10px] uppercase tracking-wider text-white/40 font-bold mb-1.5">Active Implementation Step</label>
            <select
              id="mobile-navigation-select"
              value={currentStep}
              onChange={(e) => setCurrentStep(e.target.value)}
              className="w-full bg-[#1e1e1e] border border-white/10 rounded-lg py-2 px-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {SIDEBAR_STEPS.map((step) => (
                <option key={step.id} value={step.id}>
                  Step {step.num}: {step.name} ({step.category})
                </option>
              ))}
            </select>
          </div>

          {/* RENDERING INDIVIDUAL ACTIVE STEP PANELS */}

          {/* STEP 1: RAG ARCHITECTURE WORKFLOW */}
          {currentStep === 'logic' && (
            <div id="step-panel-logic" className="flex flex-col gap-8 animate-fade-in">
              <section className="max-w-3xl">
                <span className="text-indigo-400 text-xs font-bold uppercase tracking-[0.25em] mb-2 block">Step 01 &middot; Core Workflow</span>
                <h2 className="text-3xl md:text-4xl font-serif font-light mb-4 text-white">The RAG Architectural Blueprint</h2>
                <p className="text-white/60 text-sm leading-relaxed">
                  In this module, study the complete end-to-end flow of Retrieval-Augmented Generation (RAG). 
                  Standard chatbots fail by relying solely on historical pre-trained weights, leading to hallucinations. 
                  RAG grounds responses in verified private repositories through vectorized semantic similarities.
                </p>
              </section>

              {/* Graphical workflow charts comparison */}
              <div id="workflow-comparison-container" className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                
                {/* Traditional Pipeline Card */}
                <div id="traditional-pipeline-card" className="bg-[#121212]/50 border border-white/5 rounded-xl p-6 flex flex-col justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-white/40 uppercase tracking-wider mb-4 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-red-400"></span> Traditional LLM Query
                    </h3>
                    <div className="space-y-4 py-6">
                      <div className="flex justify-between items-center bg-white/5 p-3 rounded border border-white/10">
                        <span className="text-xs font-bold uppercase text-white/50 font-mono">1. Prompt Entry</span>
                        <span className="text-xs text-indigo-300 font-semibold truncate max-w-[150px]">"Rate limit Details?"</span>
                      </div>
                      <div className="flex justify-center">
                        <ChevronRight className="w-5 h-5 text-red-500/40 rotate-90" />
                      </div>
                      <div className="flex justify-between items-center bg-white/5 p-3 rounded border border-white/10">
                        <span className="text-xs font-bold uppercase text-white/50 font-mono">2. Raw LLM Pass</span>
                        <span className="text-xs text-amber-300 truncate font-semibold italic">Guesses Answer ("hallucination")</span>
                      </div>
                    </div>
                  </div>
                  <div className="bg-red-950/10 border border-red-500/10 p-3.5 rounded-lg text-xs text-red-200/70">
                    <strong>Limitation:</strong> High risk of hallucinations and complete ignorance of internal enterprise datasets or policy overrides.
                  </div>
                </div>

                {/* Grounded RAG Pipeline Card */}
                <div id="grounded-rag-pipeline-card" className="bg-[#121212] border border-indigo-500/10 rounded-xl p-6 flex flex-col justify-between relative shadow-[0_4px_24px_rgba(79,70,229,0.05)]">
                  <div className="absolute top-0 right-0 bg-indigo-500/10 text-indigo-300 text-[9px] font-mono tracking-widest uppercase py-1 px-3.5 rounded-bl-xl font-bold border-l border-b border-indigo-500/10">
                    Recommended Spec
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-indigo-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-400"></span> RAG Grounded Query
                    </h3>
                    
                    <div className="space-y-2 py-2">
                      <div className="bg-white/5 p-2 rounded-lg border border-white/10 text-xs flex justify-between items-center">
                        <span className="text-white/40 font-mono text-[10px]">1. QUERY:</span>
                        <span className="text-white font-mono text-[11px]">"Where option delete?"</span>
                      </div>
                      <div className="text-center text-indigo-500">&darr;</div>
                      <div className="bg-white/5 p-2 rounded-lg border border-white/10 text-xs flex justify-between items-center">
                        <span className="text-white/40 font-mono text-[10px]">2. EMBED & MATCH:</span>
                        <span className="text-emerald-400 font-mono text-[11px] font-bold">Match Score: 0.82</span>
                      </div>
                      <div className="text-center text-indigo-500">&darr;</div>
                      <div className="bg-white/5 p-2 rounded-lg border border-white/10 text-xs flex justify-between items-center">
                        <span className="text-white/40 font-mono text-[10px]">3. CONSTRUCT PROMPT:</span>
                        <span className="text-yellow-400 font-mono text-[11px] truncate">Combine Context + History</span>
                      </div>
                      <div className="text-center text-indigo-500">&darr;</div>
                      <div className="bg-[#1a1a2e] p-2.5 rounded-lg border border-indigo-500/20 text-xs flex justify-between items-center">
                        <span className="text-indigo-300 font-mono text-[10px]">4. LLM PASS:</span>
                        <span className="text-indigo-200 font-sans italic font-medium">Faithfully references Context!</span>
                      </div>
                    </div>
                  </div>
                  <div className="bg-emerald-950/10 border border-emerald-500/20 p-3 rounded-lg text-xs text-emerald-200/80 mt-4">
                    <strong>Benefit:</strong> The LLM is forced to rely on retrieve data chunks rather than pre-trained guesses.
                  </div>
                </div>

              </div>

              {/* Detailed step breakdown block */}
              <div id="step-breakdown-card" className="bg-[#121212] border border-white/5 rounded-xl p-6">
                <h3 className="text-base font-semibold text-white mb-4">Workflow Requirements Reference Checklist</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <div className="bg-white/5 p-4 rounded-lg flex gap-3">
                    <Database className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-1">Knowledge db (docs.json)</h4>
                      <p className="text-[11px] text-white/50">Acts as local directory. Flat list of JSON matching objects containing specific company references.</p>
                    </div>
                  </div>
                  <div className="bg-white/5 p-4 rounded-lg flex gap-3">
                    <Sliders className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-1">Chunking Algorithm</h4>
                      <p className="text-[11px] text-white/50">Documents are divided into custom chunks matching character dimensions so similarities yield specific granular returns.</p>
                    </div>
                  </div>
                  <div className="bg-white/5 p-4 rounded-lg flex gap-3">
                    <Code className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-1">Vectors & Similarity</h4>
                      <p className="text-[11px] text-white/50">Calculations of Cosine similarity metrics with strict thresholds prevents LLMs from receiving noise blocks.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STEP 2: DOCUMENTS MANAGER */}
          {currentStep === 'docs' && (
            <div id="step-panel-docs" className="flex flex-col gap-8 animate-fade-in">
              <section className="max-w-3xl">
                <span className="text-indigo-400 text-xs font-bold uppercase tracking-[0.25em] mb-2 block">Step 02 &middot; Knowledge Base</span>
                <h2 className="text-4xl font-serif font-light mb-4">The Custom Client Docs Library</h2>
                <p className="text-white/60 text-sm leading-relaxed">
                  Every grounded response starts with structured data records. 
                  Edit, add, or reset documents inside your RAG Knowledge Store (`docs.json`) to control what rules and policies the LLM is permitted to consult.
                </p>
              </section>

              {/* Grid content containing Doc Library & Add Panel */}
              <div id="documents-grid-layout" className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* Visual inventory list */}
                <div id="doc-inventory-container" className="lg:col-span-2 bg-[#121212] border border-white/5 rounded-xl p-6 flex flex-col gap-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs font-bold text-white/30 uppercase tracking-widest block flex items-center gap-2">
                      <FileText className="w-4 h-4 text-indigo-500" /> Loaded Documents ({docs.length})
                    </span>
                    <button
                      id="reset-docs-btn"
                      onClick={resetAllDocumentsToDefaults}
                      disabled={loadingDocs}
                      className="px-3 py-1 border border-white/10 rounded text-xs text-white/50 hover:text-white hover:bg-white/5 transition-colors flex items-center gap-1.5 disabled:opacity-50"
                    >
                      <RefreshCw className={`w-3 h-3 ${loadingDocs ? 'animate-spin' : ''}`} />
                      Reset to Defaults
                    </button>
                  </div>

                  {loadingDocs ? (
                    <div className="py-20 text-center text-white/40 flex flex-col items-center gap-3">
                      <RefreshCw className="w-6 h-6 animate-spin text-indigo-500" />
                      Loading document records...
                    </div>
                  ) : docs.length === 0 ? (
                    <div className="py-20 text-center border-2 border-dashed border-white/10 rounded-lg text-white/40">
                      Empty knowledge database. Please add a document record to start.
                    </div>
                  ) : (
                    <div id="documents-grid" className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-[450px] overflow-y-auto pr-1">
                      {docs.map((doc) => (
                        <div
                          key={doc.id}
                          className="p-4 rounded-lg bg-[#0a0a0a] border border-white/10 flex flex-col justify-between group hover:border-indigo-500/20 transition-all"
                        >
                          <div>
                            <div className="flex justify-between items-start gap-2 mb-2">
                              <h4 className="text-xs font-semibold text-white tracking-wide truncate max-w-[180px]">
                                {doc.title}
                              </h4>
                              <button
                                id={`delete-doc-${doc.id}`}
                                onClick={() => handleDeleteDocument(doc.id)}
                                className="text-white/20 hover:text-red-400 p-0.5 rounded transition-colors"
                                title="Delete document"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            </div>
                            <p className="text-[11px] text-white/50 line-clamp-4 leading-relaxed italic">
                              "{doc.content}"
                            </p>
                          </div>
                          <div className="mt-4 pt-2 border-t border-white/5 text-[9px] text-white/30 flex justify-between items-center font-mono">
                            <span>ID: {doc.id}</span>
                            <span>{doc.content.length} chars</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Form to submit custom documents */}
                <div id="add-document-card" className="bg-[#121212] border border-white/5 rounded-xl p-6">
                  <h3 className="text-xs font-bold text-white/30 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <Plus className="w-4 h-4 text-emerald-400" /> Add Document Record
                  </h3>

                  <form id="add-doc-form" onSubmit={handleAddDocument} className="space-y-4">
                    <div>
                      <label className="block text-[10px] uppercase font-bold text-white/40 tracking-wider mb-1">
                        Document Title
                      </label>
                      <input
                        id="doc-title-input"
                        type="text"
                        placeholder="e.g. Server Rate Limits"
                        value={newTitle}
                        onChange={(e) => setNewTitle(e.target.value)}
                        className="w-full bg-[#050505] border border-white/10 rounded-lg p-2.5 text-xs text-white placeholder-white/30 tracking-wide focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-[10px] uppercase font-bold text-white/40 tracking-wider mb-1">
                        Content Body
                      </label>
                      <textarea
                        id="doc-content-input"
                        rows={6}
                        placeholder="e.g. Free Tier customers can make 60 requests per minute..."
                        value={newContent}
                        onChange={(e) => setNewContent(e.target.value)}
                        className="w-full bg-[#050505] border border-white/10 rounded-lg p-2.5 text-xs text-white placeholder-white/30 tracking-wide focus:outline-none focus:ring-2 focus:ring-indigo-500 leading-relaxed font-sans"
                        required
                      />
                    </div>

                    <button
                      id="save-doc-btn"
                      type="submit"
                      disabled={loadingDocs || !newTitle.trim() || !newContent.trim()}
                      className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-bold py-2.5 px-4 rounded-lg text-xs uppercase tracking-widest transition-all shadow-[0_0_15px_rgba(79,70,229,0.2)]"
                    >
                      {loadingDocs ? 'Updating Data...' : 'Insert Document'}
                    </button>
                  </form>

                  <div className="mt-6 p-3.5 bg-indigo-950/10 border border-indigo-500/10 rounded-lg text-[10px] text-indigo-300/80 leading-relaxed">
                    <strong>Assignment Guideline: </strong> Keep your document content descriptive and packed with factual statements so similarity searches resolve easily.
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* STEP 3: DOCUMENT CHUNKING */}
          {currentStep === 'chunking' && (
            <div id="step-panel-chunking" className="flex flex-col gap-8 animate-fade-in">
              <section className="max-w-3xl">
                <span className="text-indigo-400 text-xs font-bold uppercase tracking-[0.25em] mb-2 block">Step 03 &middot; Data Chunking</span>
                <h2 className="text-4xl font-serif font-light mb-4 text-white">Segmentizing Content Blocks</h2>
                <p className="text-white/60 text-sm leading-relaxed">
                  LMs have limited token boundaries. To search document contexts quickly and precisely, large paragraphs must be partition into discrete chunks with uniform lengths. 
                  Below, simulate text chunks segmenting on boundaries using sentence, character or word configurations.
                </p>
              </section>

              <div id="chunking-core-grid" className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* Configuration controls panel */}
                <div id="chunking-settings-card" className="bg-[#121212] border border-white/5 rounded-xl p-6">
                  <h3 className="text-xs font-bold text-white/30 uppercase tracking-widest mb-6 flex items-center gap-2">
                    <SlidersHorizontal className="w-4 h-4 text-indigo-400" /> Controls & Parameters
                  </h3>

                  <div className="space-y-6">
                    <div>
                      <div className="flex justify-between items-center mb-1.5">
                        <label className="text-[10px] uppercase font-bold text-white/40 tracking-wider">
                          Chunk Size
                        </label>
                        <span className="text-xs font-bold text-indigo-400 font-mono">{chunkSize} characters</span>
                      </div>
                      <input
                        id="chunk-size-slider"
                        type="range"
                        min="30"
                        max="400"
                        step="10"
                        value={chunkSize}
                        onChange={(e) => setChunkSize(Number(e.target.value))}
                        className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between items-center mb-1.5">
                        <label className="text-[10px] uppercase font-bold text-white/40 tracking-wider">
                          Overlap Size
                        </label>
                        <span className="text-xs font-bold text-indigo-400 font-mono">{chunkOverlap} characters</span>
                      </div>
                      <input
                        id="chunk-overlap-slider"
                        type="range"
                        min="0"
                        max="100"
                        step="5"
                        value={chunkOverlap}
                        onChange={(e) => setChunkOverlap(Number(e.target.value))}
                        className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                      />
                    </div>

                    <div>
                      <label className="block text-[10px] uppercase font-bold text-white/40 tracking-wider mb-1.5">
                        Parsing Strategy
                      </label>
                      <div className="grid grid-cols-3 gap-2">
                        {(['char', 'word', 'sentence'] as const).map((strat) => (
                          <button
                            key={strat}
                            id={`chunk-strat-${strat}`}
                            type="button"
                            onClick={() => setChunkStrategy(strat)}
                            className={`py-2 px-1 text-[10px] font-bold uppercase rounded-lg border transition-all ${
                              chunkStrategy === strat
                                ? 'bg-indigo-600/20 border-indigo-500 text-white shadow'
                                : 'bg-[#0a0a0a] border-white/5 text-white/40 hover:text-white/80 hover:bg-white/5'
                            }`}
                          >
                            {strat}
                          </button>
                        ))}
                      </div>
                    </div>

                    <button
                      id="run-chunking-btn"
                      onClick={handleTriggerChunking}
                      disabled={loadingChunks || docs.length === 0}
                      className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-bold py-2.5 px-4 rounded-lg text-xs uppercase tracking-widest transition-all shadow-[0_0_15px_rgba(79,70,229,0.2)] flex items-center justify-center gap-2"
                    >
                      {loadingChunks ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          Processing Chunks...
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-3.5 h-3.5" />
                          Run Chunking Model
                        </>
                      )}
                    </button>
                  </div>

                  <div className="mt-8 pt-4 border-t border-white/5">
                    <span className="text-[10px] font-bold text-white/30 uppercase tracking-widest block mb-2">Rationale</span>
                    <p className="text-[10px] text-white/40 tracking-wide leading-snug">
                       Overlaps ensure that context occurring on block splits (e.g. a key phone number or sentence) isn't bisected or ruined on vector mapping.
                    </p>
                  </div>
                </div>

                {/* Live Output list */}
                <div id="chunks-output-panel" className="lg:col-span-2 bg-[#121212] border border-white/5 rounded-xl p-6 flex flex-col gap-4">
                  <span className="text-xs font-bold text-white/30 uppercase tracking-widest mb-2 block flex items-center gap-2">
                    <Layers className="w-4 h-4 text-indigo-500" /> Segmented Chunks Pipeline Results ({chunks.length})
                  </span>

                  {chunks.length === 0 ? (
                    <div className="py-20 text-center text-white/40">
                      Compute custom chunk parameters to preview isolated segments.
                    </div>
                  ) : (
                    <div id="chunks-scroll-list" className="space-y-3.5 max-h-[480px] overflow-y-auto pr-2">
                      {chunks.map((ch, idx) => (
                        <div
                          key={ch.id}
                          className="p-3 bg-[#0a0a0a] border-l-2 border-indigo-500 rounded-r-lg border-y border-r border-white/5 group hover:border-white/15 transition-all"
                        >
                          <div className="flex justify-between items-center mb-1.5 text-[9px] text-white/30 font-mono uppercase">
                            <span className="text-indigo-400 font-bold font-sans">Chunk #{idx + 1} &middot; {ch.id}</span>
                            <span>From Doc: {ch.docTitle}</span>
                          </div>
                          <p className="text-xs text-white/80 italic leading-relaxed">
                            "{ch.text}"
                          </p>
                          <div className="mt-3 flex gap-4 text-[9px] font-mono text-white/30">
                            <span>Chars: {ch.charCount}</span>
                            <span>Words: {ch.wordCount}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

              </div>
            </div>
          )}

          {/* STEP 4: VECTOR STORE AND EMBEDDINGS */}
          {currentStep === 'embeddings' && (
            <div id="step-panel-embeddings" className="flex flex-col gap-8 animate-fade-in">
              <section className="max-w-3xl">
                <span className="text-indigo-400 text-xs font-bold uppercase tracking-[0.25em] mb-2 block">Step 04 &middot; Embedding Generation</span>
                <h2 className="text-4xl font-serif font-light mb-4">Continuous Vector Multi-space</h2>
                <p className="text-white/60 text-sm leading-relaxed">
                  Computers cannot search human language blocks using string matching or regular expressions. 
                  Instead, we translate our segmented chunks into high-density numeric lists of vectors. 
                  Below, pass the in-memory chunks into Gemini `gemini-embedding-2-preview` to lock them in a vector database index.
                </p>
              </section>

              <div id="embeddings-main-grid" className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* Generation triggers panel */}
                <div id="embedding-status-card" className="bg-[#121212] border border-white/5 rounded-xl p-6 flex flex-col justify-between">
                  <div>
                    <h3 className="text-xs font-bold text-white/30 uppercase tracking-widest mb-4 flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-emerald-400" /> Vector Database Control
                    </h3>

                    {embMeta ? (
                      <div className="space-y-4 mb-6">
                        <div className="bg-[#0a0a0a] border border-white/5 p-3.5 rounded-lg flex items-center justify-between">
                          <span className="text-[10px] uppercase text-white/40 font-mono">Index Status</span>
                          <span className="text-xs text-emerald-400 font-bold uppercase flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> Complete
                          </span>
                        </div>
                        <div className="bg-[#0a0a0a] border border-white/5 p-3.5 rounded-lg flex items-center justify-between">
                          <span className="text-[10px] uppercase text-white/40 font-mono">Dimensions</span>
                          <span className="text-xs font-bold font-mono text-indigo-300">{embMeta.dimension} vectors</span>
                        </div>
                        <div className="bg-[#0a0a0a] border border-white/5 p-3.5 rounded-lg flex items-center justify-between">
                          <span className="text-[10px] uppercase text-white/40 font-mono">Row Count</span>
                          <span className="text-xs font-bold font-mono text-indigo-300">{embMeta.count} chunks vectorized</span>
                        </div>
                      </div>
                    ) : (
                      <div className="p-4 bg-amber-950/20 border border-amber-500/20 rounded-lg text-xs text-amber-300 leading-relaxed mb-6">
                        <strong>Embeddings Missing:</strong> Chunks require vector indexing before similarity searches can execute.
                      </div>
                    )}

                    <button
                      id="generate-embeddings-btn"
                      onClick={handleGenerateEmbeddings}
                      disabled={loadingEmbeddings || chunks.length === 0}
                      className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-bold py-2.5 px-4 rounded-lg text-xs uppercase tracking-widest transition-all shadow-[0_0_15px_rgba(79,70,229,0.2)] flex items-center justify-center gap-2"
                    >
                      {loadingEmbeddings ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          Mapping Vectors...
                        </>
                      ) : (
                        <>
                          <RefreshCw className="w-3.5 h-3.5" />
                          {embeddings.length > 0 ? 'Force Recompute Embeddings' : 'Generate AI Embeddings'}
                        </>
                      )}
                    </button>
                  </div>

                  <div className="mt-8 pt-4 border-t border-white/5 text-[10px] text-white/40 leading-relaxed font-sans">
                    <strong>Reference SDK call:</strong><br/>
                    <code className="text-indigo-400 font-mono block mt-1">
                      await ai.models.embedContent(&#123; model: 'gemini-embedding-2-preview', contents: text &#125;)
                    </code>
                  </div>
                </div>

                {/* Table representation of vector preview */}
                <div id="vector-preview-panel" className="lg:col-span-2 bg-[#121212] border border-white/5 rounded-xl p-6 flex flex-col gap-4">
                  <span className="text-xs font-bold text-white/30 uppercase tracking-widest mb-2 block flex items-center gap-2">
                    <Database className="w-4 h-4 text-indigo-550" /> Flat Embeddings Matrix index list
                  </span>

                  {embeddings.length === 0 ? (
                    <div className="py-20 text-center text-white/40 italic">
                      Zero vector keys. Trigger generation to populate vector space index.
                    </div>
                  ) : (
                    <div id="embeddings-scroll-list" className="space-y-3.5 max-h-[420px] overflow-y-auto pr-1">
                      {embeddings.map((emb, idx) => (
                        <div
                          key={emb.chunkId}
                          className="p-3 bg-[#0a0a0a] border border-white/10 rounded-lg flex flex-col gap-2 group hover:border-indigo-500/20 transition-all"
                        >
                          <div className="flex justify-between items-center text-[10px] font-mono">
                            <span className="text-[10px] text-white/40 uppercase font-bold tracking-wider">Chunk Key: {emb.chunkId}</span>
                            <span className="text-[10px] text-emerald-400 uppercase font-black tracking-widest">Dimension: {emb.dimension} normalized decimals</span>
                          </div>
                          
                          <p className="text-xs text-white/50 italic font-medium truncate">
                            "{emb.text}"
                          </p>

                          {/* Visual vector blocks array */}
                          <div className="p-2.5 bg-black rounded border border-white/5 font-mono text-[10px] text-indigo-300 leading-none flex gap-1.5 flex-wrap items-center">
                            <span className="text-white/30 uppercase font-sans text-[8px] font-bold tracking-widest border border-white/10 px-1 py-0.5 rounded mr-1">vector output:</span>
                            [ {emb.vectorPreview.join(', ')}, ... ]
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

              </div>
            </div>
          )}

          {/* STEP 5: SIMILARITY SEARCH */}
          {currentStep === 'similarity' && (
            <div id="step-panel-similarity" className="flex flex-col gap-8 animate-fade-in">
              <section className="max-w-3xl">
                <span className="text-indigo-400 text-xs font-bold uppercase tracking-[0.25em] mb-2 block">Step 05 &middot; Similarity Search</span>
                <h2 className="text-4xl font-serif font-light mb-4">Vector Space Matrix Computations</h2>
                <p className="text-white/60 text-sm leading-relaxed">
                  When a user enters a query, we embed it using the identical model to map it in the exact same vector space. 
                  Then, compute the Cosine Similarity metric against all database vectors to rank and filter candidate sections matching scores over the configuration threshold.
                </p>
              </section>

              <div id="similarity-grid-root" className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* Search query controller inputs */}
                <div id="similarity-input-card" className="bg-[#121212] border border-white/5 rounded-xl p-6 flex flex-col gap-6">
                  <h3 className="text-xs font-bold text-white/30 uppercase tracking-widest flex items-center gap-2">
                    <Search className="w-4 h-4 text-indigo-400" /> Query parameters
                  </h3>

                  <form id="similarity-query-form" onSubmit={handleExecuteSearch} className="space-y-4">
                    <div>
                      <label className="block text-[10px] uppercase font-bold text-white/40 tracking-wider mb-1.5">
                        Ask keyword or phrase
                      </label>
                      <div className="relative">
                        <input
                          id="search-query-field"
                          type="text"
                          placeholder="e.g. Rate limits premium subscriber"
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="w-full bg-[#050505] border border-white/10 rounded-lg py-2.5 pl-3 pr-8 text-xs text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        />
                        <button
                          id="search-submit-inline"
                          type="submit"
                          className="absolute right-2 top-2 text-white/40 hover:text-white transition-colors"
                        >
                          <Search className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <label className="text-[9px] uppercase font-bold text-white/40 tracking-wider">
                            Threshold
                          </label>
                          <span className="text-[10px] font-bold text-indigo-400 font-mono">{searchThreshold}</span>
                        </div>
                        <input
                          id="search-threshold-slider"
                          type="range"
                          min="0.50"
                          max="0.95"
                          step="0.05"
                          value={searchThreshold}
                          onChange={(e) => setSearchThreshold(Number(e.target.value))}
                          className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <label className="text-[9px] uppercase font-bold text-white/40 tracking-wider">
                            Top K
                          </label>
                          <span className="text-[10px] font-bold text-indigo-400 font-mono">{searchK}</span>
                        </div>
                        <input
                          id="search-k-slider"
                          type="range"
                          min="1"
                          max="5"
                          step="1"
                          value={searchK}
                          onChange={(e) => setSearchK(Number(e.target.value))}
                          className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        />
                      </div>
                    </div>

                    <button
                      id="execute-search-btn"
                      type="submit"
                      disabled={loadingSearch || !searchQuery.trim()}
                      className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-bold py-2.5 px-4 rounded-lg text-xs uppercase tracking-widest transition-all shadow-[0_0_15px_rgba(79,70,229,0.2)] flex items-center justify-center gap-1.5"
                    >
                      {loadingSearch ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          Searching Matrix...
                        </>
                      ) : (
                        <>
                          <Play className="w-3.5 h-3.5 fill-current" />
                          Execute Search
                        </>
                      )}
                    </button>
                  </form>

                  <div className="pt-4 border-t border-white/5 space-y-3.5">
                    <span className="text-[10px] font-bold text-white/30 uppercase tracking-widest block">Query Vector Preview</span>
                    {queryVectorPreview.length > 0 ? (
                      <div className="p-2.5 bg-black rounded border border-white/5 font-mono text-[9px] text-emerald-400 truncate">
                        QA: [ {queryVectorPreview.join(', ')} ]
                      </div>
                    ) : (
                      <div className="text-[10px] text-white/30 italic">No search queried yet.</div>
                    )}
                  </div>
                </div>

                {/* Similarity output visual map charts */}
                <div id="similarity-results-panel" className="lg:col-span-2 bg-[#121212] border border-white/5 rounded-xl p-6 flex flex-col gap-4">
                  <span className="text-xs font-bold text-white/30 uppercase tracking-widest mb-2 block flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-emerald-500" /> Vector Matching Analysis Output
                  </span>

                  {embeddings.length === 0 ? (
                    <div className="py-20 text-center text-white/40 italic rounded-lg border-2 border-dashed border-white/10 p-4">
                      Please generate vectors in step 4 before testing similarity search query matching matrices.
                    </div>
                  ) : searchResults.length === 0 ? (
                    <div className="py-20 text-center text-white/40">
                      Submit a search query to view computed cosine thresholds.
                    </div>
                  ) : (
                    <div id="similarity-results-list" className="space-y-4 max-h-[480px] overflow-y-auto pr-1">
                      {searchResults.map((resItem, idx) => {
                        const isMatch = resItem.score >= searchThreshold;
                        return (
                          <div
                            key={resItem.chunk.id}
                            className={`p-3.5 rounded-lg border transition-all ${
                              isMatch
                                ? 'bg-[#0a0a0a] border-emerald-500/20 shadow-[0_2px_12px_rgba(16,185,129,0.02)]'
                                : 'bg-[#121212]/50 border-white/5 opacity-50'
                            }`}
                          >
                            <div className="flex justify-between items-center mb-2.5">
                              <div className="flex items-center gap-2">
                                <span className="text-[10px] font-sans font-black text-white/30 uppercase">Top Rank #{idx + 1}</span>
                                <span className="text-[10px] font-mono text-indigo-400 border border-indigo-500/20 px-1.5 py-0.5 rounded">
                                  {resItem.chunk.id}
                                </span>
                              </div>
                              <div className="flex items-center gap-2">
                                <span className="text-[10px] uppercase font-mono text-white/40">Similarity:</span>
                                <span className={`text-xs font-mono font-bold ${isMatch ? 'text-emerald-400' : 'text-amber-500'}`}>
                                  {resItem.score}
                                </span>
                                <span className={`text-[9px] uppercase font-black px-1.5 py-0.5 rounded ${
                                  isMatch ? 'bg-emerald-950/20 text-emerald-400 border border-emerald-500/20' : 'bg-white/5 text-white/30'
                                }`}>
                                  {isMatch ? 'PASS / grounded' : 'DISCARDED'}
                                </span>
                              </div>
                            </div>

                            <p className="text-xs text-white/70 italic leading-relaxed pl-2 border-l border-white/10">
                              "{resItem.chunk.text}"
                            </p>

                            {/* Scoring bar representation */}
                            <div className="mt-4 flex items-center gap-2.5">
                              <div className="flex-1 h-1.5 bg-[#050505] rounded-full overflow-hidden border border-white/5">
                                <div
                                  className={`h-full rounded-full transition-all duration-300 ${isMatch ? 'bg-emerald-400' : 'bg-amber-500'}`}
                                  style={{ width: `${Math.min(100, Math.max(0, resItem.score * 100))}%` }}
                                ></div>
                              </div>
                              <span className="text-[9px] text-white/40 font-mono tracking-widest uppercase">
                                Match Threshold {searchThreshold}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

              </div>
            </div>
          )}

          {/* STEP 6: DYNAMIC PROMPT DESIGN */}
          {currentStep === 'prompt' && (
            <div id="step-panel-prompt" className="flex flex-col gap-8 animate-fade-in font-sans">
              <section className="max-w-3xl">
                <span className="text-indigo-400 text-xs font-bold uppercase tracking-[0.25em] mb-2 block">Step 06 &middot; Intelligent Prompting</span>
                <h2 className="text-4xl font-serif font-light mb-4">Dynamic Grounded prompt construction</h2>
                <p className="text-white/60 text-sm leading-relaxed">
                  The magic of RAG happens when we reconstruct the system core parameters. 
                  Below, review the precise layout design of how retrieved context chunks, system guides, conversation logging, and the current question match together to build the instructions sent to Gemini.
                </p>
              </section>

              <div id="prompt-design-bento-grid" className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                
                {/* Visual architectural components */}
                <div id="prompt-hierarchy-view" className="bg-[#121212] border border-white/5 rounded-xl p-6 flex flex-col relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-4 opacity-5 text-indigo-300">
                    <Terminal className="w-24 h-24" />
                  </div>
                  
                  <h3 className="text-xs font-bold text-white/30 uppercase tracking-[0.2em] mb-6 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span> Prompt Component Blueprint
                  </h3>

                  <div className="space-y-4 font-mono text-xs">
                    <div className="p-3 bg-indigo-950/10 rounded border-l-2 border-indigo-500/50">
                      <span className="text-indigo-400 uppercase text-[9px] block mb-1 font-sans font-bold tracking-widest">A. System core instructions</span>
                      <p className="text-white/60 italic text-[11px] leading-relaxed">
                        "You MUST answer the question using strictly ONLY the provided Context block. Refuse to guess if empty."
                      </p>
                    </div>

                    <div className="p-3 bg-emerald-950/10 rounded border-l-2 border-emerald-500/50">
                      <span className="text-emerald-400 uppercase text-[9px] block mb-1 font-sans font-bold tracking-widest">B. Retrieve Context chunks</span>
                      <p className="text-white/60 text-[11px] leading-relaxed">
                        [Context Chunk #1] "Users can reset password from Settings..." <br/>
                        [Context Chunk #2] "API Rate limits are 60 req..."
                      </p>
                    </div>

                    <div className="p-3 bg-amber-950/10 rounded border-l-2 border-amber-500/50">
                      <span className="text-amber-400 uppercase text-[9px] block mb-1 font-sans font-bold tracking-widest">C. Conversational history</span>
                      <p className="text-white/60 text-[11px] leading-normal italic">
                        User: Hello | Assistant: Greetings! How may I assist you based on local documents?
                      </p>
                    </div>

                    <div className="p-3 bg-white/5 rounded border-l-2 border-white/40">
                      <span className="text-white/85 uppercase text-[9px] block mb-1 font-sans font-bold tracking-widest font-sans">D. User raw query</span>
                      <p className="text-white/90 font-semibold text-[11px]">
                        "What is refund claim policy?"
                      </p>
                    </div>
                  </div>
                </div>

                {/* Technical rationale constraints explanation card */}
                <div id="critical-constraints-card" className="flex flex-col gap-6">
                  
                  <div className="bg-[#121212] border border-white/5 rounded-xl p-6">
                     <h3 className="text-xs font-bold text-white/30 uppercase tracking-[0.2em] mb-4">Critical Constraints rationale</h3>
                     
                     <ul className="space-y-4">
                       <li className="flex items-start gap-3">
                         <div className="w-5 h-5 rounded-full border border-white/15 flex items-center justify-center shrink-0 mt-0.5">
                           <div className="w-1.5 h-1.5 rounded-full bg-indigo-400"></div>
                         </div>
                         <div>
                           <p className="text-xs font-semibold mb-0.5 text-white">Soft/Strict Threshold guard blocks</p>
                           <p className="text-[11px] text-white/40 leading-snug">
                             Prevents LLMs from hallucinating on non-relevant entries. If search yields a top score less than 0.75, standard greeting overrides or empty prompts block LLM completions.
                           </p>
                         </div>
                       </li>

                       <li className="flex items-start gap-3">
                         <div className="w-5 h-5 rounded-full border border-white/15 flex items-center justify-center shrink-0 mt-0.5">
                           <div className="w-1.5 h-1.5 rounded-full bg-indigo-400"></div>
                         </div>
                         <div>
                           <p className="text-xs font-semibold mb-0.5 text-white">Low LLM Temperature default (0.2)</p>
                           <p className="text-[11px] text-white/40 leading-snug">
                             Reduces creativity of generation vectors. Low temperature (0.2) increases deterministic and factual replication of contexts.
                           </p>
                         </div>
                       </li>
                     </ul>
                  </div>

                  <div className="flex-1 bg-black border border-white/10 rounded-xl p-5 font-mono text-[11px] leading-relaxed relative flex flex-col justify-between">
                     <div className="absolute top-3 right-4 flex gap-1.5">
                       <div className="w-2.5 h-2.5 rounded-full bg-red-500/50"></div>
                       <div className="w-2.5 h-2.5 rounded-full bg-amber-500/50"></div>
                       <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/50"></div>
                     </div>
                     <div>
                       <span className="text-indigo-400 font-bold block mb-2 font-sans text-[10px] uppercase tracking-widest">Python code construct snippet</span>
                       <span className="text-indigo-300">prompt_layout</span> = <span className="text-emerald-300">f"""</span><br/>
                       <span className="text-white/40">Instructions: Use ONLY provided details...</span><br/>
                       <span className="text-white/40">Context:</span><br/>
                       <span className="text-amber-200">{"{retrieved_chunks_context}"}</span><br/>
                       <span className="text-white/40">History:</span><br/>
                       <span className="text-amber-200">{"{conversation_history}"}</span><br/>
                       <span className="text-white/40">Question: {"{current_user_question}"}</span><br/>
                       <span className="text-emerald-300">"""</span><br/>
                       <span className="text-indigo-300">response</span> = <span className="text-blue-300">client.models.generateContent</span>(prompt_layout)
                     </div>
                  </div>

                </div>

              </div>
            </div>
          )}

          {/* STEP 7: CHAT PLAYGROUND */}
          {currentStep === 'chat' && (
            <div id="step-panel-chat" className="flex flex-col gap-8 animate-fade-in h-[calc(100vh-210px)] min-h-[500px]">
              
              {/* Core responsive chat and details panel wrap */}
              <div id="chat-sandbox-wrapper" className="grid grid-cols-1 xl:grid-cols-3 gap-8 flex-1 overflow-hidden min-h-0">
                
                {/* Actual Chat Dialog Terminal */}
                <div id="chat-terminal-col" className="xl:col-span-2 bg-[#121212] border border-white/5 rounded-xl flex flex-col overflow-hidden min-h-[400px]">
                  
                  {/* Chat Panel Header controls */}
                  <div className="p-4 border-b border-white/10 bg-[#0a0a0a] flex justify-between items-center shrink-0">
                    <div className="flex items-center gap-2">
                      <MessageSquare className="w-4 h-4 text-indigo-400" />
                      <span className="text-xs font-bold text-white tracking-wider font-mono">Session Terminal: #{sessionId.replace('session_', '')}</span>
                    </div>

                    <div className="flex items-center gap-3">
                      {chatLog.length > 0 && (
                        <button
                          id="clear-chat-btn"
                          onClick={clearChatHistory}
                          className="text-[10px] font-mono text-white/40 hover:text-red-400 px-2 py-1 rounded hover:bg-white/5 transition-colors border border-white/10"
                        >
                          Clear Session
                        </button>
                      )}
                      <span className="text-[10px] uppercase font-bold tracking-wider text-emerald-400 flex items-center gap-1 bg-emerald-950/20 border border-emerald-500/20 px-2 py-0.5 rounded-full">
                        <span className="w-1 h-1 bg-emerald-400 rounded-full animate-ping"></span> Live Agent
                      </span>
                    </div>
                  </div>

                  {/* Message logging block */}
                  <div id="chat-messages-container" className="flex-1 p-4 overflow-y-auto space-y-4 bg-black/25">
                    {chatLog.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center p-8 text-center text-white/30 gap-3">
                        <HelpCircle className="w-10 h-10 text-white/10 border border-white/10 p-2 rounded-full" />
                        <div>
                          <p className="text-sm font-semibold mb-1">Grounded RAG Dialogue Sandbox</p>
                          <p className="text-xs text-white/40 max-w-md">
                            Ask a targeted query that maps to your loaded documents (e.g. "How can I reset password?", "What is deletion period?", "refund claims"). Let the chatbot retrieve source chunks and verify answers.
                          </p>
                        </div>
                        <div className="grid grid-cols-2 gap-2 mt-4 max-w-sm">
                          <button
                            id="quick-query-pw"
                            onClick={() => {
                              setChatMessage("How do I reset my password?");
                            }}
                            className="text-[9px] font-mono uppercase bg-white/5 border border-white/10 p-2 hover:bg-white/10 rounded transition-all text-white/70"
                          >
                            Reset Password Options
                          </button>
                          <button
                            id="quick-query-claims"
                            onClick={() => {
                              setChatMessage("What is response time for refund claims?");
                            }}
                            className="text-[9px] font-mono uppercase bg-white/5 border border-white/10 p-2 hover:bg-white/10 rounded transition-all text-white/70"
                          >
                            Refund policy FAQ
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        {chatLog.map((msg, i) => {
                          const isAI = msg.role === 'model';
                          return (
                            <div
                              key={i}
                              className={`flex flex-col ${isAI ? 'items-start' : 'items-end'} animate-fade-in`}
                            >
                              <div className="text-[9px] font-mono uppercase text-white/30 mb-1 px-1">
                                {isAI ? 'Assistant RAG' : 'Student (You)'} &middot; {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : ''}
                              </div>
                              <div
                                className={`text-xs p-3 rounded-xl max-w-[85%] leading-relaxed ${
                                  isAI
                                    ? 'bg-[#1c1c1e] text-white/90 border border-white/15'
                                    : 'bg-indigo-600 text-white shadow-md'
                                }`}
                              >
                                {msg.text}
                              </div>
                            </div>
                          );
                        })}
                        {loadingChat && (
                          <div className="flex flex-col items-start animate-pulse">
                            <span className="text-[9px] font-mono text-white/30 uppercase mb-1">Composing grounded reply...</span>
                            <div className="p-3 bg-[#1c1c1e] border border-white/10 rounded-xl text-xs text-white/40 italic flex items-center gap-2">
                              <RefreshCw className="w-3.5 h-3.5 animate-spin text-indigo-500" />
                              Scanning vector indexes & submitting dynamic prompt...
                            </div>
                          </div>
                        )}
                        <div ref={messagesEndRef}></div>
                      </div>
                    )}
                  </div>

                  {/* Submission input block */}
                  <form id="chat-input-form" onSubmit={handleSendChatMessage} className="p-4 border-t border-white/10 bg-[#0a0a0a]/80 flex gap-2 shrink-0">
                    <input
                      id="chat-message-field"
                      type="text"
                      placeholder={embeddings.length === 0 ? "Ask a question (or map custom vectors in Step 4)..." : "Enter question based on loaded policy..."}
                      value={chatMessage}
                      disabled={loadingChat}
                      onChange={(e) => setChatMessage(e.target.value)}
                      className="flex-1 bg-black border border-white/10 rounded-lg py-3 px-4 text-xs text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
                    />
                    <button
                      id="chat-send-btn"
                      type="submit"
                      disabled={loadingChat || !chatMessage.trim()}
                      className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white p-3 rounded-lg flex items-center justify-center transition-colors shadow-md"
                    >
                      <Send className="w-4 h-4 fill-current text-white" />
                    </button>
                  </form>

                </div>

                {/* Right Pipeline Grounding Audit Trail panel */}
                <div id="grounding-audit-col" className="bg-[#121212]/30 border border-white/15 rounded-xl p-5 flex flex-col gap-5 overflow-y-auto">
                  
                  <div className="pb-3 border-b border-white/10 shrink-0">
                    <h3 className="text-xs font-bold text-white/30 uppercase tracking-[0.15em] flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-indigo-400" /> Grounding Diagnostics
                    </h3>
                  </div>

                  <div className="space-y-4">
                    
                    {/* Token & Grounding Boolean stats cards */}
                    <div className="grid grid-cols-2 gap-3 shrink-0">
                      <div className="p-3 bg-[#0a0a0a] border border-white/10 rounded-lg flex flex-col justify-between">
                        <span className="text-[9px] uppercase tracking-wider text-white/40 font-mono font-bold">Is Grounded?</span>
                        {isGroundedChatState === null ? (
                          <span className="text-xs font-bold text-white/30 font-mono italic">No query yet</span>
                        ) : isGroundedChatState ? (
                          <span className="text-xs font-bold text-emerald-400 font-mono uppercase flex items-center gap-1.5 mt-1">
                            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span> YES
                          </span>
                        ) : (
                          <span className="text-xs font-bold text-amber-500 font-mono uppercase flex items-center gap-1.5 mt-1">
                            <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span> REJECTED
                          </span>
                        )}
                      </div>

                      <div className="p-3 bg-[#0a0a0a] border border-white/10 rounded-lg flex flex-col justify-between">
                        <span className="text-[9px] uppercase tracking-wider text-white/40 font-mono font-bold font-sans">Prompt tokens</span>
                        <span className="text-xs font-bold text-indigo-300 font-mono mt-1">
                          {tokensUsed ? `~${tokensUsed} tokens` : '0 tokens'}
                        </span>
                      </div>
                    </div>

                    {/* Grounding Controls */}
                    <div id="grounding-controls-container" className="p-4 bg-[#0a0a0a] border border-white/5 rounded-lg space-y-4">
                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <label className="text-[9px] uppercase font-bold text-white/40 tracking-wider">
                            RAG Threshold
                          </label>
                          <span className="text-[10px] text-indigo-400 font-mono font-bold">{chatThreshold}</span>
                        </div>
                        <input
                          id="chat-threshold-slider"
                          type="range"
                          min="0.50"
                          max="0.95"
                          step="0.05"
                          value={chatThreshold}
                          onChange={(e) => setChatThreshold(Number(e.target.value))}
                          className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <label className="text-[9px] uppercase font-bold text-white/40 tracking-wider">
                            Chat Temperature
                          </label>
                          <span className="text-[10px] text-indigo-400 font-mono font-bold">{chatTemp}</span>
                        </div>
                        <input
                          id="chat-temp-slider"
                          type="range"
                          min="0.1"
                          max="0.8"
                          step="0.05"
                          value={chatTemp}
                          onChange={(e) => setChatTemp(Number(e.target.value))}
                          className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        />
                      </div>
                    </div>

                    {/* Retrieved source text blocks matched */}
                    <div id="retrieved-sources-container" className="space-y-2.5">
                      <span className="text-[10px] font-bold text-white/30 uppercase tracking-widest block">Retrieved source paragraphs ({lastRetrievedChunks.length})</span>
                      {lastRetrievedChunks.length === 0 ? (
                        <div className="text-[10px] text-white/30 italic p-3 bg-[#0a0a0a]/50 rounded-lg border border-white/5">
                          Sources metadata list empty. Ask a question to view match records.
                        </div>
                      ) : (
                        <div className="space-y-2 max-h-[220px] overflow-y-auto pr-1">
                          {lastRetrievedChunks.map((c, i) => (
                            <div key={i} className="p-2 bg-black border border-white/10 rounded text-[11px] leading-relaxed">
                              <div className="flex justify-between text-[9px] text-[#4f46e5] font-bold font-mono mb-1">
                                <span>CHUNK: {c.chunkId}</span>
                                <span>SCORE: {c.score}</span>
                              </div>
                              <p className="text-white/60">"{c.text}"</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Dynamic prompt generation text preview */}
                    <div id="prompt-preview-container" className="space-y-2.5">
                      <span className="text-[10px] font-bold text-white/30 uppercase tracking-widest block">Raw Grounded LLM Prompt Preview</span>
                      {lastPromptPreview ? (
                        <textarea
                          id="raw-prompt-textarea-preview"
                          readOnly
                          rows={10}
                          value={lastPromptPreview}
                          className="w-full bg-[#0a0a0a] border border-white/10 rounded-lg p-3 font-mono text-[9px] text-indigo-300 leading-relaxed resize-none focus:outline-none"
                        ></textarea>
                      ) : (
                        <div className="text-[10px] text-white/30 italic p-3 bg-[#0a0a0a]/50 rounded border border-white/5">
                          A raw prompt preview will render here listing system prefix layouts, retrieved segments, logging turns, and question payload details once queries have returned values.
                        </div>
                      )}
                    </div>

                  </div>

                </div>

              </div>
            </div>
          )}

        </main>
      </div>

      {/* Bottom Status bar */}
      <footer id="app-footer" className="h-10 border-t border-white/10 flex items-center justify-between px-8 bg-[#0a0a0a] text-[10px] font-semibold text-white/30 tracking-widest uppercase shrink-0">
        <div className="flex gap-4">
          <span>Session: {sessionId.replace('session_', '')}</span>
          <span>Database: In-Memory JSON Cache</span>
        </div>
        <div className="flex gap-4 items-center">
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></span> 
            System Authorized
          </span>
          <span className="hidden sm:inline">Build: 2026.05.24</span>
        </div>
      </footer>
    </div>
  );
}
