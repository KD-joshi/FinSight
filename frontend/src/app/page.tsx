'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '@/hooks/use-chat';
import { ChatMessage } from '@/components/chat-message';
import { Sparkles, Paperclip, Send, PanelLeft, Plus, TrendingUp, Square } from 'lucide-react';
import { motion } from 'framer-motion';

export default function FinSightChat() {
  const { 
    messages, 
    isLoading, 
    isUploading, 
    error, 
    sendMessage, 
    handleConsent, 
    uploadDocument, 
    clearChat,
    loadSession,
    stop,
    threadId
  } = useChat();
  
  const [input, setInput] = useState('');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [sessions, setSessions] = useState<{thread_id: string, last_updated: string, title?: string}[]>([]);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/sessions');
      if (res.ok) {
        const data = await res.json();
        
        // The backend already sorts sessions by newest first
        const sortedData = data;
        
        // We avoid fetching the full history for EVERY session here (which causes the N+1 API spam).
        // Instead, we just use the thread_id as the title for now.
        const sessionsWithTitles = sortedData.map((session: any) => ({
          ...session,
          title: session.session_name || (session.thread_id.substring(0, 8) + '...')
        }));
        setSessions(sessionsWithTitles);
      }
    } catch (e) {
      console.error("Failed to fetch sessions", e);
    }
  };

  // Auto-scroll to bottom and refresh sessions list when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    if (!isLoading && messages.length > 0) {
      fetchSessions();
    }
  }, [messages, isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    sendMessage(input);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadDocument(file);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex h-screen bg-background overflow-hidden relative">
      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-muted/30 border-r border-border/50 transform transition-transform duration-300 ease-in-out flex flex-col ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="p-4 flex items-center justify-between border-b border-border/50">
          <h2 className="font-semibold text-foreground/80">Chat History</h2>
          <button onClick={() => setIsSidebarOpen(false)} className="p-1 rounded-md hover:bg-muted text-muted-foreground">
            <PanelLeft className="w-5 h-5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {sessions.map((session, i) => {
            const isActive = session.thread_id === threadId;
            return (
              <button 
                key={i}
                onClick={() => {
                  loadSession(session.thread_id);
                  setIsSidebarOpen(false);
                }}
                className={`w-full text-left p-3 rounded-lg text-sm transition-all duration-200 truncate ${
                  isActive 
                    ? 'bg-primary/10 text-primary border border-primary/20 font-medium shadow-sm' 
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground border border-transparent'
                }`}
              >
                {session.title || (session.thread_id.substring(0, 8) + '...')}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Content Area */}
      <div className={`flex-1 flex flex-col min-w-0 transition-all duration-300 ${isSidebarOpen ? 'ml-64' : 'ml-0'}`}>
        {/* Header */}
        <header className="flex-none p-4 md:p-6 flex items-center justify-between z-10 bg-gradient-to-b from-background via-background to-transparent">
          <div className="flex items-center gap-3">
            <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 rounded-xl bg-muted/50 text-muted-foreground hover:text-foreground transition-colors">
              <PanelLeft className="w-5 h-5" />
            </button>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-purple-500 bg-clip-text text-transparent">
                FinSight
              </h1>
              <p className="text-xs text-muted-foreground font-medium">Financial Intelligence Agent</p>
            </div>
          </div>
          <button
            onClick={() => {
              clearChat();
              fetchSessions(); // refresh history when creating a new chat
            }}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-full bg-muted/50 text-muted-foreground hover:text-foreground hover:bg-muted transition-all border border-transparent hover:border-border"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>
        </header>

        {/* Chat Area */}
        <main className="flex-1 overflow-y-auto px-4 py-8 custom-scrollbar">
          <div className="max-w-5xl mx-auto flex flex-col w-full">
            {/* Always Visible Header */}
            <div className={`flex flex-col items-center justify-center text-center space-y-8 transition-all duration-500 ${messages.length === 0 ? 'mt-16 mb-8' : 'mt-4 mb-12'}`}>
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary/20 to-purple-500/20 flex items-center justify-center mb-4 ring-1 ring-white/10"
              >
              <Sparkles className="w-10 h-10 text-primary" />
            </motion.div>
            <div className="space-y-4">
              <h2 className="text-3xl font-bold text-foreground">
                AI Financial Analyst
              </h2>
              <p className="text-muted-foreground text-lg max-w-lg mx-auto leading-relaxed">
                Ask me about SEC filings, corporate strategy, risk factors, or financial performance for top companies.
              </p>
            </div>
          </div>

          {messages.length === 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl mx-auto mt-8">
              {[
                "What were Apple's major strategic priorities in 2024?",
                "What are the biggest risk factors for Google?",
                "Compare Microsoft and Tesla's approach to AI.",
                "How much cash flow did Apple generate last quarter?"
              ].map((suggestion, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(suggestion)}
                  className="p-4 rounded-xl border border-border/50 bg-muted/20 hover:bg-muted/40 hover:border-border text-sm text-left transition-all duration-200 text-foreground/80"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          ) : (
            <div className="space-y-6 pb-20 w-full">
            {messages.map((message) => (
              <div key={message.id}>
                <ChatMessage message={message} />
                
                {/* Render Consent UI if this message requires it */}
                {message.requiresConsent && (
                  <div className="w-full max-w-5xl mx-auto pl-16 pr-4 mt-2 mb-6">
                    <div className="p-4 rounded-xl bg-primary/10 border border-primary/20 flex flex-col md:flex-row gap-4 items-center justify-between">
                      <p className="text-sm text-foreground/90 font-medium">
                        To answer this, I need to search the web for external documents.
                      </p>
                      <div className="flex gap-2 w-full md:w-auto">
                        <button
                          onClick={() => handleConsent(false)}
                          disabled={isLoading}
                          className="px-4 py-2 rounded-lg bg-muted text-foreground text-sm hover:bg-muted/80 disabled:opacity-50 transition-colors flex-1 md:flex-none"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={() => handleConsent(true)}
                          disabled={isLoading}
                          className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 disabled:opacity-50 transition-colors flex-1 md:flex-none shadow-lg shadow-primary/20"
                        >
                          Proceed
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
            
            {isLoading && (
              <div className="flex w-full max-w-5xl mx-auto gap-4 p-6">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0 shadow-sm animate-pulse">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1 flex items-center">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 rounded-full bg-primary/50 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 rounded-full bg-primary/50 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 rounded-full bg-primary/50 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
            
            {error && (
              <div className="w-full max-w-5xl mx-auto p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 text-sm text-center">
                {error}
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        )}
        </div>
      </main>

      {/* Input Area */}
      <div className="flex-none p-4 bg-gradient-to-t from-background via-background to-transparent pt-10">
        <div className="max-w-5xl mx-auto relative">
          <form
            onSubmit={handleSubmit}
            className="relative flex items-end overflow-hidden rounded-2xl bg-muted/30 border border-border/50 focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/50 transition-all shadow-lg"
          >
            <input 
              type="file" 
              accept=".pdf,.xlsx,.csv" 
              ref={fileInputRef} 
              className="hidden" 
              onChange={handleFileChange} 
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading || isUploading}
              title="Upload Document"
              className="absolute left-3 bottom-3 p-2 rounded-xl text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors disabled:opacity-50"
            >
              {isUploading ? (
                <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              ) : (
                <Paperclip className="w-5 h-5" />
              )}
            </button>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading || isUploading}
              placeholder={isUploading ? "Uploading document..." : "Ask about SEC filings, revenue, risks..."}
              className="w-full max-h-[200px] min-h-[60px] py-4 pl-14 pr-14 bg-transparent border-none resize-none focus:outline-none focus:ring-0 text-foreground placeholder:text-muted-foreground disabled:opacity-50 custom-scrollbar"
              rows={1}
            />
            <div className="absolute right-3 bottom-3">
              {isLoading || isUploading ? (
                <button
                  type="button"
                  onClick={stop}
                  className="p-2 rounded-xl bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-colors"
                  title="Stop Generating"
                >
                  <Square className="w-5 h-5 fill-current" />
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!input.trim()}
                  className="p-2 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:hover:bg-primary transition-colors"
                  title="Send Message"
                >
                  <Send className="w-5 h-5" />
                </button>
              )}
            </div>
          </form>
          <div className="text-center mt-3">
            <p className="text-[11px] text-muted-foreground/60">
              FinSight AI can make mistakes. Verify important financial information with source SEC filings.
            </p>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}
