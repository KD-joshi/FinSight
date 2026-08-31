import { useState, useCallback, useRef } from 'react';

export interface Source {
  content: string;
  metadata: {
    source_file?: string;
    ticker?: string;
    filing_type?: string;
    chunk_index?: number;
    rrf_score?: number;
    [key: string]: any;
  };
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  route?: string;
  executionTimeMs?: number;
  requiresConsent?: boolean;
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  
  const threadIdRef = useRef<string>("");
  const abortControllerRef = useRef<AbortController | null>(null);
  
  // Initialize on first use (client side only)
  if (typeof window !== 'undefined' && !threadIdRef.current) {
    threadIdRef.current = crypto.randomUUID();
  }

  const stop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsLoading(false);
      setIsUploading(false);
    }
  }, []);

  const sendMessage = useCallback(async (query: string) => {
    if (!query.trim()) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          thread_id: threadIdRef.current,
          max_retries: 1
        }),
        signal: controller.signal
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `API Error: ${response.statusText}`);
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        route: data.route,
        executionTimeMs: data.execution_time_ms,
        requiresConsent: data.requires_consent,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('Chat request aborted');
        return;
      }
      console.error('Chat error:', err);
      setError(err.message || 'Something went wrong');
      setMessages((prev) => [...prev, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'I encountered an error communicating with the FinSight server. Please ensure the backend is running.',
      }]);
    } finally {
      setIsLoading(false);
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  }, []);

  const handleConsent = useCallback(async (proceed: boolean) => {
    setIsLoading(true);
    setError(null);
    
    // Optimistically update the message to remove the consent requirement
    setMessages(prev => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg && lastMsg.requiresConsent) {
            return [
                ...prev.slice(0, -1),
                { ...lastMsg, requiresConsent: false, content: proceed ? "Consent granted. Searching web..." : "Consent denied." }
            ];
        }
        return prev;
    });

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/chat/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          thread_id: threadIdRef.current,
          proceed
        }),
        signal: controller.signal
      });

      if (!response.ok) throw new Error(`API Error: ${response.statusText}`);

      const data = await response.json();

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        route: data.route,
        executionTimeMs: data.execution_time_ms,
        requiresConsent: data.requires_consent,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('Consent resume aborted');
        return;
      }
      console.error('Consent resume error:', err);
      setError(err.message || 'Something went wrong');
    } finally {
      setIsLoading(false);
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  }, []);

  const uploadDocument = useCallback(async (file: File) => {
    const fileName = file.name.toLowerCase();
    if (!fileName.endsWith('.pdf') && !fileName.endsWith('.xlsx') && !fileName.endsWith('.csv')) {
        setError("Only PDF, XLSX, and CSV files are supported for upload right now.");
        return;
    }
    
    setIsUploading(true);
    setError(null);
    
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("thread_id", threadIdRef.current);
        
        const response = await fetch('http://127.0.0.1:8000/api/v1/upload', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });
        
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || `Upload failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        const assistantMessage: Message = {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: `Successfully uploaded ${file.name} and ingested ${data.chunks_ingested} chunks into the vector store. You can now ask questions about it!`,
            route: "upload_success"
        };
        
        setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
        if (err.name === 'AbortError') {
          console.log('Upload aborted');
          return;
        }
        console.error('Upload error:', err);
        setError(err.message || 'Failed to upload document');
    } finally {
        setIsUploading(false);
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
    }
  }, []);

  const clearChat = useCallback(() => {
    setMessages([]);
    threadIdRef.current = crypto.randomUUID(); // Reset memory thread
  }, []);

  const loadSession = useCallback(async (threadId: string) => {
    setIsLoading(true);
    setError(null);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch(`http://127.0.0.1:8000/api/v1/sessions/${threadId}`, {
        signal: controller.signal
      });
      if (!response.ok) throw new Error("Failed to fetch session history");
      
      const data = await response.json();
      threadIdRef.current = data.thread_id;
      
      const loadedMessages: Message[] = data.messages.map((msg: any) => ({
        id: crypto.randomUUID(),
        role: msg.role,
        content: msg.content
      }));
      
      setMessages(loadedMessages);
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('Session load aborted');
        return;
      }
      console.error('Error loading session:', err);
      setError(err.message || 'Failed to load session');
    } finally {
      setIsLoading(false);
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  }, []);

  return {
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
    threadId: threadIdRef.current
  };
}
