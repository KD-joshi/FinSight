import React from 'react';
import { Message } from '@/hooks/use-chat';
import { Bot, User, Clock, Route, FileText, Link as LinkIcon } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

export function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === 'user';

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "flex w-full max-w-4xl mx-auto gap-4 p-6 rounded-2xl mb-4",
        isUser ? "bg-muted/20" : "bg-transparent"
      )}
    >
      <div className={cn(
        "w-10 h-10 rounded-full flex items-center justify-center shrink-0 shadow-sm mt-1",
        isUser ? "bg-primary/20 text-primary" : "bg-gradient-to-br from-indigo-500 to-purple-600 text-white"
      )}>
        {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
      </div>
      
      <div className="flex-1 space-y-4 min-w-0">
        <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-muted prose-pre:border prose-pre:border-border max-w-none text-foreground/90">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </div>
        
        {message.sources && message.sources.length > 0 && (
          <div className="pt-4 border-t border-border/50">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
              Sources & Evidence
            </p>
            <div className="flex flex-col gap-2">
              {(() => {
                const uniqueSources = new Map();
                message.sources.forEach(s => {
                  const url = s.metadata.source || s.metadata.source_file;
                  if (url && !uniqueSources.has(url)) {
                    uniqueSources.set(url, s.metadata.title || url);
                  }
                });
                return Array.from(uniqueSources.entries()).map(([url, title], i) => (
                  <a 
                    key={i} 
                    href={url.startsWith('http') ? url : '#'} 
                    target={url.startsWith('http') ? "_blank" : "_self"}
                    rel="noopener noreferrer" 
                    className="text-sm text-primary hover:underline flex items-center gap-2"
                  >
                    {url.startsWith('http') ? <LinkIcon className="w-4 h-4 shrink-0" /> : <FileText className="w-4 h-4 shrink-0" />}
                    <span className="truncate">{title}</span>
                  </a>
                ));
              })()}
            </div>
          </div>
        )}

        {(message.executionTimeMs || message.route) && (
          <div className="flex items-center gap-4 pt-2 text-xs text-muted-foreground font-mono">
            {message.executionTimeMs && (
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {(message.executionTimeMs / 1000).toFixed(2)}s
              </span>
            )}
            {message.route && (
              <span className="flex items-center gap-1">
                <Route className="w-3 h-3" />
                {message.route} route
              </span>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}
