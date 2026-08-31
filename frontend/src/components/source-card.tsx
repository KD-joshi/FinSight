import React from 'react';
import { Source } from '@/hooks/use-chat';
import { FileText, Building2, Tag } from 'lucide-react';
import { cn } from '@/lib/utils';

export function SourceCard({ source }: { source: Source }) {
  const { metadata } = source;
  const isHighQuality = metadata.rrf_score && metadata.rrf_score > 0.02;

  return (
    <div className={cn(
      "flex flex-col gap-2 p-3 rounded-xl border text-sm transition-all duration-300",
      isHighQuality ? "bg-primary/5 border-primary/20" : "bg-muted/30 border-border"
    )}>
      <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
        {metadata.ticker && (
          <span className="flex items-center gap-1 text-primary">
            <Building2 className="w-3 h-3" />
            {metadata.ticker}
          </span>
        )}
        {metadata.filing_type && (
          <span className="flex items-center gap-1">
            <Tag className="w-3 h-3" />
            {metadata.filing_type}
          </span>
        )}
        {metadata.source_file && (
          <span className="flex items-center gap-1 truncate max-w-[150px]" title={metadata.source_file}>
            <FileText className="w-3 h-3" />
            {metadata.source_file.split('/').pop()}
          </span>
        )}
      </div>
      
      <p className="text-foreground/80 line-clamp-3 text-xs leading-relaxed">
        {source.content}
      </p>
    </div>
  );
}
