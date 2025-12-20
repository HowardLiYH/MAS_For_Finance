'use client';

import { motion } from 'framer-motion';
import { Zap, ArrowRight, Users } from 'lucide-react';
import type { TransferLog, Role } from '@/lib/types';

interface KnowledgeTransferProps {
  transfers: TransferLog[];
}

export default function KnowledgeTransfer({ transfers }: KnowledgeTransferProps) {
  const roleColors: Record<string, string> = {
    analyst: '#22d3ee',
    researcher: '#a78bfa',
    trader: '#10b981',
    risk: '#f59e0b',
  };

  if (transfers.length === 0) {
    return (
      <div className="bg-surface/30 rounded-xl p-6 border border-white/5 text-center">
        <Users className="w-8 h-8 text-muted mx-auto mb-2" />
        <p className="text-sm text-muted">No knowledge transfers yet</p>
        <p className="text-xs text-muted/60 mt-1">
          Transfers occur every 10 iterations
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Zap className="w-5 h-5 text-secondary" />
        <h2 className="text-lg font-semibold text-white">
          Knowledge Transfers
        </h2>
        <span className="text-xs text-muted bg-white/5 px-2 py-0.5 rounded-full">
          {transfers.length} total
        </span>
      </div>

      <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
        {transfers.map((transfer, idx) => {
          const color = roleColors[transfer.role] || '#fff';

          return (
            <motion.div
              key={`${transfer.timestamp}-${idx}`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.05 }}
              className="bg-surface/50 rounded-lg p-4 border border-white/5"
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span
                    className="w-6 h-6 rounded flex items-center justify-center text-xs font-bold"
                    style={{ backgroundColor: `${color}20`, color }}
                  >
                    {transfer.role.charAt(0).toUpperCase()}
                  </span>
                  <span className="text-sm font-medium capitalize">
                    {transfer.role} Transfer
                  </span>
                </div>
                <span className="text-xs text-muted">
                  τ = {transfer.transfer_tau}
                </span>
              </div>

              {/* Transfer visualization */}
              <div className="flex items-center gap-3">
                {/* Source agent */}
                <div
                  className="px-3 py-2 rounded-lg text-center"
                  style={{ backgroundColor: `${color}20`, border: `1px solid ${color}40` }}
                >
                  <div className="text-xs text-muted mb-1">Best</div>
                  <div className="font-mono font-bold" style={{ color }}>
                    {transfer.source_agent_id}
                  </div>
                </div>

                {/* Arrow with animation */}
                <motion.div
                  className="flex items-center gap-1"
                  animate={{ x: [0, 5, 0] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <div className="w-8 h-0.5" style={{ backgroundColor: color }} />
                  <ArrowRight className="w-4 h-4" style={{ color }} />
                </motion.div>

                {/* Target agents */}
                <div className="flex flex-wrap gap-2">
                  {transfer.target_agent_ids.map((targetId) => (
                    <motion.div
                      key={targetId}
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="px-2 py-1 rounded bg-white/5 text-xs font-mono text-muted"
                    >
                      {targetId}
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Methods transferred */}
              {transfer.methods_transferred.length > 0 && (
                <div className="mt-3 pt-3 border-t border-white/5">
                  <div className="text-xs text-muted mb-2">Methods Transferred</div>
                  <div className="flex flex-wrap gap-1">
                    {transfer.methods_transferred.map((method) => (
                      <span
                        key={method}
                        className="px-2 py-0.5 rounded text-[10px] font-mono"
                        style={{ backgroundColor: `${color}15`, color }}
                      >
                        {method}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
