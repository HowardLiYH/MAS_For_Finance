'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, ChevronRight, Lightbulb, TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';
import type { AgentDecision, Role } from '@/lib/types';

interface ReflectionPanelProps {
  decisions: AgentDecision[];
  selectedAgentId?: string;
}

export default function ReflectionPanel({ decisions, selectedAgentId }: ReflectionPanelProps) {
  const [expandedAgent, setExpandedAgent] = useState<string | null>(selectedAgentId || null);

  const roleColors: Record<Role, string> = {
    analyst: '#22d3ee',
    researcher: '#a78bfa',
    trader: '#10b981',
    risk: '#f59e0b',
  };

  // Get unique agents with their latest decision
  const agents = decisions.reduce((acc, decision) => {
    if (!acc.find(d => d.agent_id === decision.agent_id)) {
      acc.push(decision);
    }
    return acc;
  }, [] as AgentDecision[]);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Brain className="w-5 h-5 text-secondary" />
        <h2 className="text-lg font-semibold text-white">
          Agent Reasoning
        </h2>
      </div>
      <p className="text-xs text-muted">
        Click on an agent to see their decision rationale
      </p>

      <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
        {agents.map((decision) => {
          const isExpanded = expandedAgent === decision.agent_id;
          const color = roleColors[decision.role as Role] || '#fff';

          // Analyze the decision
          const topPrefs = Object.entries(decision.preferences)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 3);

          const selectedHighPref = decision.methods_selected.some(m =>
            topPrefs.some(([method]) => method === m)
          );

          return (
            <motion.div
              key={decision.agent_id}
              className="bg-surface/50 rounded-lg border border-white/5 overflow-hidden"
              initial={false}
            >
              {/* Header */}
              <button
                onClick={() => setExpandedAgent(isExpanded ? null : decision.agent_id)}
                className="w-full p-3 flex items-center justify-between hover:bg-white/5 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <span
                    className="w-8 h-8 rounded-lg flex items-center justify-center text-xs font-mono font-bold"
                    style={{ backgroundColor: `${color}20`, color }}
                  >
                    {decision.agent_id}
                  </span>
                  <div className="text-left">
                    <div className="text-sm font-medium text-white">
                      {decision.role.charAt(0).toUpperCase() + decision.role.slice(1)} Agent
                    </div>
                    <div className="text-xs text-muted">
                      Selected: {decision.methods_selected.join(', ')}
                    </div>
                  </div>
                </div>
                <ChevronRight
                  className={`w-4 h-4 text-muted transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                />
              </button>

              {/* Expanded content */}
              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="border-t border-white/5"
                  >
                    <div className="p-4 space-y-4">
                      {/* Reasoning */}
                      {decision.reasoning && (
                        <div className="flex gap-2">
                          <Lightbulb className="w-4 h-4 text-warning shrink-0 mt-0.5" />
                          <p className="text-sm text-white/80">
                            {decision.reasoning}
                          </p>
                        </div>
                      )}

                      {/* Context */}
                      <div className="bg-background/50 rounded-lg p-3">
                        <div className="text-xs text-muted mb-2">Market Context</div>
                        <div className="flex gap-4 text-xs">
                          <span className="flex items-center gap-1">
                            {decision.context.trend === 'bullish' ? (
                              <TrendingUp className="w-3 h-3 text-success" />
                            ) : decision.context.trend === 'bearish' ? (
                              <TrendingDown className="w-3 h-3 text-danger" />
                            ) : (
                              <span className="w-3 h-3 bg-muted rounded-full" />
                            )}
                            {decision.context.trend}
                          </span>
                          <span>
                            Vol: {(decision.context.volatility * 100).toFixed(0)}%
                          </span>
                          <span className="capitalize">
                            {decision.context.regime}
                          </span>
                        </div>
                      </div>

                      {/* Top Preferences */}
                      <div>
                        <div className="text-xs text-muted mb-2">Top Preferences</div>
                        <div className="flex flex-wrap gap-2">
                          {topPrefs.map(([method, score]) => (
                            <span
                              key={method}
                              className={`px-2 py-1 rounded text-xs font-mono ${
                                decision.methods_selected.includes(method)
                                  ? 'bg-success/20 text-success border border-success/30'
                                  : 'bg-white/5 text-muted'
                              }`}
                            >
                              {method}: {score.toFixed(2)}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Exploration indicator */}
                      {decision.exploration_used && (
                        <div className="flex items-center gap-2 text-xs text-secondary">
                          <AlertCircle className="w-3 h-3" />
                          Exploration mode active — trying less-preferred methods
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
