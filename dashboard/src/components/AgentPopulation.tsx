'use client';

import { motion } from 'framer-motion';
import { Star, TrendingUp, TrendingDown } from 'lucide-react';
import type { AgentDecision, Role, ROLE_COLORS } from '@/lib/types';

interface AgentCardProps {
  decision: AgentDecision;
  isBest: boolean;
  roleColor: string;
}

function AgentCard({ decision, isBest, roleColor }: AgentCardProps) {
  const score = Object.values(decision.selection_scores).reduce((a, b) => a + b, 0) /
    Math.max(Object.values(decision.selection_scores).length, 1);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ scale: 1.02 }}
      className={`agent-card p-3 ${isBest ? 'best' : ''}`}
      style={{ borderColor: isBest ? '#10b981' : `${roleColor}30` }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span
            className="text-sm font-mono font-semibold"
            style={{ color: roleColor }}
          >
            {decision.agent_id}
          </span>
          {isBest && (
            <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
          )}
        </div>
        <span className="text-xs text-muted">
          {score.toFixed(2)}
        </span>
      </div>

      {/* Selected Methods */}
      <div className="flex flex-wrap gap-1 mb-2">
        {decision.methods_selected.map((method) => (
          <span
            key={method}
            className="method-pill selected text-[10px]"
            style={{ borderColor: `${roleColor}50`, color: roleColor }}
          >
            {method}
          </span>
        ))}
      </div>

      {/* Exploration indicator */}
      {decision.exploration_used && (
        <div className="text-[10px] text-secondary/70 flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-secondary animate-pulse" />
          Exploring
        </div>
      )}
    </motion.div>
  );
}

interface PopulationGridProps {
  role: Role;
  decisions: AgentDecision[];
  bestAgentId: string | null;
}

function PopulationGrid({ role, decisions, bestAgentId }: PopulationGridProps) {
  const roleColors: Record<Role, string> = {
    analyst: '#22d3ee',
    researcher: '#a78bfa',
    trader: '#10b981',
    risk: '#f59e0b',
  };

  const roleLabels: Record<Role, string> = {
    analyst: 'ANALYSTS',
    researcher: 'RESEARCHERS',
    trader: 'TRADERS',
    risk: 'RISK MANAGERS',
  };

  const color = roleColors[role];

  return (
    <div className="bg-surface/30 rounded-xl p-4 border border-white/5">
      <div className="flex items-center justify-between mb-3">
        <h3
          className="text-sm font-semibold tracking-wider"
          style={{ color }}
        >
          {roleLabels[role]}
        </h3>
        <span className="text-xs text-muted">
          {decisions.length} agents
        </span>
      </div>

      <div className="grid grid-cols-5 gap-2">
        {decisions.map((decision) => (
          <AgentCard
            key={decision.agent_id}
            decision={decision}
            isBest={decision.agent_id === bestAgentId}
            roleColor={color}
          />
        ))}
      </div>
    </div>
  );
}

interface AgentPopulationProps {
  decisions: AgentDecision[];
  bestAgentIds?: Record<string, string>;
}

export default function AgentPopulation({ decisions, bestAgentIds = {} }: AgentPopulationProps) {
  const roles: Role[] = ['analyst', 'researcher', 'trader', 'risk'];

  const groupedByRole = roles.reduce((acc, role) => {
    acc[role] = decisions.filter(d => d.role === role);
    return acc;
  }, {} as Record<Role, AgentDecision[]>);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">
          Agent Populations
        </h2>
        <div className="flex items-center gap-4 text-xs text-muted">
          <span className="flex items-center gap-1">
            <Star className="w-3 h-3 text-yellow-400 fill-yellow-400" />
            Best performer
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-secondary animate-pulse" />
            Exploring
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {roles.map((role) => (
          <PopulationGrid
            key={role}
            role={role}
            decisions={groupedByRole[role] || []}
            bestAgentId={bestAgentIds[role] || null}
          />
        ))}
      </div>
    </div>
  );
}
