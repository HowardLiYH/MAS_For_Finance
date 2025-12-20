// Type definitions for PopAgent Dashboard

export interface AgentDecision {
  timestamp: string;
  iteration: number;
  agent_id: string;
  role: string;
  methods_available: string[];
  methods_selected: string[];
  selection_scores: Record<string, number>;
  preferences: Record<string, number>;
  context: MarketContext;
  reasoning: string | null;
  exploration_used: boolean;
}

export interface PipelineResult {
  pipeline_id: string;
  agents: {
    analyst: string;
    researcher: string;
    trader: string;
    risk: string;
  };
  methods: {
    analyst: string[];
    researcher: string[];
    trader: string[];
    risk: string[];
  };
  pnl: number;
  sharpe: number;
  success: boolean;
  execution_time_ms?: number;
}

export interface TransferLog {
  timestamp: string;
  role: string;
  source_agent_id: string;
  target_agent_ids: string[];
  transfer_tau: number;
  methods_transferred: string[];
  source_preferences: Record<string, number>;
}

export interface DiversityMetrics {
  role: string;
  selection_diversity: number;
  preference_entropy: number;
  unique_methods_used: number;
  total_methods_available: number;
}

export interface MarketContext {
  trend: 'bullish' | 'bearish' | 'neutral';
  volatility: number;
  regime: 'normal' | 'volatile' | 'quiet';
  recent_return?: number;
}

export interface IterationLog {
  experiment_id: string;
  iteration: number;
  timestamp: string;
  market_regime: string;
  market_context: MarketContext;
  agent_decisions: AgentDecision[];
  pipeline_results: PipelineResult[];
  best_pipeline_id: string | null;
  best_pnl: number;
  avg_pnl: number;
  knowledge_transfer: TransferLog | null;
  diversity_metrics: Record<string, DiversityMetrics>;
  iteration_duration_ms: number;
}

export interface ExperimentSummary {
  experiment_id: string;
  start_time: string;
  end_time: string;
  total_iterations: number;
  config: Record<string, any>;
  final_best_pnl: number;
  avg_pnl_first_10: number;
  avg_pnl_last_10: number;
  improvement: number;
  best_methods_by_role: Record<string, string[]>;
  method_popularity: Record<string, Record<string, number>>;
  transfer_count: number;
  final_diversity: Record<string, number>;
}

export interface Agent {
  id: string;
  role: 'analyst' | 'researcher' | 'trader' | 'risk';
  score: number;
  current_methods: string[];
  preferences: Record<string, number>;
  is_best: boolean;
}

export interface Population {
  role: string;
  agents: Agent[];
  best_agent_id: string | null;
  diversity: number;
}

export type Role = 'analyst' | 'researcher' | 'trader' | 'risk';

export const ROLE_COLORS: Record<Role, string> = {
  analyst: '#22d3ee',    // Cyan
  researcher: '#a78bfa', // Purple
  trader: '#10b981',     // Green
  risk: '#f59e0b',       // Amber
};

export const ROLE_LABELS: Record<Role, string> = {
  analyst: 'Analyst',
  researcher: 'Researcher',
  trader: 'Trader',
  risk: 'Risk Manager',
};
