// API client for PopAgent Dashboard

import type { IterationLog, ExperimentSummary } from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function fetchExperiments(): Promise<ExperimentSummary[]> {
  const res = await fetch(`${API_BASE}/experiments`);
  if (!res.ok) throw new Error('Failed to fetch experiments');
  return res.json();
}

export async function fetchExperiment(experimentId: string): Promise<{
  metadata: any;
  iterations: IterationLog[];
  summary: ExperimentSummary;
}> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}`);
  if (!res.ok) throw new Error('Failed to fetch experiment');
  return res.json();
}

export async function fetchIteration(
  experimentId: string,
  iteration: number
): Promise<IterationLog> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/iterations/${iteration}`);
  if (!res.ok) throw new Error('Failed to fetch iteration');
  return res.json();
}

// For static demo data loading
export async function loadDemoData(): Promise<IterationLog[]> {
  try {
    const res = await fetch('/data/demo_iterations.json');
    if (!res.ok) return generateMockIterations();
    return res.json();
  } catch {
    return generateMockIterations();
  }
}

// Generate mock data for demo
function generateMockIterations(): IterationLog[] {
  const iterations: IterationLog[] = [];
  const roles = ['analyst', 'researcher', 'trader', 'risk'] as const;

  const inventories: Record<string, string[]> = {
    analyst: ['RSI', 'MACD', 'BollingerBands', 'HMM_Regime', 'KalmanFilter', 'STL', 'Wavelet', 'ADX', 'Stochastic', 'VolatilityClustering'],
    researcher: ['ARIMA', 'LSTM', 'TemporalFusion', 'BootstrapEnsemble', 'QuantileRegression', 'GARCH', 'RandomForest', 'GradientBoosting'],
    trader: ['AggressiveMarket', 'PassiveLimit', 'TWAP', 'VWAP', 'KellyCriterion', 'VolatilityScaled', 'MomentumEntry', 'ContrarianEntry'],
    risk: ['MaxDrawdown', 'DailyStopLoss', 'VaRLimit', 'MaxLeverage', 'VolatilityAdjusted', 'TrailingStop', 'ConcentrationLimit'],
  };

  for (let i = 1; i <= 50; i++) {
    const trend = i % 3 === 0 ? 'bearish' : i % 5 === 0 ? 'neutral' : 'bullish';
    const volatility = 0.2 + Math.random() * 0.4;
    const regime = volatility > 0.4 ? 'volatile' : volatility < 0.25 ? 'quiet' : 'normal';

    const agentDecisions: any[] = [];

    for (const role of roles) {
      for (let a = 1; a <= 5; a++) {
        const inventory = inventories[role];
        const selected = inventory.slice(0, 3).sort(() => Math.random() - 0.5).slice(0, 2 + Math.floor(Math.random() * 2));

        const preferences: Record<string, number> = {};
        const scores: Record<string, number> = {};
        inventory.forEach(m => {
          preferences[m] = Math.random() * 2 - 0.5;
          scores[m] = Math.random();
        });

        agentDecisions.push({
          timestamp: new Date(Date.now() - (50 - i) * 4 * 3600000).toISOString(),
          iteration: i,
          agent_id: `${role.charAt(0).toUpperCase()}${a}`,
          role,
          methods_available: inventory,
          methods_selected: selected,
          selection_scores: scores,
          preferences,
          context: { trend, volatility, regime },
          reasoning: `Selected ${selected.join(', ')} based on ${trend} market conditions`,
          exploration_used: Math.random() > 0.7,
        });
      }
    }

    const pnl = (Math.random() - 0.4) * 0.05 + (i / 50) * 0.02;

    iterations.push({
      experiment_id: 'demo_exp',
      iteration: i,
      timestamp: new Date(Date.now() - (50 - i) * 4 * 3600000).toISOString(),
      market_regime: regime,
      market_context: { trend, volatility, regime },
      agent_decisions: agentDecisions,
      pipeline_results: [{
        pipeline_id: `pipe_${i}`,
        agents: { analyst: 'A1', researcher: 'R1', trader: 'T1', risk: 'M1' },
        methods: {
          analyst: ['RSI', 'HMM_Regime'],
          researcher: ['TemporalFusion', 'BootstrapEnsemble'],
          trader: ['KellyCriterion', 'MomentumEntry'],
          risk: ['MaxDrawdown', 'VaRLimit'],
        },
        pnl,
        sharpe: pnl / 0.02,
        success: pnl > 0,
      }],
      best_pipeline_id: `pipe_${i}`,
      best_pnl: pnl,
      avg_pnl: pnl * 0.8,
      knowledge_transfer: i % 10 === 0 ? {
        timestamp: new Date().toISOString(),
        role: 'analyst',
        source_agent_id: 'A1',
        target_agent_ids: ['A2', 'A3', 'A4', 'A5'],
        transfer_tau: 0.1,
        methods_transferred: ['RSI', 'HMM_Regime'],
        source_preferences: {},
      } : null,
      diversity_metrics: {
        analyst: { role: 'analyst', selection_diversity: 0.7 + Math.random() * 0.2, preference_entropy: 0.5, unique_methods_used: 8, total_methods_available: 10 },
        researcher: { role: 'researcher', selection_diversity: 0.6 + Math.random() * 0.3, preference_entropy: 0.6, unique_methods_used: 6, total_methods_available: 8 },
        trader: { role: 'trader', selection_diversity: 0.65 + Math.random() * 0.25, preference_entropy: 0.55, unique_methods_used: 6, total_methods_available: 8 },
        risk: { role: 'risk', selection_diversity: 0.75 + Math.random() * 0.15, preference_entropy: 0.45, unique_methods_used: 5, total_methods_available: 7 },
      },
      iteration_duration_ms: 100 + Math.random() * 200,
    });
  }

  return iterations;
}
