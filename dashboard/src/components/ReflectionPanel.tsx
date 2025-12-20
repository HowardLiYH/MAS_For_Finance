'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, ChevronRight, Lightbulb, TrendingUp, TrendingDown, AlertCircle, HelpCircle, Info } from 'lucide-react';
import type { AgentDecision, Role } from '@/lib/types';

interface ReflectionPanelProps {
  decisions: AgentDecision[];
  selectedAgentId?: string;
}

// Generate narrative reasoning based on context and selection
function generateNarrative(decision: AgentDecision): string {
  const role = decision.role;
  const trend = decision.context.trend || 'neutral';
  const vol = Math.round((decision.context.volatility || 0.02) * 100);
  const regime = decision.context.regime || 'normal';
  const methods = decision.methods_selected;

  const roleDescriptions: Record<string, string> = {
    analyst: 'feature engineering and trend detection',
    researcher: 'forecasting and market analysis',
    trader: 'position sizing and execution strategy',
    risk: 'risk assessment and portfolio protection',
  };

  const methodReasons: Record<string, string> = {
    // Analyst methods
    'RSI_MACD_Combo': 'identify momentum and trend reversals',
    'HMM_Regime': 'detect hidden market regime shifts',
    'Kalman_Trend': 'filter noise and extract true trend signals',
    'Wavelet_Decomp': 'analyze multi-scale price patterns',
    'VolatilityClustering': 'capture volatility regime changes',
    'TALib_Stack': 'compute comprehensive technical indicators',
    // Researcher methods
    'ARIMA_GARCH': 'forecast returns with volatility modeling',
    'XGBoost_Ensemble': 'combine multiple weak learners for robust predictions',
    'LSTM_Seq': 'capture long-term temporal dependencies',
    'BayesianUpdate': 'update beliefs with new market information',
    // Trader methods
    'MomentumStyle': 'capitalize on price trends',
    'MeanReversionStyle': 'profit from price corrections',
    'BreakoutStyle': 'catch major price breakouts',
    'VolumeProfile': 'identify key support/resistance levels',
    // Risk methods
    'VaR_CVaR': 'quantify potential downside risk',
    'DrawdownGuard': 'protect against excessive losses',
    'CorrelationCheck': 'ensure portfolio diversification',
    'LeverageLimit': 'prevent over-leveraging positions',
  };

  // Build narrative
  let narrative = `The ${role} agent observes a ${trend} market with ${vol}% volatility in a ${regime} regime. `;

  if (trend === 'bullish' && vol < 30) {
    narrative += `Given the favorable conditions, it focuses on ${roleDescriptions[role]}. `;
  } else if (trend === 'bearish') {
    narrative += `Due to the bearish sentiment, it adopts a defensive approach for ${roleDescriptions[role]}. `;
  } else if (vol > 40) {
    narrative += `The high volatility requires careful ${roleDescriptions[role]}. `;
  } else {
    narrative += `It proceeds with standard ${roleDescriptions[role]}. `;
  }

  // Explain method selection
  if (methods.length > 0) {
    const methodExplanations = methods
      .map(m => {
        const reason = methodReasons[m] || 'analyze market conditions';
        return `**${m}** (to ${reason})`;
      })
      .join(', ');
    narrative += `Selected methods: ${methodExplanations}.`;
  }

  return narrative;
}

// Tooltip component
function Tooltip({ text, children }: { text: string; children: React.ReactNode }) {
  const [show, setShow] = useState(false);
  return (
    <span className="relative inline-flex items-center" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-background border border-white/10 rounded text-xs text-white whitespace-nowrap z-50">
          {text}
        </span>
      )}
    </span>
  );
}

export default function ReflectionPanel({ decisions, selectedAgentId }: ReflectionPanelProps) {
  const [expandedAgent, setExpandedAgent] = useState<string | null>(selectedAgentId || null);
  const [showHelp, setShowHelp] = useState(false);

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
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-secondary" />
          <h2 className="text-lg font-semibold text-white">
            Agent Reasoning
          </h2>
        </div>
        <button
          onClick={() => setShowHelp(!showHelp)}
          className="flex items-center gap-1 text-xs text-muted hover:text-white transition-colors"
        >
          <HelpCircle className="w-4 h-4" />
          How to read this
        </button>
      </div>

      {/* Help Panel */}
      <AnimatePresence>
        {showHelp && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="bg-secondary/10 border border-secondary/20 rounded-lg p-4 space-y-2"
          >
            <h4 className="text-sm font-medium text-secondary flex items-center gap-2">
              <Info className="w-4 h-4" />
              Understanding Agent Reasoning
            </h4>
            <ul className="text-xs text-muted space-y-1.5">
              <li><strong className="text-white">Preference Score:</strong> Higher = agent prefers this method. Range 0-10. Updated after each iteration based on performance.</li>
              <li><strong className="text-white">Exploration Mode:</strong> Agent sometimes tries less-preferred methods to discover better strategies.</li>
              <li><strong className="text-white">Market Context:</strong> Trend (bullish/bearish/neutral), Volatility (%), and Regime (normal/volatile/quiet).</li>
              <li><strong className="text-white">Green badge:</strong> Method was selected in this iteration. Gray = available but not selected.</li>
            </ul>
          </motion.div>
        )}
      </AnimatePresence>

      <p className="text-xs text-muted">
        Click on an agent to see their decision rationale and method selection logic
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
                      {/* Generated Narrative Reasoning */}
                      <div className="bg-gradient-to-r from-secondary/10 to-primary/10 rounded-lg p-4 border border-white/5">
                        <div className="flex gap-2 mb-2">
                          <Lightbulb className="w-4 h-4 text-warning shrink-0" />
                          <span className="text-xs font-medium text-warning">Decision Narrative</span>
                        </div>
                        <p className="text-sm text-white/90 leading-relaxed">
                          {generateNarrative(decision).split('**').map((part, i) =>
                            i % 2 === 1 ? <strong key={i} className="text-primary">{part}</strong> : part
                          )}
                        </p>
                      </div>

                      {/* Context */}
                      <div className="bg-background/50 rounded-lg p-3">
                        <div className="text-xs text-muted mb-2 flex items-center gap-1">
                          Market Context
                          <Tooltip text="Real-time market conditions when this decision was made">
                            <HelpCircle className="w-3 h-3" />
                          </Tooltip>
                        </div>
                        <div className="flex gap-4 text-xs">
                          <Tooltip text="Price direction based on recent moving averages">
                            <span className="flex items-center gap-1 cursor-help">
                              {decision.context.trend === 'bullish' ? (
                                <TrendingUp className="w-3 h-3 text-success" />
                              ) : decision.context.trend === 'bearish' ? (
                                <TrendingDown className="w-3 h-3 text-danger" />
                              ) : (
                                <span className="w-3 h-3 bg-muted rounded-full" />
                              )}
                              {decision.context.trend}
                            </span>
                          </Tooltip>
                          <Tooltip text="Annualized volatility of returns. >40% is high, <20% is low">
                            <span className="cursor-help">
                              Vol: {(decision.context.volatility * 100).toFixed(0)}%
                            </span>
                          </Tooltip>
                          <Tooltip text="Market regime: normal, volatile (high uncertainty), or quiet (low activity)">
                            <span className="capitalize cursor-help">
                              {decision.context.regime}
                            </span>
                          </Tooltip>
                        </div>
                      </div>

                      {/* Top Preferences with explanations */}
                      <div>
                        <div className="text-xs text-muted mb-2 flex items-center gap-1">
                          Method Preferences
                          <Tooltip text="Scores learned over time. Higher = more successful historically. Range: 0-10">
                            <HelpCircle className="w-3 h-3" />
                          </Tooltip>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {topPrefs.map(([method, score]) => (
                            <Tooltip key={method} text={`Preference score: ${score.toFixed(2)}. ${decision.methods_selected.includes(method) ? 'SELECTED this iteration' : 'Available but not selected'}`}>
                              <span
                                className={`px-2 py-1 rounded text-xs font-mono cursor-help ${
                                  decision.methods_selected.includes(method)
                                    ? 'bg-success/20 text-success border border-success/30'
                                    : 'bg-white/5 text-muted'
                                }`}
                              >
                                {method}: {score.toFixed(2)}
                              </span>
                            </Tooltip>
                          ))}
                        </div>
                      </div>

                      {/* Exploration indicator */}
                      {decision.exploration_used && (
                        <div className="flex items-center gap-2 text-xs text-secondary bg-secondary/10 rounded-lg p-2">
                          <AlertCircle className="w-3 h-3" />
                          <span>
                            <strong>Exploration mode:</strong> Agent is trying less-preferred methods to discover potentially better strategies
                          </span>
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
