'use client';

import { motion } from 'framer-motion';
import { Database, Brain, TrendingUp, Shield, Award, ArrowRight } from 'lucide-react';
import type { PipelineResult } from '@/lib/types';

interface PipelineFlowProps {
  result: PipelineResult | null;
  isAnimating?: boolean;
}

export default function PipelineFlow({ result, isAnimating = false }: PipelineFlowProps) {
  const stages = [
    {
      id: 'input',
      label: 'Market Data',
      icon: Database,
      color: '#6b7280',
      methods: ['Price', 'Volume', 'News'],
    },
    {
      id: 'analyst',
      label: 'Analyst',
      icon: Brain,
      color: '#22d3ee',
      methods: result?.methods.analyst || ['—'],
    },
    {
      id: 'researcher',
      label: 'Researcher',
      icon: TrendingUp,
      color: '#a78bfa',
      methods: result?.methods.researcher || ['—'],
    },
    {
      id: 'trader',
      label: 'Trader',
      icon: TrendingUp,
      color: '#10b981',
      methods: result?.methods.trader || ['—'],
    },
    {
      id: 'risk',
      label: 'Risk',
      icon: Shield,
      color: '#f59e0b',
      methods: result?.methods.risk || ['—'],
    },
    {
      id: 'output',
      label: 'Decision',
      icon: Award,
      color: result?.success ? '#10b981' : '#ef4444',
      methods: [result ? `${(result.pnl * 100).toFixed(2)}%` : '—'],
    },
  ];

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">
        Pipeline Flow
      </h2>

      <div className="bg-surface/30 rounded-xl p-6 border border-white/5 overflow-x-auto">
        <div className="flex items-center justify-between min-w-[800px]">
          {stages.map((stage, idx) => (
            <div key={stage.id} className="flex items-center">
              {/* Stage node */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="relative"
              >
                {/* Icon circle */}
                <motion.div
                  className="w-16 h-16 rounded-xl flex items-center justify-center relative"
                  style={{
                    backgroundColor: `${stage.color}15`,
                    border: `2px solid ${stage.color}40`,
                  }}
                  animate={isAnimating ? {
                    boxShadow: [
                      `0 0 0 0 ${stage.color}00`,
                      `0 0 20px 5px ${stage.color}40`,
                      `0 0 0 0 ${stage.color}00`,
                    ],
                  } : {}}
                  transition={{
                    duration: 2,
                    repeat: isAnimating ? Infinity : 0,
                    delay: idx * 0.3,
                  }}
                >
                  <stage.icon
                    className="w-6 h-6"
                    style={{ color: stage.color }}
                  />

                  {/* Pulse animation */}
                  {isAnimating && (
                    <motion.div
                      className="absolute inset-0 rounded-xl"
                      style={{ border: `2px solid ${stage.color}` }}
                      initial={{ opacity: 0, scale: 1 }}
                      animate={{ opacity: [0.5, 0], scale: [1, 1.3] }}
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        delay: idx * 0.3,
                      }}
                    />
                  )}
                </motion.div>

                {/* Label */}
                <div className="text-center mt-2">
                  <div className="text-xs font-medium" style={{ color: stage.color }}>
                    {stage.label}
                  </div>
                </div>

                {/* Methods */}
                <div className="mt-2 flex flex-col items-center gap-1">
                  {stage.methods.slice(0, 2).map((method, mIdx) => (
                    <motion.span
                      key={mIdx}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: idx * 0.1 + mIdx * 0.05 }}
                      className="px-2 py-0.5 rounded text-[10px] font-mono bg-white/5 text-white/60"
                    >
                      {method}
                    </motion.span>
                  ))}
                  {stage.methods.length > 2 && (
                    <span className="text-[10px] text-muted">
                      +{stage.methods.length - 2} more
                    </span>
                  )}
                </div>
              </motion.div>

              {/* Arrow connector */}
              {idx < stages.length - 1 && (
                <motion.div
                  className="flex items-center mx-4"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: idx * 0.1 + 0.05 }}
                >
                  <motion.div
                    className="w-12 h-0.5 bg-gradient-to-r from-white/20 to-white/5"
                    animate={isAnimating ? {
                      background: [
                        'linear-gradient(to right, rgba(255,255,255,0.1), rgba(255,255,255,0.05))',
                        `linear-gradient(to right, ${stage.color}, rgba(255,255,255,0.05))`,
                        'linear-gradient(to right, rgba(255,255,255,0.1), rgba(255,255,255,0.05))',
                      ],
                    } : {}}
                    transition={{ duration: 1, repeat: isAnimating ? Infinity : 0, delay: idx * 0.3 }}
                  />
                  <ArrowRight className="w-4 h-4 text-white/20" />
                </motion.div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Result summary */}
      {result && (
        <div className={`flex items-center justify-between p-4 rounded-lg ${
          result.success ? 'bg-success/10 border border-success/20' : 'bg-danger/10 border border-danger/20'
        }`}>
          <div className="flex items-center gap-3">
            <Award className={`w-5 h-5 ${result.success ? 'text-success' : 'text-danger'}`} />
            <span className="text-sm">
              Pipeline <span className="font-mono">{result.pipeline_id}</span>
            </span>
          </div>
          <div className="flex items-center gap-6 text-sm">
            <span>
              PnL: <span className={`font-mono ${result.success ? 'text-success' : 'text-danger'}`}>
                {result.pnl >= 0 ? '+' : ''}{(result.pnl * 100).toFixed(3)}%
              </span>
            </span>
            <span>
              Sharpe: <span className="font-mono text-white">{result.sharpe.toFixed(2)}</span>
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
