'use client';

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from 'recharts';
import { motion } from 'framer-motion';
import { ArrowUpRight, ArrowDownRight, Zap } from 'lucide-react';
import type { IterationLog } from '@/lib/types';

interface LearningTimelineProps {
  iterations: IterationLog[];
  currentIteration?: number;
  onSelectIteration?: (iteration: number) => void;
}

export default function LearningTimeline({
  iterations,
  currentIteration,
  onSelectIteration,
}: LearningTimelineProps) {
  const chartData = useMemo(() => {
    return iterations.map((iter, idx) => ({
      iteration: iter.iteration,
      bestPnl: iter.best_pnl * 100, // Convert to percentage
      avgPnl: iter.avg_pnl * 100,
      transfer: iter.knowledge_transfer !== null,
      regime: iter.market_regime,
      timestamp: new Date(iter.timestamp).toLocaleDateString(),
    }));
  }, [iterations]);

  // Calculate statistics
  const stats = useMemo(() => {
    if (iterations.length < 10) return null;

    const first10 = iterations.slice(0, 10);
    const last10 = iterations.slice(-10);

    const avgFirst = first10.reduce((a, b) => a + b.best_pnl, 0) / 10;
    const avgLast = last10.reduce((a, b) => a + b.best_pnl, 0) / 10;
    const improvement = avgLast - avgFirst;

    const transferCount = iterations.filter(i => i.knowledge_transfer).length;
    const winRate = iterations.filter(i => i.best_pnl > 0).length / iterations.length;

    return { avgFirst, avgLast, improvement, transferCount, winRate };
  }, [iterations]);

  // Transfer markers for the chart
  const transferIterations = iterations
    .filter(i => i.knowledge_transfer)
    .map(i => i.iteration);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">
          Learning Progress
        </h2>

        {stats && (
          <div className="flex items-center gap-6 text-xs">
            <div className="flex items-center gap-2">
              {stats.improvement > 0 ? (
                <ArrowUpRight className="w-4 h-4 text-success" />
              ) : (
                <ArrowDownRight className="w-4 h-4 text-danger" />
              )}
              <span className={stats.improvement > 0 ? 'text-success' : 'text-danger'}>
                {(stats.improvement * 100).toFixed(2)}% improvement
              </span>
            </div>
            <div className="flex items-center gap-2 text-muted">
              <Zap className="w-4 h-4 text-secondary" />
              <span>{stats.transferCount} transfers</span>
            </div>
            <div className="text-muted">
              Win Rate: <span className="text-white">{(stats.winRate * 100).toFixed(0)}%</span>
            </div>
          </div>
        )}
      </div>

      {/* Main Chart */}
      <div className="bg-surface/30 rounded-xl p-4 border border-white/5">
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={chartData}>
            <defs>
              <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#22d3ee" stopOpacity={0}/>
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.05)"
              vertical={false}
            />

            <XAxis
              dataKey="iteration"
              stroke="#6b7280"
              fontSize={10}
              tickLine={false}
            />

            <YAxis
              stroke="#6b7280"
              fontSize={10}
              tickLine={false}
              tickFormatter={(v) => `${v.toFixed(1)}%`}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                fontSize: '12px',
              }}
              labelStyle={{ color: '#9ca3af' }}
              formatter={(value: number, name: string) => [
                `${value.toFixed(3)}%`,
                name === 'bestPnl' ? 'Best PnL' : 'Avg PnL'
              ]}
            />

            {/* Zero line */}
            <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="3 3" />

            {/* Transfer markers */}
            {transferIterations.map(iter => (
              <ReferenceLine
                key={iter}
                x={iter}
                stroke="#a78bfa"
                strokeDasharray="3 3"
                strokeOpacity={0.5}
              />
            ))}

            {/* Area under best PnL */}
            <Area
              type="monotone"
              dataKey="bestPnl"
              fill="url(#pnlGradient)"
              stroke="none"
            />

            {/* Best PnL line */}
            <Line
              type="monotone"
              dataKey="bestPnl"
              stroke="#22d3ee"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#22d3ee' }}
            />

            {/* Average PnL line */}
            <Line
              type="monotone"
              dataKey="avgPnl"
              stroke="#6b7280"
              strokeWidth={1}
              strokeDasharray="5 5"
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-4 text-xs text-muted">
          <span className="flex items-center gap-2">
            <span className="w-4 h-0.5 bg-primary" />
            Best PnL
          </span>
          <span className="flex items-center gap-2">
            <span className="w-4 h-0.5 bg-muted border-dashed" style={{ borderTop: '1px dashed' }} />
            Avg PnL
          </span>
          <span className="flex items-center gap-2">
            <span className="w-0.5 h-4 bg-secondary opacity-50" />
            Knowledge Transfer
          </span>
        </div>
      </div>

      {/* Iteration selector */}
      {onSelectIteration && (
        <div className="flex gap-1 overflow-x-auto pb-2">
          {iterations.map((iter) => (
            <motion.button
              key={iter.iteration}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onSelectIteration(iter.iteration)}
              className={`
                w-6 h-6 rounded text-[10px] font-mono transition-colors
                ${currentIteration === iter.iteration
                  ? 'bg-primary text-background'
                  : iter.best_pnl > 0
                    ? 'bg-success/20 text-success'
                    : 'bg-danger/20 text-danger'
                }
                ${iter.knowledge_transfer ? 'ring-1 ring-secondary' : ''}
              `}
            >
              {iter.iteration}
            </motion.button>
          ))}
        </div>
      )}
    </div>
  );
}
