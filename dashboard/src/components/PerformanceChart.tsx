'use client';

import { useMemo, useState } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { TrendingUp, TrendingDown, Activity, Target, HelpCircle, Calendar } from 'lucide-react';
import type { IterationLog } from '@/lib/types';

interface StatCardProps {
  label: string;
  value: string;
  change?: number;
  icon: React.ReactNode;
  tooltip?: string;
}

// Tooltip component
function MetricTooltip({ text, children }: { text: string; children: React.ReactNode }) {
  const [show, setShow] = useState(false);
  return (
    <span className="relative inline-flex items-center" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-background border border-white/10 rounded text-xs text-white whitespace-nowrap z-50 max-w-[200px] text-center">
          {text}
        </span>
      )}
    </span>
  );
}

function StatCard({ label, value, change, icon, tooltip }: StatCardProps) {
  return (
    <div className="bg-surface/50 rounded-lg p-4 border border-white/5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-muted flex items-center gap-1">
          {label}
          {tooltip && (
            <MetricTooltip text={tooltip}>
              <HelpCircle className="w-3 h-3 cursor-help" />
            </MetricTooltip>
          )}
        </span>
        <span className="text-muted/50">{icon}</span>
      </div>
      <div className="flex items-end gap-2">
        <span className="text-2xl font-semibold text-white">{value}</span>
        {change !== undefined && (
          <span className={`text-xs flex items-center ${change >= 0 ? 'text-success' : 'text-danger'}`}>
            {change >= 0 ? <TrendingUp className="w-3 h-3 mr-0.5" /> : <TrendingDown className="w-3 h-3 mr-0.5" />}
            {Math.abs(change).toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  );
}

interface PerformanceChartProps {
  iterations: IterationLog[];
}

export default function PerformanceChart({ iterations }: PerformanceChartProps) {
  const { chartData, stats } = useMemo(() => {
    let cumulativePnl = 0;

    const data = iterations.map((iter) => {
      cumulativePnl += iter.best_pnl;
      return {
        iteration: iter.iteration,
        pnl: iter.best_pnl * 100,
        cumulative: cumulativePnl * 100,
        sharpe: iter.pipeline_results[0]?.sharpe || 0,
      };
    });

    // Calculate stats
    const totalPnl = cumulativePnl * 100;
    const avgPnl = iterations.reduce((a, b) => a + b.best_pnl, 0) / iterations.length * 100;
    const winRate = iterations.filter(i => i.best_pnl > 0).length / iterations.length * 100;
    const maxDrawdown = calculateMaxDrawdown(data.map(d => d.cumulative));

    // Calculate improvement
    const first10Avg = iterations.slice(0, 10).reduce((a, b) => a + b.best_pnl, 0) / 10;
    const last10Avg = iterations.slice(-10).reduce((a, b) => a + b.best_pnl, 0) / 10;
    const improvement = ((last10Avg - first10Avg) / Math.abs(first10Avg || 0.01)) * 100;

    return {
      chartData: data,
      stats: { totalPnl, avgPnl, winRate, maxDrawdown, improvement },
    };
  }, [iterations]);

  // Get date range from iterations
  const dateRange = useMemo(() => {
    if (iterations.length === 0) return null;
    const first = iterations[0]?.timestamp;
    const last = iterations[iterations.length - 1]?.timestamp;
    return { start: first, end: last };
  }, [iterations]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">
          Performance Metrics
        </h2>
        {dateRange && (
          <div className="flex items-center gap-2 text-xs text-muted bg-surface/30 px-3 py-1.5 rounded-lg">
            <Calendar className="w-3 h-3" />
            <span>{dateRange.start} → {dateRange.end}</span>
            <span className="text-white/50">|</span>
            <span>{iterations.length * 4}h of data</span>
          </div>
        )}
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="Total Return"
          value={`${stats.totalPnl >= 0 ? '+' : ''}${stats.totalPnl.toFixed(2)}%`}
          icon={<TrendingUp className="w-4 h-4" />}
          tooltip="Cumulative profit/loss from all trades. Positive = profitable."
        />
        <StatCard
          label="Win Rate"
          value={`${stats.winRate.toFixed(0)}%`}
          icon={<Target className="w-4 h-4" />}
          tooltip="Percentage of iterations with positive PnL. >50% is good."
        />
        <StatCard
          label="Max Drawdown"
          value={`-${stats.maxDrawdown.toFixed(2)}%`}
          icon={<TrendingDown className="w-4 h-4" />}
          tooltip="Largest peak-to-trough decline. Lower is better. <10% is excellent."
        />
        <StatCard
          label="Learning Improvement"
          value={`${stats.improvement >= 0 ? '+' : ''}${stats.improvement.toFixed(0)}%`}
          change={stats.improvement}
          icon={<Activity className="w-4 h-4" />}
          tooltip="Compares first 10 vs last 10 iterations. Positive = agents are learning."
        />
      </div>

      {/* Cumulative PnL chart */}
      <div className="bg-surface/30 rounded-xl p-4 border border-white/5">
        <h3 className="text-sm font-medium text-muted mb-4">Cumulative Returns</h3>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="cumulativeGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
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
              formatter={(value: number) => [`${value.toFixed(3)}%`, 'Cumulative']}
            />

            <Area
              type="monotone"
              dataKey="cumulative"
              stroke="#10b981"
              strokeWidth={2}
              fill="url(#cumulativeGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function calculateMaxDrawdown(cumulativeReturns: number[]): number {
  let maxDrawdown = 0;
  let peak = cumulativeReturns[0];

  for (const value of cumulativeReturns) {
    if (value > peak) {
      peak = value;
    }
    const drawdown = peak - value;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }

  return maxDrawdown;
}
