'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  Layers,
  Brain,
  BarChart3,
  Zap,
  PlayCircle,
  PauseCircle,
  SkipForward,
  SkipBack,
  RefreshCw,
} from 'lucide-react';
import AgentPopulation from '@/components/AgentPopulation';
import MethodInventory from '@/components/MethodInventory';
import LearningTimeline from '@/components/LearningTimeline';
import PerformanceChart from '@/components/PerformanceChart';
import ReflectionPanel from '@/components/ReflectionPanel';
import PipelineFlow from '@/components/PipelineFlow';
import KnowledgeTransfer from '@/components/KnowledgeTransfer';
import { loadDemoData } from '@/lib/api';
import type { IterationLog } from '@/lib/types';

type TabId = 'overview' | 'population' | 'methods' | 'pipeline' | 'reasoning';

export default function Dashboard() {
  const [iterations, setIterations] = useState<IterationLog[]>([]);
  const [currentIteration, setCurrentIteration] = useState(1);
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Load demo data
  useEffect(() => {
    loadDemoData().then((data) => {
      setIterations(data);
      setCurrentIteration(data.length);
      setIsLoading(false);
    });
  }, []);

  // Auto-play through iterations
  useEffect(() => {
    if (!isPlaying || iterations.length === 0) return;

    const interval = setInterval(() => {
      setCurrentIteration((prev) => {
        if (prev >= iterations.length) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 500);

    return () => clearInterval(interval);
  }, [isPlaying, iterations.length]);

  // Get current iteration data
  const currentData = iterations.find(i => i.iteration === currentIteration);
  const transfers = iterations
    .filter(i => i.knowledge_transfer && i.iteration <= currentIteration)
    .map(i => i.knowledge_transfer!);

  const tabs = [
    { id: 'overview' as const, label: 'Overview', icon: Activity },
    { id: 'population' as const, label: 'Population', icon: Layers },
    { id: 'methods' as const, label: 'Methods', icon: BarChart3 },
    { id: 'pipeline' as const, label: 'Pipeline', icon: Zap },
    { id: 'reasoning' as const, label: 'Reasoning', icon: Brain },
  ];

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        >
          <RefreshCw className="w-8 h-8 text-primary" />
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <span className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </span>
              PopAgent Dashboard
            </h1>
            <p className="text-muted mt-1">
              Multi-Agent LLM Trading with Adaptive Method Selection
            </p>
          </div>

          {/* Playback controls */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 bg-surface/50 rounded-lg p-2">
              <button
                onClick={() => setCurrentIteration(Math.max(1, currentIteration - 1))}
                className="p-2 hover:bg-white/10 rounded transition-colors"
                disabled={currentIteration <= 1}
              >
                <SkipBack className="w-4 h-4" />
              </button>

              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="p-2 hover:bg-white/10 rounded transition-colors"
              >
                {isPlaying ? (
                  <PauseCircle className="w-5 h-5 text-primary" />
                ) : (
                  <PlayCircle className="w-5 h-5 text-primary" />
                )}
              </button>

              <button
                onClick={() => setCurrentIteration(Math.min(iterations.length, currentIteration + 1))}
                className="p-2 hover:bg-white/10 rounded transition-colors"
                disabled={currentIteration >= iterations.length}
              >
                <SkipForward className="w-4 h-4" />
              </button>
            </div>

            <div className="text-right">
              <div className="text-2xl font-mono font-bold text-primary">
                {currentIteration}
                <span className="text-muted text-sm font-normal">/{iterations.length}</span>
              </div>
              <div className="text-xs text-muted">Iteration</div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mt-6">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all
                ${activeTab === tab.id
                  ? 'bg-primary/20 text-primary border border-primary/30'
                  : 'bg-surface/30 text-muted hover:text-white hover:bg-surface/50'
                }
              `}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>
      </header>

      {/* Main Content */}
      <main className="space-y-8">
        {activeTab === 'overview' && currentData && (
          <>
            {/* Performance Charts */}
            <PerformanceChart iterations={iterations.slice(0, currentIteration)} />

            {/* Learning Timeline */}
            <LearningTimeline
              iterations={iterations.slice(0, currentIteration)}
              currentIteration={currentIteration}
              onSelectIteration={setCurrentIteration}
            />

            {/* Knowledge Transfers */}
            <KnowledgeTransfer transfers={transfers} />
          </>
        )}

        {activeTab === 'population' && currentData && (
          <AgentPopulation
            decisions={currentData.agent_decisions}
            bestAgentIds={{
              analyst: currentData.pipeline_results[0]?.agents.analyst,
              researcher: currentData.pipeline_results[0]?.agents.researcher,
              trader: currentData.pipeline_results[0]?.agents.trader,
              risk: currentData.pipeline_results[0]?.agents.risk,
            }}
          />
        )}

        {activeTab === 'methods' && currentData && (
          <MethodInventory
            decisions={iterations
              .slice(Math.max(0, currentIteration - 10), currentIteration)
              .flatMap(i => i.agent_decisions)
            }
          />
        )}

        {activeTab === 'pipeline' && currentData && (
          <PipelineFlow
            result={currentData.pipeline_results[0] || null}
            isAnimating={isPlaying}
          />
        )}

        {activeTab === 'reasoning' && currentData && (
          <ReflectionPanel decisions={currentData.agent_decisions} />
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 pt-6 border-t border-white/5 text-center text-xs text-muted">
        <p>PopAgent: Population-Based Multi-Agent LLM Trading System</p>
        <p className="mt-1">NeurIPS 2025 Submission</p>
      </footer>
    </div>
  );
}
