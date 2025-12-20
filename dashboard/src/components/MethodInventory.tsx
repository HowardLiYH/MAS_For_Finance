'use client';

import { useMemo } from 'react';
import { motion } from 'framer-motion';
import type { AgentDecision, Role } from '@/lib/types';

interface MethodHeatmapProps {
  decisions: AgentDecision[];
  role: Role;
}

function MethodHeatmap({ decisions, role }: MethodHeatmapProps) {
  const roleColors: Record<Role, string> = {
    analyst: '#22d3ee',
    researcher: '#a78bfa',
    trader: '#10b981',
    risk: '#f59e0b',
  };

  const color = roleColors[role];

  // Get all unique methods and agents
  const { methods, agents, selectionMatrix } = useMemo(() => {
    const roleDecisions = decisions.filter(d => d.role === role);
    const allMethods = new Set<string>();
    const allAgents = new Set<string>();

    roleDecisions.forEach(d => {
      allAgents.add(d.agent_id);
      d.methods_available.forEach(m => allMethods.add(m));
    });

    const methods = Array.from(allMethods).sort();
    const agents = Array.from(allAgents).sort();

    // Build selection matrix
    const matrix: Record<string, Record<string, number>> = {};
    agents.forEach(agent => {
      matrix[agent] = {};
      methods.forEach(method => {
        matrix[agent][method] = 0;
      });
    });

    roleDecisions.forEach(d => {
      d.methods_selected.forEach(m => {
        if (matrix[d.agent_id]) {
          matrix[d.agent_id][m] = (matrix[d.agent_id][m] || 0) + 1;
        }
      });
    });

    return { methods, agents, selectionMatrix: matrix };
  }, [decisions, role]);

  // Get intensity for cell
  const getIntensity = (agent: string, method: string): number => {
    const count = selectionMatrix[agent]?.[method] || 0;
    return Math.min(count / 5, 1); // Normalize to 0-1
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr>
            <th className="p-1 text-left text-muted font-normal">Agent</th>
            {methods.map(method => (
              <th
                key={method}
                className="p-1 text-center font-normal text-muted"
                style={{ writingMode: 'vertical-rl', textOrientation: 'mixed', height: '80px' }}
              >
                {method}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {agents.map(agent => (
            <tr key={agent}>
              <td className="p-1 font-mono" style={{ color }}>{agent}</td>
              {methods.map(method => {
                const intensity = getIntensity(agent, method);
                return (
                  <td key={method} className="p-0.5">
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="heatmap-cell mx-auto"
                      style={{
                        backgroundColor: intensity > 0
                          ? `${color}${Math.round(intensity * 200 + 55).toString(16).padStart(2, '0')}`
                          : 'rgba(255,255,255,0.03)',
                        boxShadow: intensity > 0.5 ? `0 0 10px ${color}40` : 'none',
                      }}
                      title={`${agent}: ${method} (${Math.round(intensity * 100)}%)`}
                    />
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface MethodInventoryProps {
  decisions: AgentDecision[];
}

export default function MethodInventory({ decisions }: MethodInventoryProps) {
  const roles: Role[] = ['analyst', 'researcher', 'trader', 'risk'];

  const roleLabels: Record<Role, string> = {
    analyst: 'Analyst Methods',
    researcher: 'Researcher Methods',
    trader: 'Trader Methods',
    risk: 'Risk Methods',
  };

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">
        Method Selection Heatmap
      </h2>
      <p className="text-xs text-muted">
        Brighter cells indicate methods selected more frequently by each agent
      </p>

      <div className="grid grid-cols-2 gap-4">
        {roles.map(role => (
          <div
            key={role}
            className="bg-surface/30 rounded-xl p-4 border border-white/5"
          >
            <h3 className="text-sm font-medium text-white/80 mb-3">
              {roleLabels[role]}
            </h3>
            <MethodHeatmap decisions={decisions} role={role} />
          </div>
        ))}
      </div>
    </div>
  );
}
