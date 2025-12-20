# PopAgent Dashboard

A modern React/Next.js visualization dashboard for the PopAgent multi-agent LLM trading system.

## Features

- **Population View**: Visualize all 20 agents (5 per role) with their current method selections
- **Method Heatmap**: See which methods are being selected most frequently by each agent
- **Learning Timeline**: Track PnL performance and knowledge transfer events over iterations
- **Pipeline Flow**: Animated visualization of data flowing through the agent pipeline
- **Agent Reasoning**: View LLM explanations for method selection decisions
- **Knowledge Transfer**: Visualize when and how knowledge is transferred between agents

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:3000
```

## API Integration

The dashboard can connect to the FastAPI backend for live data:

```bash
# Start the backend (from project root)
python -m trading_agents.cli api --port 8000

# Set API URL (optional, defaults to http://localhost:8000)
export NEXT_PUBLIC_API_URL=http://localhost:8000

# Start dashboard
npm run dev
```

## Components

| Component | Description |
|-----------|-------------|
| `AgentPopulation` | Grid view of agent populations by role |
| `MethodInventory` | Heatmap showing method selection patterns |
| `LearningTimeline` | Line chart of PnL over iterations |
| `PerformanceChart` | Cumulative returns and key metrics |
| `PipelineFlow` | Animated pipeline visualization |
| `ReflectionPanel` | Agent reasoning and decision explanations |
| `KnowledgeTransfer` | Transfer event timeline |

## Demo Mode

The dashboard includes mock data generation for demo purposes. Without a connected backend, it will automatically generate realistic sample data.

## Styling

Uses Tailwind CSS with a custom dark theme inspired by trading terminals:
- Dark background with cyan accent color
- Monospace fonts for data
- Subtle grid pattern background
- Glow effects for emphasis

## NeurIPS Presentation

This dashboard is designed for research presentation purposes, suitable for:
- Live demos during talks
- Figure generation for papers
- Interactive exploration of experimental results
