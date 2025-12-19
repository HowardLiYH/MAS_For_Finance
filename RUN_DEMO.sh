#!/usr/bin/env bash
# Multi-Agent Trading System Demo Scripts

echo "=== Multi-Agent Trading System ==="
echo ""

# Check if data exists
if [ ! -d "data/bybit" ] || [ -z "$(ls -A data/bybit 2>/dev/null)" ]; then
    echo "⚠️  No Bybit data found in data/bybit/"
    echo "   Copy your CSV files: cp /path/to/Bybit_CSV_Data/*.csv data/bybit/"
    echo ""
fi

echo "Available commands:"
echo ""
echo "  # Multi-asset mode (5 coins with cross-asset features)"
echo "  python -m trading_agents.cli multi --config configs/multi_asset.yaml"
echo ""
echo "  # Single-asset mode (individual coins)"
echo "  python -m trading_agents.cli run --config configs/single/btc.yaml"
echo "  python -m trading_agents.cli run --config configs/single/eth.yaml"
echo "  python -m trading_agents.cli run --config configs/single/sol.yaml"
echo "  python -m trading_agents.cli run --config configs/single/doge.yaml"
echo "  python -m trading_agents.cli run --config configs/single/xrp.yaml"
echo ""

# Default: run multi-asset
echo "Running multi-asset demo..."
python -m trading_agents.cli multi --config configs/multi_asset.yaml
