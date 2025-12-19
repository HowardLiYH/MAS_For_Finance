"""Command-line interface for the trading system."""
from __future__ import annotations
import argparse
from pathlib import Path

from .workflow import run_single_iteration, run_multi_asset


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single iteration (single asset)
  trading-agents run --symbol BTCUSD.PERP --interval 4h

  # Run with custom config
  trading-agents run --symbol BTCUSD.PERP --interval 4h --config configs/btc4h.yaml

  # Run multi-asset mode (5 coins with cross-asset features)
  trading-agents multi --config configs/multi_asset.yaml

  # Run multi-asset with specific symbols
  trading-agents multi --symbols BTC ETH SOL
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command (single asset)
    run_parser = subparsers.add_parser("run", help="Run a single trading iteration")
    run_parser.add_argument("--symbol", default="BTCUSD.PERP", help="Trading symbol")
    run_parser.add_argument("--interval", default="4h", help="Time interval")
    run_parser.add_argument("--config", type=Path, help="Path to config file")

    # Multi-asset command
    multi_parser = subparsers.add_parser("multi", help="Run multi-asset trading iteration")
    multi_parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file (default: configs/multi_asset.yaml)"
    )
    multi_parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override symbols to trade (e.g., --symbols BTC ETH SOL)"
    )

    args = parser.parse_args()

    if args.command == "run":
        result = run_single_iteration(
            symbol=args.symbol,
            interval=args.interval,
            config_path=args.config,
        )
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Iteration: {result['iteration']}")
        print(f"Features shape: {result['features_shape']}")
        print(f"Trend shape: {result['trend_shape']}")
        print(f"Execution: {result['execution']}")
        print(f"Risk Review: {result['risk_review']}")
        print(f"Scores: {result['scores']}")

    elif args.command == "multi":
        result = run_multi_asset(
            config_path=args.config,
            symbols=args.symbols,
        )
        print(f"\n{'='*60}")
        print("MULTI-ASSET RESULTS")
        print(f"{'='*60}")
        print(f"Iteration: {result['iteration']}")
        print(f"Mode: {result['mode']}")
        print(f"Symbols: {result['symbols']}")
        print(f"Market context available: {result['market_context_available']}")
        print()

        for symbol, asset_result in result.get("results", {}).items():
            print(f"--- {symbol} ---")
            print(f"  Features shape: {asset_result['features_shape']}")
            print(f"  Execution: {asset_result['execution'].get('side', 'N/A')} "
                  f"{asset_result['execution'].get('position_size', 'N/A')}")
            print(f"  Risk: {asset_result['risk_review'].get('verdict', 'N/A')}")
            print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
