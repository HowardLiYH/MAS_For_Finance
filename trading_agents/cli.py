"""Command-line interface for the trading system."""
from __future__ import annotations
import argparse
from pathlib import Path

from . import run_single_iteration


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single iteration
  trading-agents run --symbol BTCUSD.PERP --interval 4h

  # Run with custom config
  trading-agents run --symbol BTCUSD.PERP --interval 4h --config configs/btc4h.yaml
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a single trading iteration")
    run_parser.add_argument("--symbol", default="BTCUSD.PERP", help="Trading symbol")
    run_parser.add_argument("--interval", default="4h", help="Time interval")
    run_parser.add_argument("--config", type=Path, help="Path to config file")

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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
