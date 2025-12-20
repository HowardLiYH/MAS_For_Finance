"""Command-line interface for the trading system."""
from __future__ import annotations
import argparse
from pathlib import Path

from .workflow import (
    run_single_iteration,
    run_multi_asset,
    WorkflowEngine,
)


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
  trading-agents run --symbol BTCUSD.PERP --interval 4h --config configs/single/btc.yaml

  # Run multi-asset mode (5 coins with cross-asset features)
  trading-agents multi --config configs/multi_asset.yaml

  # Run multi-asset with specific symbols
  trading-agents multi --symbols BTC ETH SOL

  # Run paper trading (requires Bybit Testnet API keys)
  trading-agents paper --symbols BTC ETH

  # Send performance report
  trading-agents report --days 30
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

    # Paper trading command
    paper_parser = subparsers.add_parser("paper", help="Run paper trading with Bybit Testnet")
    paper_parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file (default: configs/multi_asset.yaml)"
    )
    paper_parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override symbols to trade"
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate and send performance report")
    report_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Lookback days for report (default: 30)"
    )
    report_parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file"
    )

    # Admin status command
    status_parser = subparsers.add_parser("status", help="Show admin agent status")
    status_parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file"
    )

    # Selector (PopAgent) command
    selector_parser = subparsers.add_parser(
        "selector",
        help="Run PopAgent with adaptive method selection"
    )
    selector_parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file (default: configs/multi_asset.yaml)"
    )
    selector_parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to run (default: 10)"
    )
    selector_parser.add_argument(
        "--population-size",
        type=int,
        default=5,
        help="Agents per role (default: 5)"
    )
    selector_parser.add_argument(
        "--max-methods",
        type=int,
        default=3,
        help="Methods each agent selects (default: 3)"
    )

    # Population (legacy) command
    population_parser = subparsers.add_parser(
        "population",
        help="Run legacy fixed-variant population (use 'selector' for new approach)"
    )
    population_parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file"
    )
    population_parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to run (default: 10)"
    )

    # Backtest command (population-based historical backtest)
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run population-based historical backtest with real price data"
    )
    backtest_parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file (default: configs/multi_asset.yaml)"
    )
    backtest_parser.add_argument(
        "--symbol",
        type=str,
        default="BTC",
        help="Symbol to backtest (default: BTC)"
    )
    backtest_parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--population-size",
        type=int,
        default=5,
        help="Agents per role (default: 5)"
    )
    backtest_parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)"
    )
    backtest_parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/experiments",
        help="Directory for experiment logs"
    )

    # Export command for NeurIPS
    export_parser = subparsers.add_parser(
        "export",
        help="Export experiment data for NeurIPS paper (figures, tables, traces)"
    )
    export_parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment ID to export"
    )
    export_parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/experiments",
        help="Directory containing experiment logs"
    )
    export_parser.add_argument(
        "--output-dir",
        type=str,
        default="exports/neurips",
        help="Output directory for figures and tables"
    )
    export_parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["pdf", "latex"],
        help="Output formats: pdf, png, latex, csv"
    )

    # API server command
    api_parser = subparsers.add_parser(
        "api",
        help="Start the dashboard API server"
    )
    api_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)"
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)"
    )
    api_parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/experiments",
        help="Directory containing experiment logs"
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
        _print_multi_results(result)

    elif args.command == "paper":
        try:
            from .workflow import run_paper_trading
            result = run_paper_trading(
                config_path=args.config,
                symbols=args.symbols,
            )
            print("\n[PAPER TRADING] Enabled - Orders submitted to Bybit Testnet")
            _print_multi_results(result)
        except ImportError as e:
            print(f"Error: {e}")
            print("Install paper trading dependencies: pip install trading-agents[paper-trading]")
        except ValueError as e:
            print(f"Error: {e}")
            print("Set BYBIT_TESTNET_KEY and BYBIT_TESTNET_SECRET environment variables")

    elif args.command == "report":
        engine = WorkflowEngine(config_path=args.config)
        results = engine.admin.send_performance_report(args.days)
        print(f"\n{'='*60}")
        print(f"PERFORMANCE REPORT ({args.days} days)")
        print(f"{'='*60}")
        for channel, success in results.items():
            status = "✓ Sent" if success else "✗ Failed"
            print(f"  {channel}: {status}")

    elif args.command == "status":
        engine = WorkflowEngine(config_path=args.config)
        status = engine.admin.get_status()
        print(f"\n{'='*60}")
        print("ADMIN AGENT STATUS")
        print(f"{'='*60}")
        for key, value in status.items():
            print(f"  {key}: {value}")

    elif args.command == "selector":
        _run_selector_mode(args)

    elif args.command == "population":
        _run_population_mode(args)

    elif args.command == "backtest":
        _run_backtest_mode(args)

    elif args.command == "export":
        _run_export_mode(args)

    elif args.command == "api":
        _run_api_server(args)

    else:
        parser.print_help()


def _print_multi_results(result: dict):
    """Print multi-asset results."""
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


def _run_selector_mode(args):
    """Run PopAgent with adaptive method selection."""
    import pandas as pd
    from .population import (
        SelectorWorkflow,
        SelectorWorkflowConfig,
        get_inventory_sizes,
    )

    print(f"\n{'='*60}")
    print("🧬 POPAGENT: ADAPTIVE METHOD SELECTION")
    print(f"{'='*60}")

    # Show inventory sizes
    inv_sizes = get_inventory_sizes()
    print("\nMethod Inventories:")
    for role, size in inv_sizes.items():
        print(f"  {role.capitalize()}: {size} methods (agents select {args.max_methods})")

    # Create workflow
    config = SelectorWorkflowConfig(
        population_size=args.population_size,
        max_methods_per_agent=args.max_methods,
    )
    workflow = SelectorWorkflow(config)

    print(f"\nPopulation: {args.population_size} agents per role")
    print(f"Running {args.iterations} iterations...\n")

    # Run iterations
    for i in range(args.iterations):
        # Create dummy price data for demo
        price_data = pd.DataFrame({
            "open": [100 + i * 0.1],
            "high": [101 + i * 0.1],
            "low": [99 + i * 0.1],
            "close": [100.5 + i * 0.1],
            "volume": [1000000],
        })

        # Simulate market context
        context = {
            "trend": "bullish" if i % 3 != 0 else "bearish",
            "volatility": 0.3 + (i % 5) * 0.1,
            "regime": "normal" if i % 4 != 0 else "volatile",
        }

        summary = workflow.run_iteration(price_data, context)

        print(f"Iteration {summary.iteration}: "
              f"Best PnL={summary.best_pnl:.4f}, "
              f"Avg PnL={summary.avg_pnl:.4f}, "
              f"Transfer={'✓' if summary.transfer_performed else '-'}")

    # Print final summary
    print(f"\n{'='*60}")
    print("LEARNING RESULTS")
    print(f"{'='*60}")

    progress = workflow.get_learning_progress()
    print(f"\nImprovement: {progress['improvement']:.4f}")
    print(f"Best methods by role:")
    for role, methods in progress['best_methods'].items():
        print(f"  {role}: {methods}")

    print(f"\nMethod popularity:")
    for role, popularity in progress['method_popularity'].items():
        top_methods = sorted(popularity.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  {role}: {[(m, f'{p:.1%}') for m, p in top_methods]}")


def _run_population_mode(args):
    """Run legacy fixed-variant population mode."""
    print(f"\n{'='*60}")
    print("POPULATION MODE (Legacy Fixed Variants)")
    print(f"{'='*60}")
    print("\n⚠️  This is the legacy approach with fixed agent variants.")
    print("   Use 'selector' command for the new adaptive method selection.\n")

    from .population import PopulationWorkflow

    workflow = PopulationWorkflow()

    for i in range(args.iterations):
        result = workflow.run_iteration(
            price_data=None,  # Would need real data
            market_context={},
        )
        print(f"Iteration {i+1}: {result}")

    print("\n[Population Mode Complete]")


def _run_backtest_mode(args):
    """Run population-based historical backtest."""
    from datetime import datetime, timezone
    from pathlib import Path
    import pandas as pd

    from .population import (
        SelectorWorkflow,
        SelectorWorkflowConfig,
        get_inventory_sizes,
    )
    from .backtesting import BacktestEngine
    from .services.experiment_logger import ExperimentLogger

    print(f"\n{'='*60}")
    print("📊 POPULATION BACKTEST")
    print(f"{'='*60}")

    # Show inventory sizes
    inv_sizes = get_inventory_sizes()
    print("\nMethod Inventories:")
    for role, size in inv_sizes.items():
        print(f"  {role.capitalize()}: {size} methods")

    # Load price data
    symbol = args.symbol.upper()
    csv_path = Path(f"data/bybit/Bybit_{symbol}.csv")

    if not csv_path.exists():
        print(f"\n❌ Error: Price data not found at {csv_path}")
        print(f"   Available symbols: BTC, ETH, SOL, DOGE, XRP")
        return

    print(f"\nLoading {symbol} price data from {csv_path}...")
    price_df = pd.read_csv(csv_path)

    # Handle timestamp column
    if "timestamp_utc" in price_df.columns:
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp_utc"], utc=True)
    elif "timestamp" in price_df.columns:
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)

    price_df.set_index("timestamp", inplace=True)
    price_df.sort_index(inplace=True)

    print(f"Loaded {len(price_df)} bars ({price_df.index[0]} to {price_df.index[-1]})")

    # Parse dates
    start_date = None
    end_date = None

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Create workflow
    config = SelectorWorkflowConfig(
        population_size=args.population_size,
        max_methods_per_agent=3,
    )
    workflow = SelectorWorkflow(config)

    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=args.capital,
        commission=0.001,
    )

    # Create logger
    logger = ExperimentLogger(
        experiment_id=f"backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        log_dir=args.log_dir,
        config={
            "symbol": symbol,
            "population_size": args.population_size,
            "initial_capital": args.capital,
            "start_date": args.start,
            "end_date": args.end,
        }
    )

    print(f"\nPopulation: {args.population_size} agents per role")
    print(f"Initial capital: ${args.capital:,.2f}")
    print(f"Logging to: {args.log_dir}")
    print("\nStarting backtest...\n")

    # Run population backtest
    results = engine.run_population_backtest(
        price_df=price_df,
        selector_workflow=workflow,
        start_date=start_date,
        end_date=end_date,
        iteration_bar_count=1,  # Every 4h bar
        logger=logger,
    )

    # Print results
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")

    metrics = results["backtest_metrics"]
    print(f"\n📈 Performance Metrics:")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Final Capital: ${metrics['final_capital']:,.2f}")
    print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate: {metrics['win_rate_pct']:.2f}%")

    print(f"\n🧬 Learning Progress:")
    progress = results["learning_progress"]
    print(f"  Improvement: {progress.get('improvement', 0):.4f}")

    print(f"\n🏆 Best Methods by Role:")
    for role, methods in results["best_methods"].items():
        print(f"  {role.capitalize()}: {methods}")

    print(f"\n📊 Method Popularity:")
    for role, popularity in results["method_popularity"].items():
        top_methods = sorted(popularity.items(), key=lambda x: x[1], reverse=True)[:3]
        formatted = [(m, f'{p:.1%}') for m, p in top_methods]
        print(f"  {role.capitalize()}: {formatted}")

    print(f"\n✅ Experiment logged to: {logger.summary_file}")
    print(f"{'='*60}\n")


def _run_export_mode(args):
    """Export experiment data for NeurIPS paper."""
    from .services.neurips_export import export_for_neurips

    print(f"\n{'='*60}")
    print("📄 NEURIPS EXPORT")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment_id}")
    print(f"Output dir: {args.output_dir}")
    print(f"Formats: {args.format}")
    print(f"{'='*60}\n")

    try:
        outputs = export_for_neurips(
            log_dir=args.log_dir,
            experiment_id=args.experiment_id,
            output_dir=args.output_dir,
            formats=args.format,
        )

        print("\n✅ Export complete!")
        print(f"Generated {len(outputs)} files:")
        for name, path in outputs.items():
            print(f"  - {name}: {path}")

    except FileNotFoundError:
        print(f"\n❌ Error: Experiment '{args.experiment_id}' not found in {args.log_dir}")
        print("   Run 'trading-agents backtest' first to generate experiment data.")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")


def _run_api_server(args):
    """Start the dashboard API server."""
    print(f"\n{'='*60}")
    print("🌐 POPAGENT API SERVER")
    print(f"{'='*60}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Log dir: {args.log_dir}")
    print(f"{'='*60}\n")

    try:
        from .api.server import run_server
        run_server(host=args.host, port=args.port, log_dir=args.log_dir)
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("   Install required packages: pip install fastapi uvicorn")
    except Exception as e:
        print(f"\n❌ Server failed: {e}")


if __name__ == "__main__":
    main()
