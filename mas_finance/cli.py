from pathlib import Path
import argparse
from .config import load_config, OrchestratorInput
from .orchestrator.graph import iterate_once

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--interval", type=str, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "agents.yaml"),
    )
    args = parser.parse_args()

    appcfg = load_config(args.config)
    if args.symbol:
        appcfg.symbol = args.symbol
    if args.interval:
        appcfg.timeframe = args.interval

    # Typed object with known fields (no more “attribute defined outside __init__”)
    c = OrchestratorInput(
        symbol=appcfg.symbol,
        interval=appcfg.timeframe,
        appcfg=appcfg,
    )
    iterate_once(c)

if __name__ == "__main__":
    main()
