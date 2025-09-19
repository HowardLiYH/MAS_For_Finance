"""CLI entrypoint: reads YAML and runs the pipeline."""
import argparse, sys, yaml

# For local import path accomodation
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
# If having multiple core from multiple folders use the following instead:
# sys.path.insert(0, str(ROOT/"src"))
sys.path.append(str(ROOT / "src"))

# NEW: load .env (requires `python-dotenv` in your env)
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")   # <— explicitly point to your repo’s .env

from core.data_pipeline import run_pipeline

def main():

    # agrparse is designed for reading arguments from the command line

    # sets up an argument parser so the script can accept command-line arguments
    # Description is only a label/description for --help purpouse
    ap = argparse.ArgumentParser(description="Phase 1.1 runner")

    # Defines what kind of input the script accepts from the command line
    # Accepting optional flag named '--config'
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "btc4h.yaml"))

    # Reads what the user type in the terminal and return in args
    args = ap.parse_args()

    # Here our args.config == 'configs/btc4h.yaml'
    # FInd the file path and open the yaml file
    with open(args.config, "r", encoding="utf-8") as f:
        # Reads the yaml file and convert it into python dict
        cfg = yaml.safe_load(f)
        # Here **cfg unpacks the dict into keywords argument
    out = run_pipeline(**cfg)
    print("✨Summary of our execution:")
    for key, value in out.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
