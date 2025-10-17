# Ensure the bundled src/ is importable and load .env for keys
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]         # .../MAS_Final_With_Agents
SRC  = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env so OPENAI_API_KEY / SERPAPI_KEY are available for news fetch
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass  # ok if python-dotenv not installed
