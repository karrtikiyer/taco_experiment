from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
TEMPERATURE = 0.7
TOP_P = 0.95
NUM_SAMPLES = 10
MAX_NEW_TOKENS = 2048

K_VALUES = [1, 3, 5, 8, 10]

SMOKE_TEST_SIZE = 20
FULL_TEST_SIZE = 100
SEED = 42

MAX_EXECUTION_MEMORY_BYTES = 512 * 1024 * 1024  # 512 MB per executed solution
MEMORY_WARNING_MB = 4096  # warn if parent RSS exceeds this during evaluation

CODEBLEU_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
