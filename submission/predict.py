"""Inference entry point required by the organisers' Docker guide: read /data/<split>/, write /result/submission.jsonl.

Each output line: {"id", "tool_calls", "state_hash", "time"} with time = milliseconds for that sample (speech recognition
included). Load-model time and predict time are printed to stdout. Hyper-parameters are fixed defaults (seed, greedy
decoding, 25 s budget per sample); the model folders come from environment variables set in the Dockerfile:
  LIFEMATE_LLM      /code/saved_models/llm      (Qwen3-4B fine-tuned)
  LIFEMATE_ADAPTER  /code/saved_models/adapter  (LoRA adapter; absent when the adapter is merged into the LLM weights)
  LIFEMATE_ASR      /code/saved_models/asr      (PhoWhisper-large fine-tuned, merged)
No network is used; only the split's own zaic_tools (validate / call / get_tool_calls / get_state_hash) touch the
environment, with exactly one ToolEnv per sample.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src" if (HERE / "src").is_dir() else HERE.parent / "src"))   # /code/src in the image, ../src in the repo
from lifemate.container import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
