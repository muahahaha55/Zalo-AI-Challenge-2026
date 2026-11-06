"""Split-level pipeline: transcripts -> agent -> submission lines. Shared by scripts/run_infer.py and the container entry point.

Only the organisers' public ToolEnv API is used (validate / call / get_tool_calls / get_state_hash), exactly one ToolEnv
is created per sample, and every call made on it ends up in the submission.
"""
from __future__ import annotations

import importlib
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .agent import Agent, AgentConfig, Generate, OracleBackend, Outcome
from .jsonl import iter_jsonl
from .paths import processed_dir, split_dir
from .prompt import build_messages, load_catalog


class Transcripts:
    """Text of every user utterance of a sample: the gold `audio_text` (train only) or an ASR output file."""

    def __init__(self, asr: dict[str, str] | None = None):
        self._asr = asr

    @classmethod
    def clean(cls) -> Transcripts:
        return cls(None)

    @classmethod
    def from_asr_file(cls, path: str | Path) -> Transcripts:
        return cls({d["path"]: d["text"] for d in iter_jsonl(path)})

    @classmethod
    def from_spec(cls, spec: str, split: str) -> Transcripts:
        """'clean' or 'asr:<tag>' (data/processed/asr_<tag>_<split>.jsonl)."""
        if spec == "clean":
            return cls.clean()
        if spec.startswith("asr:"):
            return cls.from_asr_file(processed_dir() / f"asr_{spec[4:]}_{split}.jsonl")
        raise ValueError(f"unknown transcript source {spec!r} (use 'clean' or 'asr:<tag>')")

    def request(self, item: dict) -> str:
        return item["audio_text"] if self._asr is None else self._asr[item["audio"]]

    def history(self, item: dict) -> list[str]:
        users = [h for h in item["history"] if h["role"] == "user"]
        return [h["audio_text"] for h in users] if self._asr is None else [self._asr[h["audio"]] for h in users]


def load_tool_env_class(split: str):
    """The split's own zaic_tools (each split ships its own version)."""
    sys.path.insert(0, str(split_dir(split)))
    for name in [m for m in sys.modules if m == "zaic_tools" or m.startswith("zaic_tools.")]:
        del sys.modules[name]                       # a previous split's copy must not be reused
    return importlib.import_module("zaic_tools").ToolEnv


class HFBackend:
    """Greedy Hugging Face generation, optionally with a LoRA adapter. Deterministic (no sampling)."""

    def __init__(self, model_path: str, adapter: str | None = None, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).cuda().eval()
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).eval()
        self.model = model

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def __call__(self, messages: list[dict]) -> str:
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                           enable_thinking=False, return_dict=True).to("cuda")
        with self._torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


@dataclass
class RunStats:
    outcomes: list[Outcome] = field(default_factory=list)

    def summary(self) -> str:
        if not self.outcomes:
            return "no samples"
        t = sorted(o.elapsed_s for o in self.outcomes)
        reasons = Counter(o.stop_reason.value for o in self.outcomes)
        return (f"{len(t)} samples; mean {sum(t) / len(t):.2f}s p95 {t[int(.95 * (len(t) - 1))]:.2f}s max {t[-1]:.2f}s; "
                f"stop reasons {dict(reasons.most_common())}")


class SplitRunner:
    """Run the agent over the samples of one split, yielding one submission line per sample, in order."""

    def __init__(self, split: str, config: AgentConfig, transcripts: Transcripts, backend: Generate | None = None,
                 oracle: bool = False, tool_env_cls=None):
        if (backend is None) == (not oracle):
            raise ValueError("pass either a backend or oracle=True")
        self.split = split
        self.transcripts = transcripts
        self.catalog = load_catalog(split_dir(split) / f"{split}_tools_catalog.json")
        self.agent = Agent(self.catalog, backend or (lambda messages: ""), config)
        self.oracle = oracle
        self.tool_env_cls = tool_env_cls or load_tool_env_class(split)
        self.stats = RunStats()

    def items(self, ids: Iterable[str] | None = None, limit: int = 0, shard: tuple[int, int] | None = None) -> list[dict]:
        items = list(iter_jsonl(split_dir(self.split) / f"{self.split}.jsonl"))
        if ids is not None:
            keep = set(ids)
            items = [i for i in items if i["id"] in keep]
        if limit:
            items = items[:limit]
        if shard:
            k, n = shard
            items = items[k::n]
        return items

    def run_item(self, item: dict) -> Outcome:
        request, history = self.transcripts.request(item), self.transcripts.history(item)
        env = self.tool_env_cls(self.split, item["id"], root=str(split_dir(self.split).parent))   # exactly one env per sample
        generate = OracleBackend(build_messages(item, self.catalog, request, history)) if self.oracle else None
        outcome = self.agent.run(item, env, request, history, generate=generate)
        self.stats.outcomes.append(outcome)
        return outcome

    def run(self, items: list[dict], on_progress: Callable[[int, int, float], None] | None = None) -> Iterator[dict]:
        t0 = time.monotonic()
        for n, item in enumerate(items, 1):
            outcome = self.run_item(item)
            if on_progress:
                on_progress(n, len(items), (time.monotonic() - t0) / n)
            yield {"id": item["id"], "tool_calls": outcome.tool_calls, "state_hash": outcome.state_hash}
