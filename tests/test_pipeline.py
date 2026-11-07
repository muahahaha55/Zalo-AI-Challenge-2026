"""Integration check on the real train data: the oracle backend replays gold, so the whole pipeline must score 100.
Skipped when the competition data is not on this machine."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate import pipeline  # noqa: E402
from lifemate.agent import AgentConfig  # noqa: E402
from lifemate.metric import load_fuzzy_params, score_sample  # noqa: E402
from lifemate.paths import ROOT, processed_dir, split_dir  # noqa: E402

HAVE_DATA = (split_dir("train") / "train.jsonl").exists() and (processed_dir() / "gold_hash_train.json").exists()


def test_transcripts_clean_and_asr(tmp_path):
    item = {"audio": "a/x.wav", "audio_text": "clean", "history": [
        {"role": "user", "audio": "a/u1.wav", "audio_text": "h-clean"}, {"role": "assistant", "text": "ok"}]}
    t = pipeline.Transcripts.clean()
    assert t.request(item) == "clean" and t.history(item) == ["h-clean"]
    p = tmp_path / "asr.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in [{"path": "a/x.wav", "text": "noisy"}, {"path": "a/u1.wav", "text": "h-noisy"}]), encoding="utf-8")
    t = pipeline.Transcripts.from_asr_file(p)
    assert t.request(item) == "noisy" and t.history(item) == ["h-noisy"]
    with pytest.raises(ValueError):
        pipeline.Transcripts.from_spec("whisper", "train")


@pytest.mark.skipif(not HAVE_DATA, reason="competition data not available")
def test_oracle_replay_scores_100_and_reports_stop_reasons():
    runner = pipeline.SplitRunner("train", AgentConfig(use_rules=False), pipeline.Transcripts.clean(), oracle=True)
    items = runner.items(limit=60)
    rows = list(runner.run(items))
    gold_hash = json.loads((processed_dir() / "gold_hash_train.json").read_text(encoding="utf-8"))
    fuzzy = load_fuzzy_params(split_dir("train") / "train_tools_catalog.json")
    scores = [score_sample(r["tool_calls"], r["state_hash"], it["label"]["tool_calls"], gold_hash[it["id"]], fuzzy) for r, it in zip(rows, items, strict=True)]
    assert scores == [1.0] * 60
    assert {o.stop_reason.value for o in runner.stats.outcomes} <= {"end", "ask"}
    assert "60 samples" in runner.stats.summary() and ROOT.exists()


def test_live_transcripts_batch_one_sample_cache_and_survive_failures(tmp_path):
    calls = []

    class FakeTranscriber:
        def transcribe(self, paths):
            calls.append([p.name for p in paths])
            return [f"text of {p.name}" for p in paths]

    item = {"audio": "audio/s/s.wav", "history": [{"role": "user", "audio": "audio/s/s_u1.wav"}, {"role": "assistant", "text": "ok"}]}
    live = pipeline.LiveTranscripts(FakeTranscriber(), tmp_path)
    live.prepare(item)
    live.prepare(item)                                                              # cached: no second recognition
    assert calls == [["s.wav", "s_u1.wav"]]                                         # one batch holding the request and the history
    assert live.request(item) == "text of s.wav" and live.history(item) == ["text of s_u1.wav"]

    class Broken:
        def transcribe(self, paths):
            raise RuntimeError("cuda error")

    broken = pipeline.LiveTranscripts(Broken(), tmp_path)
    broken.prepare(item)
    assert broken.failures == 1 and broken.request(item) == "" and broken.history(item) == [""]


def test_stub_transcripts_and_null_backend():
    item = {"audio": "a", "history": [{"role": "user", "audio": "b"}, {"role": "assistant"}, {"role": "user", "audio": "c"}]}
    stub = pipeline.StubTranscripts()
    assert stub.request(item) == "" and stub.history(item) == ["", ""]
    assert pipeline.NullBackend()([{"role": "user", "content": "x"}]) == "END"
