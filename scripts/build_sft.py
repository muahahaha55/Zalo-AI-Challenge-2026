"""Build SFT data from train: data/processed/sft_{train,dev}.jsonl (+ dev_ids.txt) and verify round-trip.
Usage: python -I -X utf8 scripts/build_sft.py [--tokenizer Qwen/Qwen3-4B] [--asr-tag phowl_ft] [--aug-oos 100 --aug-variants 1] [--hist-max 700]
Extra data (train split only, never dev) goes to SEPARATE files so the base files stay unchanged:
  --aug-oos N / --aug-variants K   sft_train_aug.jsonl  (see lifemate.augment.synthetic_samples)
  --aug-mask N                     sft_train_mask.jsonl: masked copies of N train samples, app and tool names replaced by random
                                   strings per sample (lifemate.masking; teaches reading tool descriptions instead of names)
  --hist-max N                     sft_train_hist.jsonl (earlier turns of train samples' history, see lifemate.augment.history_samples)
  --retract-copies K / --retract-replace N / --retract-cancel M
                                   sft_train_retract.jsonl: the real self-correction samples repeated K times (safe: real labels), plus N
                                   "<short request>, à thôi, <real request>" and M "<short request>, à thôi khỏi" compositions that teach
                                   ignoring what the speaker took back (lifemate.augment.retraction_*)
  --callgen FILE                   sft_train_cg.jsonl: samples from scripts/gen_calls.py + gen_text.py (transplanted real call
                                   segments, request text written by a local model), filtered by lifemate.callgen.text_ok
  --hold-apps a,b,c                'unseen apps' experiment: samples whose gold uses these apps become holdout/eval_ids.txt, the others
                                   lose the apps from their prompts; everything is written to data/processed/holdout/"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.augment import history_samples, retraction_oversample, retraction_samples, split_holdout, synthetic_samples  # noqa: E402
from lifemate.grouping import near_duplicate_groups  # noqa: E402
from lifemate.jsonl import read_jsonl, write_jsonl  # noqa: E402
from lifemate.manifest import write_manifest  # noqa: E402
from lifemate.paths import processed_dir, split_dir  # noqa: E402
from lifemate.prompt import END, build_messages, load_catalog  # noqa: E402

DEV_SHARE = 0.1


def choose_dev_groups(groups: list[int], seed: int) -> set[int]:
    """Whole groups (shuffled with `seed`) until ~DEV_SHARE of the samples: near-duplicates never straddle train and dev.
    NOTE: the shuffle depends on the numeric group labels, so changing how groups are numbered changes the dev set."""
    sizes = Counter(groups)
    order = sorted(sizes)
    random.Random(seed).shuffle(order)
    chosen: set[int] = set()
    covered = 0
    for gid in order:
        if covered >= DEV_SHARE * len(groups):
            break
        chosen.add(gid)
        covered += sizes[gid]
    return chosen


def check_round_trip(item: dict, msgs: list[dict]) -> None:
    """The assistant turns must decode back to the gold calls, and END must appear exactly when the gold does not end in ask."""
    import json
    decoded = [json.loads(m["content"]) for m in msgs if m["role"] == "assistant" and m["content"] != END]
    calls = item["label"]["tool_calls"]
    assert decoded == [{"name": c["name"], "arguments": c["arguments"]} for c in calls], item["id"]
    assert (msgs[-1]["content"] == END) == (not calls or calls[-1]["name"] != "sys.ask_missing_info"), item["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--asr-tag", default=None, help="also write *_asr.jsonl with ASR transcripts (data/processed/asr_<tag>_train.jsonl)")
    ap.add_argument("--aug-oos", type=int, default=0)
    ap.add_argument("--aug-variants", type=int, default=0)
    ap.add_argument("--aug-mask", type=int, default=0, help="N: also write sft_train_mask.jsonl with N name-masked train samples")
    ap.add_argument("--retract-copies", type=int, default=0, help="repeat the real self-correction samples K times")
    ap.add_argument("--retract-replace", type=int, default=0, help="N compositions '<retracted request>, <marker>, <real request>'")
    ap.add_argument("--retract-cancel", type=int, default=0, help="M compositions '<request>, <marker> khỏi' (answer: no call)")
    ap.add_argument("--hold-apps", default="", help="comma-separated apps to hide from training ('unseen apps' experiment)")
    ap.add_argument("--hist-max", type=int, default=0, help="N: also write sft_train_hist.jsonl with up to N samples from train histories")
    ap.add_argument("--callgen", default="", help="callgen_text.jsonl from scripts/gen_text.py -> sft_train_cg.jsonl")
    a = ap.parse_args()

    train_dir, out = split_dir("train"), processed_dir()
    items = read_jsonl(train_dir / "train.jsonl")
    cat = load_catalog(train_dir / "train_tools_catalog.json")
    if a.hold_apps:
        hold = {x.strip() for x in a.hold_apps.split(",") if x.strip()}
        items, eval_ids = split_holdout(items, hold)
        out = out / "holdout"
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_ids.txt").write_text("\n".join(eval_ids) + "\n")
        (out / "hold_apps.txt").write_text(",".join(sorted(hold)) + "\n")
        print(f"HOLDOUT {sorted(hold)}: {len(eval_ids)} eval samples held out of training, {len(items)} samples left for training/dev")
    groups = near_duplicate_groups([it["audio_text"] for it in items])
    dev_groups = choose_dev_groups(groups, a.seed)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def emit(path: Path, rows) -> None:
        write_jsonl(path, rows)
        written.append(path)

    asr = {}
    if a.asr_tag:
        asr = {d["path"]: d["text"] for d in read_jsonl(processed_dir() / f"asr_{a.asr_tag}_train.jsonl")}

    rows: dict[str, list[dict]] = {"sft_train": [], "sft_dev": [], "sft_train_asr": [], "sft_dev_asr": []}
    for it, gid in zip(items, groups):
        side = "dev" if gid in dev_groups else "train"
        history = [h["audio_text"] for h in it["history"] if h["role"] == "user"]
        msgs = build_messages(it, cat, it["audio_text"], history)
        check_round_trip(it, msgs)
        rows[f"sft_{side}"].append({"id": it["id"], "messages": msgs})
        if asr:   # same targets, but the request and history turns are the ASR transcripts (noisy, as at inference)
            noisy = [asr[h["audio"]] for h in it["history"] if h["role"] == "user"]
            rows[f"sft_{side}_asr"].append({"id": it["id"], "messages": build_messages(it, cat, asr[it["audio"]], noisy)})
    for name, data in rows.items():
        if data:
            emit(out / f"{name}.jsonl", data)

    train_only = [it for it, gid in zip(items, groups) if gid not in dev_groups]
    if a.aug_oos or a.aug_variants:
        aug, st = synthetic_samples(train_only, cat, asr, random.Random(a.seed + 1), a.aug_oos, a.aug_variants)
        emit(out / "sft_train_aug.jsonl", aug)
        print(f"augmentation -> sft_train_aug.jsonl: {len(aug)} samples ({st['oos']} synthetic oos, {st['variants']} variants)")
    if a.aug_mask:
        from lifemate.masking import masked_samples
        masked, st = masked_samples(train_only, cat, asr, random.Random(a.seed + 5), a.aug_mask)
        emit(out / "sft_train_mask.jsonl", masked)
        print(f"name masking -> sft_train_mask.jsonl: {len(masked)} samples; {st}")
    if a.hist_max:
        hist, st = history_samples(train_only, cat, asr, random.Random(a.seed + 2), a.hist_max)
        emit(out / "sft_train_hist.jsonl", hist)
        print(f"history turns -> sft_train_hist.jsonl: {len(hist)} samples; candidates {st}")
    if a.retract_copies or a.retract_replace or a.retract_cancel:
        real = retraction_oversample(train_only, cat, asr, random.Random(a.seed + 4), a.retract_copies)
        comp, st = retraction_samples(train_only, cat, random.Random(a.seed + 3), a.retract_replace, a.retract_cancel)
        emit(out / "sft_train_retract.jsonl", real + comp)
        print(f"retraction -> sft_train_retract.jsonl: {len(real) + len(comp)} samples "
              f"({len(real)} real repeats, {st['replace']} replace, {st['cancel']} cancel)")
    if a.callgen:
        from lifemate.callgen import apply_holdout, text_ok
        allowed = {it["id"] for it in train_only}          # base and donors must be training samples of THIS split
        hold = {x.strip() for x in a.hold_apps.split(",") if x.strip()}
        from lifemate.leakcheck import normalize
        real_texts = {normalize(it["audio_text"]) for it in items}      # a generated sentence that equals a real one adds nothing
        cg_rows, why = [], Counter()
        for s in read_jsonl(a.callgen):
            if s["base"] not in allowed or not set(s["donors"]) <= allowed:
                why["not_train"] += 1
                continue
            if s.get("verified") is False:                 # the local model judged the text not to match the calls
                why["not_verified"] += 1
                continue
            if s.get("roundtrip") is False:                # scripts/verify_callgen.py: acting on the text did not reach the label
                why["roundtrip_failed"] += 1
                continue
            s = apply_holdout(s, hold) if hold else s
            if s is None:
                why["held_app"] += 1
                continue
            if normalize(s["audio_text"]) in real_texts:
                why["copied_real_sentence"] += 1
                continue
            ok, reason = text_ok(s["audio_text"], s, cat)
            why[reason] += 1
            if ok:
                msgs = build_messages(s, cat, s["audio_text"], [])
                check_round_trip(s, msgs)
                cg_rows.append({"id": s["id"], "messages": msgs})
        emit(out / "sft_train_cg.jsonl", cg_rows)
        inputs_extra = [Path(a.callgen)]
        print(f"callgen -> sft_train_cg.jsonl: {len(cg_rows)} samples; {dict(why)}")
    (out / "dev_ids.txt").write_text("\n".join(r["id"] for r in rows["sft_dev"]) + "\n")
    written.append(out / "dev_ids.txt")
    print(f"groups={len(set(groups))} train={len(rows['sft_train'])} dev={len(rows['sft_dev'])} (dev groups={len(dev_groups)})")
    inputs = [train_dir / "train.jsonl", train_dir / "train_tools_catalog.json"]
    if a.asr_tag:
        inputs.append(processed_dir() / f"asr_{a.asr_tag}_train.jsonl")
    if a.callgen:
        inputs += inputs_extra
    write_manifest(out / "MANIFEST.build_sft.json", "build_sft", vars(a), inputs, written)

    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        both = rows["sft_train"] + rows["sft_dev"]
        full = sorted(len(tok.apply_chat_template(r["messages"], tokenize=True)) for r in both)
        first = sorted(len(tok(r["messages"][1]["content"])["input_ids"]) for r in both)
        q = lambda v, p: v[int(p * (len(v) - 1))]  # noqa: E731
        print("full-sample tokens  mean=%.0f p50=%d p95=%d max=%d" % (sum(full) / len(full), q(full, .5), q(full, .95), full[-1]))
        print("first user msg tok  mean=%.0f p50=%d p95=%d max=%d" % (sum(first) / len(first), q(first, .5), q(first, .95), first[-1]))


if __name__ == "__main__":
    main()
