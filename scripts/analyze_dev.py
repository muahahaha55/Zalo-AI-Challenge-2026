"""Error analysis of a dev submission (Step 10): where do the lost points come from?

  .venv/Scripts/python.exe -I -X utf8 scripts/analyze_dev.py --sub outputs/logs/dev_sft_phowl.jsonl [--ids data/processed/dev_ids.txt] [--show 12]

Every sample whose score < 1 gets ONE primary cause (priority order below); the table shows how many samples and how many points
(1 - score) each cause costs. Also prints the most frequent wrong parameters and the worst examples (gold vs predicted).
Dev/train only: it uses the train labels and get_state hashes cached by eval.py, never public/private test data.
"""
import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("lm_eval", ROOT / "scripts" / "eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
from lifemate.metric import calls_match, load_fuzzy_params, score_sample  # noqa: E402

ASK = "sys.ask_missing_info"


def pair_leftovers(sub, gold, fuzzy):
    """Greedy exact matches first; then pair what is left by tool name. Returns (param_diffs, missing, extra)."""
    sub, gold = list(sub), list(gold)
    for g in list(gold):
        for s in sub:
            if calls_match(s, g, fuzzy):
                sub.remove(s); gold.remove(g)
                break
    diffs, missing = [], []
    for g in gold:
        cand = [s for s in sub if s.get("name") == g["name"]]
        if not cand:
            missing.append(g["name"])
            continue
        def n_diff(c):
            ca, ga_ = c.get("arguments") or {}, g["arguments"] or {}
            return sum(ca.get(k) != ga_.get(k) for k in set(ca) | set(ga_))
        best = min(cand, key=n_diff)
        sub.remove(best)
        sa, ga = best.get("arguments") or {}, g["arguments"] or {}
        for k in sorted(set(sa) | set(ga)):
            if sa.get(k) != ga.get(k):
                diffs.append((g["name"], k, ga.get(k, "<absent>"), sa.get(k, "<absent>")))
    return diffs, missing, [s.get("name") for s in sub]


def classify(it, s, gold_hash, fuzzy):
    G, S = it["label"]["tool_calls"], s["tool_calls"]
    sc = score_sample(S, s["state_hash"], G, gold_hash, fuzzy)
    if sc >= 0.999999:
        return sc, "perfect", {}
    gold_ask = any(c["name"] == ASK for c in G)
    sub_ask = any(c["name"] == ASK for c in S)
    detail = {}
    if not G:
        return sc, "oos: gọi tool dù phải để trống", {"first": S[0]["name"]}
    if not S:
        return sc, "không gọi gì (END ngay) dù phải hành động/hỏi", {}
    if gold_ask and not sub_ask:
        return sc, "phải hỏi lại (ask) nhưng đi thực thi", {"first": S[0]["name"]}
    if sub_ask and not gold_ask:
        return sc, "hỏi lại (ask) nhưng phải thực thi", {}
    diffs, missing, extra = pair_leftovers(S, G, fuzzy)
    detail = {"diffs": diffs, "missing": missing, "extra": extra}
    hash_ok = s["state_hash"] == gold_hash
    if gold_ask:
        return sc, ("ask: sai tên tool/tham số" if (diffs or missing or extra) else "ask: lệch nhẹ"), detail
    if not hash_ok:
        if diffs:
            return sc, "exec: sai tham số (hash lệch)", detail
        if missing and not extra:
            return sc, "exec: thiếu call (hash lệch)", detail
        if extra and not missing:
            return sc, "exec: thừa call làm đổi trạng thái (hash lệch)", detail
        return sc, "exec: thiếu và thừa call (hash lệch)", detail
    if extra and not missing and not diffs:
        return sc, "exec: hash đúng, thừa call (precision)", detail
    if missing and not extra and not diffs:
        return sc, "exec: hash đúng, thiếu call (recall)", detail
    return sc, "exec: hash đúng, call lệch tham số/thừa/thiếu", detail


def short(calls, n=3):
    out = []
    for c in calls[:n]:
        a = json.dumps(c.get("arguments", {}), ensure_ascii=False)
        out.append(f"{c['name']}({a[:70]})")
    return " ; ".join(out) + (f" ... (+{len(calls) - n})" if len(calls) > n else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--ids", default=str(ROOT / "data" / "processed" / "dev_ids.txt"))
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "analysis_dev.jsonl"))
    a = ap.parse_args()

    items = ev.load_train()
    keep = set(Path(a.ids).read_text().split())
    items = [i for i in items if i["id"] in keep]
    fuzzy = load_fuzzy_params(ev.DATA / "train" / "train_tools_catalog.json")
    gh = ev.gold_hashes(ev.load_train())
    sub = {json.loads(l)["id"]: json.loads(l) for l in open(a.sub, encoding="utf-8")}
    missing_ids = [i["id"] for i in items if i["id"] not in sub]
    print(f"N={len(items)} mẫu dev; thiếu trong submission: {len(missing_ids)}")

    rows = []
    for it in items:
        s = sub.get(it["id"], {"tool_calls": [], "state_hash": None})
        sc, cause, detail = classify(it, s, gh[it["id"]], fuzzy)
        rows.append((it, s, sc, cause, detail))
    total = len(rows)
    print(f"điểm = {100 * sum(r[2] for r in rows) / total:.2f}\n")

    # ---- cause table
    agg = defaultdict(lambda: [0, 0.0])
    for it, s, sc, cause, _ in rows:
        agg[cause][0] += 1
        agg[cause][1] += 1 - sc
    lost_total = sum(v[1] for k, v in agg.items() if k != "perfect")
    print(f"{'nguyên nhân chính':62s} {'số mẫu':>7s} {'điểm mất':>9s} {'% tổng mất':>10s}")
    for cause, (n, lost) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        if cause == "perfect":
            continue
        print(f"{cause:62s} {n:7d} {100 * lost / total:8.2f}đ {100 * lost / lost_total:9.1f}%")
    print(f"{'(đúng hoàn toàn)':62s} {agg['perfect'][0]:7d}\n")

    # ---- predicted vs gold kind
    def pk(calls):
        return "oos" if not calls else ("ask" if any(c["name"] == ASK for c in calls) else "exec")
    conf = Counter((ev.kind(it), pk(s["tool_calls"])) for it, s, *_ in rows)
    print("ma trận gold -> dự đoán (loại mẫu):")
    for g in ("exec", "ask", "oos"):
        print(f"  gold={g:4s} ->", "  ".join(f"{p}={conf[(g, p)]}" for p in ("exec", "ask", "oos")))

    # ---- wrong params / missing / extra
    pc, mc, ec = Counter(), Counter(), Counter()
    examples = defaultdict(list)
    for it, s, sc, cause, d in rows:
        for tool, key, gv, sv in d.get("diffs", []):
            pc[(tool, key)] += 1
            if len(examples[(tool, key)]) < 2:
                examples[(tool, key)].append((gv, sv))
        mc.update(d.get("missing", []))
        ec.update(d.get("extra", []))
    print("\ntham số sai nhiều nhất (tool.tham_số: số lần, ví dụ gold -> dự đoán):")
    for (tool, key), n in pc.most_common(12):
        ex = "; ".join(f"{str(g)[:28]!r}->{str(p)[:28]!r}" for g, p in examples[(tool, key)])
        print(f"  {tool}.{key}: {n}   {ex}")
    print("\ntool bị thiếu nhiều nhất:", mc.most_common(8))
    print("tool bị thừa nhiều nhất :", ec.most_common(8))

    # ---- by app / by number of calls
    by_app = defaultdict(list)
    for it, s, sc, *_ in rows:
        for app in {c["name"].split(".")[0] for c in it["label"]["tool_calls"]} or {"(oos)"}:
            by_app[app].append(sc)
    print("\nđiểm theo app xuất hiện trong gold (n, trung bình):")
    for app, v in sorted(by_app.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {app:18s} n={len(v):3d} mean={100 * sum(v) / len(v):6.1f}")

    # ---- worst examples
    bad = [r for r in rows if r[2] < 1]
    bad.sort(key=lambda r: (r[2], r[0]["id"]))
    print(f"\n{a.show} mẫu sai điển hình (điểm 0 trước):")
    for it, s, sc, cause, d in bad[: a.show]:
        print(f"- {it['id']} [{ev.kind(it)}] score={sc:.2f} | {cause}")
        print(f"    nói : {it['audio_text'][:110]}")
        print(f"    gold: {short(it['label']['tool_calls'])}")
        print(f"    pred: {short(s['tool_calls'])}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for it, s, sc, cause, d in rows:
            f.write(json.dumps({"id": it["id"], "kind": ev.kind(it), "score": sc, "cause": cause,
                                "diffs": d.get("diffs", []), "missing": d.get("missing", []), "extra": d.get("extra", [])},
                               ensure_ascii=False) + "\n")
    print(f"\nchi tiết từng mẫu: {a.out}")


if __name__ == "__main__":
    main()
