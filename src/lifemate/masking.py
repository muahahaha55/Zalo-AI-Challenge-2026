"""Tool-name masking for SFT (training only; inference never sees masked names).

Idea (Hammer, Lin et al. 2024, arXiv:2410.04587, "function masking"): in a copy of a training sample, every app prefix and
tool name is replaced by a random string, consistently inside that sample and differently in every sample. The model can
then no longer pick a tool by a name it memorised; it has to read the tool description, which is what an unseen app at
test time offers.

What is masked, and why it stays consistent:
- tool names `app.tool` -> `<app alias>.<tool alias>` in the [TOOL] list, the gold calls, the calls in [LỊCH SỬ], the
  `tool_name` argument of `sys.ask_missing_info`, and any tool name quoted in a tool description;
- the app prefix of `x-ref` values (`contacts.contacts` -> `<contacts alias>.contacts`), so the real app name does not leak
  through a reference.
What is NOT masked:
- parameter names: the prompt (`prompt.compact_tool`) shows no parameter description, so a masked parameter name would
  leave the model nothing to infer its meaning from (Hammer keeps parameter descriptions; this prompt does not show them);
- `sys.ask_missing_info` itself (the same name exists at test time), tool descriptions, request text, tool results.
Train-split samples only (never dev), like the other builders in lifemate.augment.
"""
from __future__ import annotations

import json
import random
import re
import string

from .prompt import END, build_messages

SYS = "sys"
ASK = "sys.ask_missing_info"
_ALPHA = string.ascii_lowercase
_ALNUM = string.ascii_lowercase + string.digits


def _random_word(rng: random.Random, taken: set[str], lo: int = 4, hi: int = 6) -> str:
    """A lowercase word that starts with a letter and is not in `taken` (real names and aliases already used)."""
    while True:
        w = rng.choice(_ALPHA) + "".join(rng.choice(_ALNUM) for _ in range(rng.randint(lo, hi) - 1))
        if w not in taken:
            taken.add(w)
            return w


class Masker:
    """Per-sample alias table. Aliases are created on first use, so the result depends only on the rng state and on
    the (fixed) order in which names are met."""

    def __init__(self, rng: random.Random, catalog: dict):
        self.rng = rng
        real_parts = set()
        for name in catalog:
            real_parts.update(name.split("."))
        self.taken = set(real_parts)          # never produce a real app or tool word: an alias must not mean anything
        self.apps: dict[str, str] = {}
        self.tools: dict[str, str] = {}

    def app(self, app: str) -> str:
        if app == SYS:
            return app
        if app not in self.apps:
            self.apps[app] = _random_word(self.rng, self.taken)
        return self.apps[app]

    def tool(self, name: str) -> str:
        if name == ASK or "." not in name:
            return name
        if name not in self.tools:
            self.tools[name] = f"{self.app(name.split('.', 1)[0])}.{_random_word(self.rng, self.taken)}"
        return self.tools[name]

    def ref(self, value: str) -> str:
        """x-ref 'app.collection': only the app part is masked (the collection word carries the meaning, like a description)."""
        if not isinstance(value, str) or "." not in value:
            return value
        app, _, rest = value.partition(".")
        return f"{self.app(app)}.{rest}"


def _plain_copy(o):
    """Copy of plain JSON data (a sample or a catalog entry; no env involved), as elsewhere in lifemate.augment."""
    return json.loads(json.dumps(o))


def _mask_call(c: dict, m: Masker) -> dict:
    out = _plain_copy(c)
    out["name"] = m.tool(c["name"])
    if c["name"] == ASK and isinstance(out.get("arguments"), dict) and isinstance(out["arguments"].get("tool_name"), str):
        out["arguments"]["tool_name"] = m.tool(out["arguments"]["tool_name"])
    return out


def _mask_tool(t: dict, m: Masker, pattern: re.Pattern | None) -> dict:
    out = _plain_copy(t)
    out["name"] = m.tool(t["name"])
    if "app" in out and isinstance(out["app"], str):
        out["app"] = m.app(out["app"])
    if pattern is not None and isinstance(out.get("description"), str):
        out["description"] = pattern.sub(lambda g: m.tool(g.group(0)), out["description"])
    for p in (out.get("parameters") or {}).get("properties", {}).values():
        if isinstance(p, dict) and "x-ref" in p:
            p["x-ref"] = m.ref(p["x-ref"])
    return out


def mask_sample(item: dict, catalog: dict, rng: random.Random) -> tuple[dict, dict, dict]:
    """Return (masked item, masked catalog, alias table {real tool name: alias, for every name given an alias}).
    The input is not modified.
    The masked catalog holds exactly the tools the masked prompt can show (installed tools + sys.ask_missing_info)."""
    m = Masker(rng, catalog)
    installed = item["context"]["installed_apps"]
    # alias order: installed tools first (prompt order), then any other tool the sample's calls mention
    for info in installed.values():
        for tn in info["tools"]:
            m.tool(tn)
    calls = list(item["label"]["tool_calls"])
    for h in item["history"]:
        if h["role"] == "assistant":
            calls += h.get("tool_calls") or []
    for c in calls:
        m.tool(c["name"])
        if c["name"] == ASK and isinstance((c.get("arguments") or {}).get("tool_name"), str):
            m.tool(c["arguments"]["tool_name"])
    # a description may quote ANY catalog tool, also one of an app this sample does not install (ewallet.pay_qr quotes
    # food_delivery.list_restaurants): every quoted tool name gets an alias, so no real name survives in the prompt
    names = sorted((n for n in catalog if n != ASK and "." in n), key=len, reverse=True)
    pattern = re.compile(r"(?<![\w.])(" + "|".join(map(re.escape, names)) + r")(?![\w])") if names else None

    new = _plain_copy(item)
    new["context"]["installed_apps"] = {
        m.app(a): {**info, "tools": [m.tool(tn) for tn in info["tools"]]} for a, info in installed.items()}
    new["label"]["tool_calls"] = [_mask_call(c, m) for c in item["label"]["tool_calls"]]
    for h in new["history"]:
        if h["role"] == "assistant" and h.get("tool_calls"):
            h["tool_calls"] = [_mask_call(c, m) for c in h["tool_calls"]]
    cat = {}
    for info in installed.values():
        for tn in info["tools"]:
            t = _mask_tool(catalog[tn], m, pattern)
            cat[t["name"]] = t
    cat[ASK] = catalog[ASK]
    return new, cat, dict(m.tools)


def _check_round_trip(item: dict, msgs: list[dict]) -> None:
    """Same contract as scripts/build_sft.check_round_trip, on the masked sample: the assistant turns decode to the
    masked gold calls, and END is last exactly when the gold does not end in an ask."""
    decoded = [json.loads(x["content"]) for x in msgs if x["role"] == "assistant" and x["content"] != END]
    calls = item["label"]["tool_calls"]
    assert decoded == [{"name": c["name"], "arguments": c["arguments"]} for c in calls], item["id"]
    assert (msgs[-1]["content"] == END) == (not calls or calls[-1]["name"] != ASK), item["id"]


def masked_samples(train_items: list[dict], catalog: dict, asr: dict, rng: random.Random, n: int):
    """Masked copies of up to `n` train samples (each sample at most once, drawn in a seeded random order). Returns (rows, stats).
    A row's transcripts are the ASR text with probability 0.5 when `asr` is given, else the clean text (as in
    lifemate.augment.synthetic_samples). Samples whose tools are missing from the catalog are skipped."""
    order = list(train_items)
    rng.shuffle(order)
    rows: list[dict] = []
    stats = {"masked": 0, "skipped_unknown_tool": 0, "asr": 0}
    for it in order:
        if stats["masked"] >= n:
            break
        tools = [tn for info in it["context"]["installed_apps"].values() for tn in info["tools"]]
        if any(tn not in catalog for tn in tools) or ASK not in catalog:
            stats["skipped_unknown_tool"] += 1
            continue
        use_asr = bool(asr) and rng.random() < 0.5
        new, cat, _ = mask_sample(it, catalog, rng)
        if use_asr:
            text = asr[it["audio"]]
            history = [asr[h["audio"]] for h in it["history"] if h["role"] == "user"]
            stats["asr"] += 1
        else:
            text = it["audio_text"]
            history = [h["audio_text"] for h in it["history"] if h["role"] == "user"]
        msgs = build_messages(new, cat, text, history)
        _check_round_trip(new, msgs)
        rows.append({"id": f"{it['id']}#mask", "messages": msgs})
        stats["masked"] += 1
    rng.shuffle(rows)
    return rows, stats
