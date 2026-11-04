"""Prompt construction shared by SFT data building and inference (must be identical in both)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
END = "END"

SYSTEM = (
    "Bạn là trợ lý giọng nói LifeMate. Mỗi lượt bạn chỉ xuất ĐÚNG MỘT dòng:\n"
    '- một tool call dạng JSON {"name": "...", "arguments": {...}}, hoặc\n'
    "- END khi đã xong hoặc khi không có app nào xử lý được yêu cầu.\n"
    "Quy tắc: chỉ dùng tool đã liệt kê; đọc dữ liệu (get/list/search) trước khi dùng ID; không gọi thừa; "
    "thiếu tham số bắt buộc thì gọi sys.ask_missing_info rồi dừng; ID phải lấy từ kết quả tool."
)


def load_catalog(*paths) -> dict[str, dict]:
    """{tool_name: tool dict} merged over catalogs."""
    out: dict[str, dict] = {}
    for p in paths:
        for app in json.loads(Path(p).read_text(encoding="utf-8"))["apps"]:
            for t in app["tools"]:
                out[t["name"]] = t
    return out


def _param(name: str, p: dict, required: set) -> str:
    t = p.get("type", "any")
    if "enum" in p:
        t = "|".join(map(str, p["enum"]))
    extra = []
    if "x-ref" in p:
        extra.append("ref=" + p["x-ref"])
    if p.get("x-match") == "fuzzy":
        extra.append("text")
    for k in ("minimum", "maximum", "format"):
        if k in p:
            extra.append(f"{k}={p[k]}")
    s = f"{name}{'*' if name in required else ''}:{t}"
    return s + (f"[{','.join(extra)}]" if extra else "")


def compact_tool(t: dict) -> str:
    sch = t.get("parameters", {})
    req = set(sch.get("required", []))
    args = ", ".join(_param(k, v, req) for k, v in sch.get("properties", {}).items())
    line = f"{t['name']}({args})"
    if sch.get("x-one-of"):
        line += " ONE_OF=" + " | ".join("+".join(g) for g in sch["x-one-of"])
    if t.get("description"):
        line += " # " + t["description"]
    return line


def schema_block(installed_apps: dict, catalog: dict) -> str:
    lines = []
    for app, info in installed_apps.items():
        for tn in info["tools"]:
            lines.append(compact_tool(catalog[tn]))
    lines.append(compact_tool(catalog["sys.ask_missing_info"]))
    return "\n".join(lines)


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def history_block(history: list, user_texts: list[str]) -> str:
    """user_texts: transcript of each user turn in history order (gold audio_text in train, ASR at inference)."""
    out, ui = [], 0
    for h in history:
        if h["role"] == "user":
            out.append("Người dùng: " + user_texts[ui])
            ui += 1
        else:
            if h.get("tool_calls"):
                for c, r in zip(h["tool_calls"], h.get("tool_results", [])):
                    out.append("  gọi " + dumps(c) + " -> " + dumps(r))
            out.append("Trợ lý: " + h.get("text", ""))
    return "\n".join(out) if out else "(không có)"


def build_first_user(ctx: dict, history: list, history_texts: list[str], transcript: str, catalog: dict) -> str:
    dt = datetime.fromisoformat(ctx["current_time"])
    return (
        f"[THỜI GIAN] {ctx['current_time']} ({WEEKDAYS[dt.weekday()]}) múi giờ {ctx['timezone']}\n"
        f"[VỊ TRÍ] {dumps(ctx.get('location', {}))}\n"
        f"[HỒ SƠ] {dumps(ctx.get('user_profile', {}))}\n"
        f"[TOOL]\n{schema_block(ctx['installed_apps'], catalog)}\n"
        f"[LỊCH SỬ]\n{history_block(history, history_texts)}\n"
        f"[YÊU CẦU]\n{transcript}"
    )


def result_message(res) -> str:
    return "KẾT QUẢ: " + dumps(res)


def call_text(c: dict) -> str:
    return dumps({"name": c["name"], "arguments": c["arguments"]})


def build_messages(sample: dict, catalog: dict, transcript: str, history_texts: list[str], with_target: bool = True) -> list[dict]:
    """Chat messages for a train sample. Assistant turns = gold calls (+ END unless the last call is sys.ask_missing_info)."""
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_first_user(sample["context"], sample["history"], history_texts, transcript, catalog)},
    ]
    if not with_target:
        return msgs
    calls, results = sample["label"]["tool_calls"], sample["label"]["tool_results"]
    for i, c in enumerate(calls):
        msgs.append({"role": "assistant", "content": call_text(c)})
        if c["name"] == "sys.ask_missing_info":
            return msgs
        msgs.append({"role": "user", "content": result_message(results[i])})
    msgs.append({"role": "assistant", "content": END})
    return msgs
