import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate import datacheck  # noqa: E402

CALL = '{"name":"a.t","arguments":{"x":1}}'
ASK = '{"name":"sys.ask_missing_info","arguments":{"tool_name":"a.t","params":["x"]}}'


def conv(*assistant_and_results, first_user="u"):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": first_user}]
    for i, content in enumerate(assistant_and_results):
        msgs.append({"role": "assistant", "content": content})
        if i < len(assistant_and_results) - 1:
            msgs.append({"role": "user", "content": "KẾT QUẢ: {}"})
    return {"id": "x", "messages": msgs}


def test_valid_shapes_pass():
    assert datacheck.check_conversation(conv(CALL, "END")) == []
    assert datacheck.check_conversation(conv("END")) == []
    assert datacheck.check_conversation(conv(CALL, ASK)) == []                      # ask ends the sample, no END after it


def test_structural_mistakes_are_reported():
    assert datacheck.check_conversation(conv(CALL))                                  # ends with a call that is not ask: END missing
    assert datacheck.check_conversation(conv("END", CALL))                           # END before the last turn
    assert datacheck.check_conversation(conv(ASK, CALL, "END"))                      # ask not last
    assert datacheck.check_conversation(conv("not json", "END"))
    assert datacheck.check_conversation(conv('{"name":"a.t","arguments":[1]}', "END"))
    bad = conv(CALL, "END")
    bad["messages"][3]["role"] = "assistant"                                         # alternation broken
    assert datacheck.check_conversation(bad)
    assert datacheck.check_conversation({"id": "y", "messages": [{"role": "user", "content": "u"}]})


def test_file_checks_find_duplicates_dev_leaks_and_held_apps():
    rows = [dict(conv(CALL, "END"), id="train_1#aug0"), dict(conv(CALL, "END"), id="train_1#aug0"),
            dict(conv("END"), id="dev_9#h1"), dict(conv(CALL, "END", first_user="dùng flights.search"), id="train_2")]
    rep = datacheck.check_rows("f", rows, dev_ids={"dev_9"}, hold_apps={"flights"})
    text = " | ".join(rep.problems)
    assert "duplicate id" in text and "dev sample" in text and "held-out app" in text
    assert rep.rows == 4 and rep.step_examples == 7


def test_clean_file_has_no_problems_and_counts_steps():
    rows = [dict(conv(CALL, CALL, "END"), id="a"), dict(conv("END"), id="b")]
    rep = datacheck.check_rows("f", rows, dev_ids={"zzz"})
    assert rep.problems == [] and rep.rows == 2 and rep.step_examples == 4


class CharTokenizer:
    """One token per character: enough to exercise the length logic."""

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        return "".join(m["content"] for m in messages) + "<gen>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text)}


def test_step_lengths_and_over_limit_count():
    row = conv(CALL, "END")
    lengths = datacheck.step_token_lengths(row["messages"], CharTokenizer())
    assert len(lengths) == 2 and lengths[0] < lengths[1]                              # the second step has more context
    rep = datacheck.check_rows("f", [row], tokenizer=CharTokenizer(), max_len=lengths[0])
    assert rep.too_long == 1 and rep.longest == lengths[1]
    assert json.dumps(rep.problems) == "[]"


def test_base_id_strips_the_augmentation_suffix():
    assert datacheck.base_id("train_0007#aug0") == "train_0007" and datacheck.base_id("train_0007") == "train_0007"
