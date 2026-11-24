"""按相同申请重放三种处置规则。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    samples = read_json("task8_sensitive/results/mini_demo.json")["samples"]
    policies = ["一律放行", "有命中即拦", "按接收者查表"]
    replay = [
        (s["id"], s["expect"], "allow", "block" if s["recognized"] else "allow", s["decision"])
        for s in samples
    ]
    yield "逐项重放", ["编号", "预期", *policies], replay
    scores = []
    for gold, label in [("allow", "放行"), ("review", "复核"), ("block", "阻断")]:
        group = [record for record in replay if record[1] == gold]
        hits = [sum(record[index] == gold for record in group) for index in (2, 3, 4)]
        scores.append((f"{label}（{len(group)}）", *hits))
    yield "图1 换处置规则后，哪类申请受影响", ["项目", *policies], scores


if __name__ == "__main__":
    write_tables(__file__, tables())
