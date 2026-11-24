"""按篇名配对两种语体，复算平均句数绝对误差。"""
from statistics import mean
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    records = read_json("l1_optional_checkable/results/guwen_vs_baihua_v2.json")["per_text"]
    domains = {
        label: {r["name"]: r for r in records if r["domain"] == label}
        for label in ("古文", "白话")
    }
    classical, modern = domains["古文"], domains["白话"]
    if classical.keys() != modern.keys():
        raise ValueError("古文和白话篇名不匹配")
    errors = [
        (label, mean(abs(record["sents_delta"]) for record in texts.values()))
        for label, texts in domains.items()
    ]
    yield "图1 每篇平均偏离金标的句数", ["项目", "句数误差"], errors
    pairs = [(name, raw["sents_delta"], modern[name]["sents_delta"],
              raw["macroF1_出现类"], modern[name]["macroF1_出现类"])
             for name, raw in classical.items()]
    yield "逐篇配对", ["篇名", "古文句数偏差", "白话句数偏差", "古文出现类宏F1", "白话出现类宏F1"], pairs


if __name__ == "__main__":
    write_tables(__file__, tables())
