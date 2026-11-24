"""比较解码耗时与留出新闻的标点识别质量。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    cases = read_json("task6_bert/results/batch_decode_v2.json")["test_cases"]
    methods = [("串行", "serial"), *((f"批 {n}", f"batch_{n}") for n in (1, 8, 16, 32, 64))]
    timings = []
    for label, method in methods:
        times = [cases[c]["results"][method]["median_seconds"] * 1000
                 for c in ("march_news", "classical", "classical_long")]
        timings.append((label, *times))
    yield "图1 窗口固定后批量解码的耗时", ["项目", "新闻", "古文", "加长古文"], timings

    scores = read_json("task6_bert/results/eval_v2.json")["heldout"]["per_label"]
    names = ["逗号", "句号", "顿号", "分号", "冒号", "问号", "感叹号"]
    by_symbol = [(f"{name}  {symbol}", 100 * scores[symbol]["precision"], 100 * scores[symbol]["recall"])
                 for name, symbol in zip(names, "，。、；：？！")]
    yield "图2 新闻留出集的逐类精确率与召回率", ["项目", "精确率", "召回率"], by_symbol


if __name__ == "__main__":
    write_tables(__file__, tables())
