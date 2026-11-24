"""比较两种检索预算下的命中数。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    languages = read_json("course_labs/results/segmentation_benchmark.json")["languages"]
    totals, differences = [], []
    for lang, label in [("zh", "中文"), ("en", "英文")]:
        retrieval = languages[lang]["retrieval"]
        for budget in ["前三块", "等字数"]:
            hits = []
            for method in ["fixed_length", "lexical_score", "editor_paragraph"]:
                row = retrieval[method]
                recall = row["recall"]["3"] if budget == "前三块" else row["recall_at_char_budget_top_5"]
                hits.append(round(recall * row["queries"]))
            totals.append((label, budget, retrieval["fixed_length"]["queries"], *hits))
            differences.append((f"{label} · {budget}", hits[1] - hits[0]))
    yield "图1 词汇切分的优势随阅读口径反转", ["项目", "命中差"], differences
    yield "命中题数", ["语言", "口径", "题数", "定长", "词汇切分", "编辑段落"], totals


if __name__ == "__main__":
    write_tables(__file__, tables())
