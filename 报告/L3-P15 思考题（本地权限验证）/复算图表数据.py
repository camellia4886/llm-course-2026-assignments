"""统计预期处置与本地模型建议的三分类混淆矩阵。"""
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    cases = read_json("course_labs/results/safety_boundary_model.json")["cases"]
    labels = {"ALLOW": "放行", "REVIEW": "复核", "DENY": "拒绝"}
    counts = Counter((case["gold"], case["model_advice"]) for case in cases)
    matrix = [(label, *(counts[gold, prediction] for prediction in labels)) for gold, label in labels.items()]
    yield "图1 本地模型建议与预设权限的混淆矩阵", ["项目", *labels.values()], matrix


if __name__ == "__main__":
    write_tables(__file__, tables())
