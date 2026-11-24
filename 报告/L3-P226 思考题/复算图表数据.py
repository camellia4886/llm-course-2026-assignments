"""把 SFT 与组相对更新的检查点按实际更新次数对齐。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    result = read_json("task3_survey/results/tiny_posttrain.json")
    by_arm = {}
    for name, row in result["arms"].items():
        history = row["history"]
        updates_per_round = row["parameter_updates"] // result["configuration"]["rounds"]
        by_arm[name] = {
            checkpoint["step"] * updates_per_round: round(checkpoint["test_accuracy"] * result["heldout_pairs"])
            for checkpoint in history
        }
    steps = sorted(set(by_arm["sft"]) | set(by_arm["grpo"]))
    rows = [(step, by_arm["sft"].get(step), by_arm["grpo"].get(step)) for step in steps]
    yield "图1 按实际参数更新数重排留出成绩", ["项目", "SFT", "组相对更新"], rows


if __name__ == "__main__":
    write_tables(__file__, tables())
