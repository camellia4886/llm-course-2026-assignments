"""从匿名盲评汇总导出译诗四项通过数和开发留出失分。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    poetry = read_json("task16_prompteng/results/v2/summary.json")["poetry"]
    holdout = poetry["holdout"]
    criteria = ("relation", "imagery", "fidelity", "format")
    scores = [(version, *(row["passed"][key] for key in criteria)) for version, row in holdout.items()]
    yield "图1 留出译诗的四项通过情况", ["项目", "语义关系", "意象", "忠实", "格式"], scores
    errors = [(version, poetry["development"][version]["errors"], row["errors"])
              for version, row in holdout.items()]
    yield "图2 提示在哪个集合减少了失分", ["项目", "开发", "留出"], errors


if __name__ == "__main__":
    write_tables(__file__, tables())
