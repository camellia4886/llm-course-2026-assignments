"""从准确率和测试题数还原各个距离探针的答对数。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    arms = read_json("course_labs/results/next_bert_probe.json")["arms"]
    configs = [("局部窗口", "local"), ("全局注意力", "global"),
               ("全局 + 跨度", "global_span"), ("全局 + 检索", "global_retrieval")]
    counts = []
    for label, arm in configs:
        test = arms[arm]["test"]
        correct = [round(test[group]["accuracy"] * test[group]["size"]) for group in ("short", "long")]
        counts.append((label, *correct))
    yield "图1 同一探针中短距与长距答对多少题", ["项目", "短距离", "长距离"], counts


if __name__ == "__main__":
    write_tables(__file__, tables())
