"""比较三元模型剪枝和语料规模。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    original, pruned = read_json("task1_ngram/results/ppl_trigram_prune.json")
    fields = [("保留类型", "types"), ("训练时间", "train_s"),
              ("总用时", "total_s"), ("测试困惑度", "PPL")]
    indices = [(label, 100, 100 * pruned[key] / original[key]) for label, key in fields]
    yield "图1 剪枝后的库存、耗时与困惑度", ["项目", "未剪枝", "最低频次 2"], indices

    scales = read_json("task1_ngram/results/kenlm.json")["training_scale"]
    storage = [(f"{len(row['months'])} 个月", row["arpa_bytes"] / row["words"]) for row in scales]
    yield "图2 累计月份与单位输入的存储成本", ["项目", "单位存储"], storage


if __name__ == "__main__":
    write_tables(__file__, tables())
