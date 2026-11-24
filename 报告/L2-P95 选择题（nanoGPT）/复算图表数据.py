"""导出训练批次损失和单层前向理论运算构成。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    result = read_json("course_labs/results/nanogpt_exercise.json")
    training = [(f"{row['step']} 步", row["loss"]) for row in result["history"]]
    yield "图1 训练批次损失的四个记录点", ["项目", "本次训练批"], training

    flops = result["full_square_flops_per_block"]
    total = sum(flops.values())
    terms = [("Q/K/V 投影", "qkv"), ("输出投影", "attention_projection"),
             ("前馈网络", "ffn"), ("注意力配对", "attention_pairing")]
    shares = [(label, 100 * flops[key] / total) for label, key in terms]
    yield "图2 一个 Transformer 块的运算分布", ["项目", "占比"], shares


if __name__ == "__main__":
    write_tables(__file__, tables())
