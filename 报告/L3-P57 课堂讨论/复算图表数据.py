"""核算两种平均损失对应的样本权重和监督区间占比。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    result = read_json("course_labs/results/sft_loss_check.json")
    # 手算张量有 1 个短样本目标和 3 个长样本目标。
    # 样本均值 = (S+L)/2，token 均值 = (S+3L)/4。
    a, b = result["example_mean"], result["token_mean"]
    short_loss, long_loss = 3 * a - 2 * b, 2 * b - a
    weighted = []
    for weight in (0, 0.25, 0.5, 0.75, 1):
        label = f"{weight:.0%}" + {0.5: " · 样本均值", 0.75: " · token 均值"}.get(weight, "")
        weighted.append((label, (1 - weight) * short_loss + weight * long_loss))
    yield "图1 同一组预测在不同权重下的损失", ["项目", "加权损失"], weighted

    masks = [(index, row["tokens"], row["supervised"], row["supervised"] / row["tokens"])
             for index, row in enumerate(result["examples"], start=1)]
    yield "模板掩码", ["会话", "实际token", "监督token", "监督占比"], masks


if __name__ == "__main__":
    write_tables(__file__, tables())
