"""图表复算脚本共用的文件读写，所有结果均以仓库根目录为基准。"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def diagnostic_tables(rows, columns):
    """每个诊断用例依次保存组别、原句、金标和两种策略的预测。"""
    yield "逐句诊断", ["分组", "输入", "预期", *columns], rows
    scores = []
    for name in dict.fromkeys(row[0] for row in rows):
        group = [row for row in rows if row[0] == name]
        hits = [sum(row[index] == row[2] for row in group) for index in (3, 4)]
        scores.append((f"{name}（{len(group)}）", *hits))
    yield "图1 各现象的逐句通过数", ["项目", *columns], scores


def write_tables(script_path, tables):
    parser = argparse.ArgumentParser(description="从实验记录复算报告所用的 CSV")
    parser.add_argument("--output", type=Path,
                        default=Path(script_path).resolve().parent / "复算输出",
                        help="CSV 输出目录，默认写入本题的复算输出目录")
    output = parser.parse_args().output
    output.mkdir(exist_ok=True, parents=True)
    for name, header, rows in tables:
        path = output / (name + ".csv")
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        print(path)
