"""固定类别集与分类器，计算更换词向量的五折均值差。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    rows = read_json("task4_gensim/results/classification_v3.json")
    means = {(r["classes"], r["clf"], r["vec"]): r["f1_mean"] for r in rows}
    differences = []
    for size in (2, 4, 6):
        gains = [100 * (means[size, clf, "ft"] - means[size, clf, "w2v"]) for clf in ("LR", "SVM")]
        differences.append((f"{size} 类", *gains))
    yield "图1 在同一分类器上更换词向量", ["项目", "逻辑回归", "线性 SVM"], differences
    columns = ["classes", "clf", "vec", "n", "f1_mean", "f1_std"]
    yield "交叉验证汇总", ["类别数", "分类器", "向量", "样本数", "五折平均F1", "折间标准差"], [
        [row[key] for key in columns] for row in rows
    ]


if __name__ == "__main__":
    write_tables(__file__, tables())
