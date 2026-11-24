"""逐标点比较领域微调，并输出规则干预的逐句结果。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import read_json, write_tables


def tables():
    result = read_json("task17_domain/results/domain_sft_v2.json")
    names = ["逗号", "句号", "顿号", "分号", "冒号", "问号", "感叹号"]
    changes = []
    for label, symbol in zip(names, "，。、；：？！"):
        scores = [100 * result["arms"][arm]["古文_per_label"][symbol] for arm in ("基线", "精选SFT")]
        changes.append((f"{label}  {symbol}", *scores))
    yield "图1 领域微调前后，哪些符号发生变化", ["项目", "基线", "精选 SFT"], changes

    fields = ["name", "gold", "predicted", "with_rule", "exact", "exact_with_rule"]
    for arm, examples in result["constraint_variants"].items():
        rows = [[example[key] for key in fields] for example in examples]
        yield arm + "逐句干预", ["名称", "金标", "规则前", "规则后", "修改前正确", "修改后正确"], rows


if __name__ == "__main__":
    write_tables(__file__, tables())
