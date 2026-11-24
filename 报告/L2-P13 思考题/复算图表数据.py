"""用十二条人工构造句比较两条去重规则。"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import diagnostic_tables, write_tables

SAMPLES = [
    ("助词重复", "他到了了。", "他到了。"),
    ("助词重复", "她写的的信到了。", "她写的信到了。"),
    ("助词重复", "名单中有三个人了了。", "名单中有三个人了。"),
    ("助词重复", "这是新的的版本。", "这是新的版本。"),
    ("合法重复", "请看看这份材料。", "请看看这份材料。"),
    ("合法重复", "他天天读书。", "他天天读书。"),
    ("合法重复", "妈妈今天上班。", "妈妈今天上班。"),
    ("合法重复", "事情渐渐明朗。", "事情渐渐明朗。"),
    ("其他错误", "我把书放桌子。", "我把书放在桌子上。"),
    ("其他错误", "他以经到了。", "他已经到了。"),
    ("其余无错", "消息已经送达。", "消息已经送达。"),
    ("其余无错", "他已经回来了。", "他已经回来了。"),
]


def tables():
    rows = []
    for group, text, gold in SAMPLES:
        broad = re.sub(r"([\u4e00-\u9fff])\1", r"\1", text)
        narrow = re.sub(r"([了的])\1", r"\1", text)
        rows.append((group, text, gold, broad, narrow))
    return diagnostic_tables(rows, ["所有叠字去重", "仅删重复了或的"])


if __name__ == "__main__":
    write_tables(__file__, tables())
