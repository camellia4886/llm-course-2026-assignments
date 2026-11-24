"""用十二条人工构造句检验古汉语任职状态规则。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import diagnostic_tables, write_tables

SAMPLES = [
    ("任命", "甲拜太守。", "已任"),
    ("任命", "甲出为太守。", "已任"),
    ("任命", "以甲为太守。", "已任"),
    ("任命", "甲领太守事。", "已任"),
    ("拟议", "拟以甲为太守。", "拟议"),
    ("拟议", "议拜甲为太守。", "拟议"),
    ("拟议", "欲使甲领太守事。", "拟议"),
    ("未任", "甲未拜太守。", "未任"),
    ("未任", "将拜甲为太守，诏寝。", "未任"),
    ("存疑", "或云甲为太守。", "待核"),
    ("存疑", "一书记甲为太守，别本作乙。", "待核"),
    ("存疑", "甲为太守，传闻未审。", "待核"),
]


def predict_status(text):
    cues = [("待核", ("或云", "别本", "传闻")), ("未任", ("未拜",)), ("拟议", ("拟", "议拜"))]
    for label, words in cues:
        if any(word in text for word in words):
            return label
    return "已任"


def tables():
    rows = [(group, text, gold, "已任", predict_status(text)) for group, text, gold in SAMPLES]
    return diagnostic_tables(rows, ["见人官即记已任", "加入状态线索"])


if __name__ == "__main__":
    write_tables(__file__, tables())
