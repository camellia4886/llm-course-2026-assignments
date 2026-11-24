# -*- coding: utf-8 -*-
"""从人民日报 9801 原始语料重建文章并做规则弱标注。

原始行格式：19980101-01-001-007 词1 词2{pinyin} ...
- 同一文章 ID 前缀（19980101-01-001）的行属于同一篇文章
- 自选标签：按关键词规则弱标注（体育/外交/经济/科技/政务/文化），
  命中数不足的标为"其他"；弱标注的噪声将在报告中人工抽检估计。
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "1998-01-2003版-带音.txt"
OUT = Path(__file__).parent / "articles_9801.jsonl"

PINYIN = re.compile(r"\{[a-züA-ZÜ0-9]+\}")
SENT_ID = re.compile(r"^\d{8}-\d{2}-\d{3}-\d{3}\s+")
ART_ID = re.compile(r"^(\d{8}-\d{2}-\d{3})")

RULES = {
    "体育": ["体育", "比赛", "足球", "球队", "教练", "运动员", "冠军", "奥运会",
             "联赛", "比分", "球员", "球门", "裁判", "乒乓球", "羽毛球", "游泳", "田径"],
    "外交": ["外交", "大使", "访问", "会见", "会谈", "双边", "联合国", "外长",
             "国务卿", "友好关系", "国际社会", "总统", "首脑", "领导人"],
    "经济": ["经济", "市场", "企业", "金融", "银行", "投资", "贸易", "价格",
             "增长", "改革", "国有企业", "财政", "税收", "农业", "工业", "商品"],
    "科技": ["科技", "科学", "技术", "研究", "卫星", "计算机", "信息产业",
             "网络", "基因", "航天", "发明", "专利", "实验室"],
    "政务": ["会议", "召开", "讲话", "报告", "同志", "党委", "政府", "人大",
             "政协", "常委会", "总书记", "代表", "审议", "提案"],
    "文化": ["文化", "艺术", "文学", "电影", "电视", "音乐", "出版", "图书",
             "博物馆", "文物", "戏剧", "作家", "演员"],
}


def clean(line: str):
    line = SENT_ID.sub("", line)
    line = PINYIN.sub("", line)
    return line.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW, help="1998-01 raw GBK file")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"Pass the course's raw January file with --source: {args.source}")
    articles = {}  # art_id -> list of sentences
    with args.source.open(encoding="gbk", errors="replace") as f:
        for line in f:
            m = ART_ID.match(line)
            if not m:
                continue
            text = clean(line)
            if text:
                articles.setdefault(m.group(1), []).append(text)

    print(f"重建文章数: {len(articles)}")
    labeled = Counter()
    out_rows = []
    for art_id, sents in articles.items():
        text = "".join(sents)
        scores = {c: sum(text.count(kw) for kw in kws)
                  for c, kws in RULES.items()}
        best, score = max(scores.items(), key=lambda kv: kv[1])
        label = best if score >= 3 else "其他"
        labeled[label] += 1
        out_rows.append({"id": art_id, "label": label,
                         "len": len(text), "text": text})
    print("标签分布:", dict(labeled))
    with args.output.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已写出 {len(out_rows)} 篇 -> {args.output}")


if __name__ == "__main__":
    main()
