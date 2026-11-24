# -*- coding: utf-8 -*-
"""清理人民日报 1998 上半年带音标注语料。

原始格式（GBK 编码，每行一句）：
    19980101-01-001-007 １９９７年 ， 是 中国 发展 历史 上{shang5} 非常 重要 ...
需要清理的"多余 tag 和标签"：
    1. 行首句子 ID（如 19980101-01-001-007）
    2. 词后的拼音标注（如 上{shang5}、这{zhe4}）
保留：
    空格分词结果、标点、全角数字等正文内容
"""
import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO.parents[1] / "1998-上半年-带音(1)" / "1998-上半年-带音"
OUT_DIR = REPO / "data"

SENT_ID = re.compile(r"^\d{8}-\d{2}-\d{3}-\d{3}\s+")
PINYIN = re.compile(r"\{[a-züA-ZÜ0-9]+\}")  # 词后拼音标注 {shang5} {zhe4}


def clean_line(line: str) -> str:
    line = SENT_ID.sub("", line)
    line = PINYIN.sub("", line)
    return line.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    files = {}
    for path in sorted(args.source_dir.glob("*.txt")):
        match = re.match(r"1998-(0[1-6])(?:-|$)", path.stem)
        if match:
            month = match[1]
            if month in files:
                parser.error(f"Multiple input files for month {month}")
            files[month] = path
    expected = {f"{n:02d}" for n in range(1, 7)}
    if files.keys() != expected:
        parser.error(f"Expected the six raw 1998 monthly files in {args.source_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"lines": 0, "words": 0, "chars": 0, "pinyin_removed": 0,
             "sentence_ids": 0, "decode_replacements": 0}
    for month, f in sorted(files.items()):
        out_path = args.out_dir / f"{month}.txt"
        n_pinyin = 0
        with open(f, encoding="gbk", errors="replace") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                n_pinyin += len(PINYIN.findall(line))
                stats["sentence_ids"] += bool(SENT_ID.match(line))
                stats["decode_replacements"] += line.count("\ufffd")
                cleaned = clean_line(line)
                if not cleaned:
                    continue
                fout.write(cleaned + "\n")
                stats["lines"] += 1
                stats["words"] += len(cleaned.split())
                stats["chars"] += len(cleaned)
        stats["pinyin_removed"] += n_pinyin
        print(f"{f.name} -> {out_path.name}  拼音标注 {n_pinyin} 处")
    print(f"\n合计：{stats['lines']} 句，{stats['words']} 词，{stats['chars']} 字符，"
          f"清理句子 ID {stats['sentence_ids']} 个，拼音标注 {stats['pinyin_removed']} 处，"
          f"GBK 解码替换位 {stats['decode_replacements']}")


if __name__ == "__main__":
    main()
