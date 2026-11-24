# 数据说明

## 语料

人民日报 1998 年上半年（1–6 月）切分标注版语料（2003 版带音，GBK 编码），经 `task1_ngram/clean_corpus.py` 清理：

- 去掉行首句子 ID（如 `19980101-01-001-007`）
- 去掉词后拼音标注（如 `上{shang5}`、`这{zhe4}`）
- 保留空格分词结果、全部正文标点、全角数字

清理后共 6 个文件、123,761 句。完整原始和清理后的课堂语料、由此产生的文章和 BERT 训练材料均留在本机。复现时请使用课程提供的语料重新生成。

| 文件 | 月份 | 用途 |
|---|---|---|
| `01.txt`–`06.txt` | 1998 年 1–6 月 | 任务1 n-gram 全部使用 |
| `04.txt`–`06.txt` | 4–6 月 | 任务 4 的 v2 跨月表示训练。v3 的独立训练语料另见下文 |
| `01.txt` | 1 月 | 任务 7 BERT 断句训练集 |
| `02.txt` | 2 月 | 任务 7 BERT 断句验证集 |
| `03.txt` | 3 月 | 任务 7 BERT 断句独立测试集 |

CMRC、DRCD 与 SQuAD 的复现地址和校验信息见 [公开数据说明](public_benchmarks/README.md)。

## 派生数据

- `task1_ngram/`：直接读取本目录，无派生文件
- `task4_gensim/articles_9801.jsonl`：由 `prep_articles.py` 从 1998-01 带音原文按文章 ID 重建 + 关键词规则弱标注（体育/外交/经济/科技/政务/文化/其他）
- `task6_bert/data/train.jsonl`、`test.jsonl`：由 `build_data.py` 切句去标点生成，每行 `{"input": 无标点字符序列, "labels": [每字标签]}`（0=O，1–7=，。、；：？！）
- `task17_domain/guwen.tsv`：古文标准标点文本（篇名\t原文），训练 5 篇 + 测试 2 篇，公共领域

## 重跑预处理（可选）

将课程提供的带音原文解压后，按实际路径指定语料目录和一月原文件。

```bash
# 语料清理（输出到 data/）
python task1_ngram/clean_corpus.py --source-dir "/path/to/1998-上半年-带音"
# 文章重建与弱标注
python task4_gensim/prep_articles.py --source "/path/to/1998-上半年-带音/1998-01-2003版-带音.txt"
# BERT 断句数据
python task6_bert/build_data.py
```
