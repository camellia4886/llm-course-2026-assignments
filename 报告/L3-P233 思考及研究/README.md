# L3-P233 思考及研究

古文领域微调与句末规则　·　L3 Part 2 第 227–233、236–237 页

精选 SFT 得到 0.4197 的古文全文宏 F1，再加句末规则后降为 0.3691；同一规则在 12 个短句上却使整句正确数从 6 增为 8。

[下载 Word 报告](L3-P233%20思考及研究.docx)　·　[课程作业索引](../../README.md)

## 核验结果

在 constraint_variants 的 1A、1B 两行检查“乎”的位置。基线加规则能补上“此事合礼乎”的问号，却也把“此事合乎礼”截为“此事合乎？礼”。

原始记录　[domain_sft_v2.json](../../task17_domain/results/domain_sft_v2.json) · [error_examples_v2.json](../../task17_domain/results/error_examples_v2.json)

![图1 领域微调前后，哪些符号发生变化](图表/图1%20领域微调前后，哪些符号发生变化.png)

[查看图1数据（CSV）](图表/图1%20领域微调前后，哪些符号发生变化.csv)

逐项数据　[基线逐句干预](图表/基线逐句干预.csv) · [精选SFT逐句干预](图表/精选SFT逐句干预.csv)

## 复算

在仓库根目录运行。脚本用 Python 标准库生成同名 CSV，可与上方数据对照。

```shell
python "报告/L3-P233 思考及研究/复算图表数据.py"
```

实验程序与补充材料　[run_domain_sft.py](../../task17_domain/run_domain_sft.py) · [课堂习题补全.md](../../course_labs/课堂习题补全.md) · [完整复现条件](../../docs/复现指南.md)
