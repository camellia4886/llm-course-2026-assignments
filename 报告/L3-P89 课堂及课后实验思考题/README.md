# L3-P89 课堂及课后实验思考题

古诗词英译的提示对照　·　L3 Part 1 第 19、43、52、54、66、87、89–90 页

留出集上 V2 比 V1 多用 42.6% 的输入 token。四对译文文字各有差别，按篇目和种子配对的 16 项二元评分完全相同。

[下载 Word 报告](L3-P89%20课堂及课后实验思考题.docx)　·　[课程作业索引](../../README.md)

## 核验结果

在 poetry.holdout 比较 V1 和 V2。两者分别使用 516 和 736 个输入 token，四项判据的通过数均为 1、4、1、4，对应语义关系、意象、忠实、格式。在 rows 中按 poem 和 seed 配对，4 组 × 4 项通过记录完全一致，4 组英文译文逐组比对均有差别。

原始记录　[summary.json](../../task16_prompteng/results/v2/summary.json) · [review_scores.json](../../task16_prompteng/results/v2/review_scores.json)

![图1 留出译诗的四项通过情况](图表/图1%20留出译诗的四项通过情况.png)

[查看图1数据（CSV）](图表/图1%20留出译诗的四项通过情况.csv)

![图2 提示在哪个集合减少了失分](图表/图2%20提示在哪个集合减少了失分.png)

[查看图2数据（CSV）](图表/图2%20提示在哪个集合减少了失分.csv)

## 复算

在仓库根目录运行。脚本用 Python 标准库生成同名 CSV，可与上方数据对照。

```shell
python "报告/L3-P89 课堂及课后实验思考题/复算图表数据.py"
```

实验程序与补充材料　[summarize_v2.py](../../task16_prompteng/summarize_v2.py) · [run_controlled_v2.py](../../task16_prompteng/run_controlled_v2.py) · [完整复现条件](../../docs/复现指南.md)
