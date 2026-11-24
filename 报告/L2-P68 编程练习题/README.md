# L2-P68 编程练习题

标点恢复的质量与速度　·　L2 Part 2 第 68 页

两万字新闻在固定窗口下，批量从 1 提至 8，中位用时从 644.0 降至 385.3 毫秒；继续提到 64，仅再节省 19.2 毫秒。

[下载 Word 报告](L2-P68%20编程练习题.docx)　·　[课程作业索引](../../README.md)

## 核验结果

在 test_cases.march_news.results 中固定窗口策略，对比 batch_1、batch_8、batch_64 的三次测量中位数。图 1 另列 serial，其窗口边界由自适应策略确定。

原始记录　[batch_decode_v2.json](../../task6_bert/results/batch_decode_v2.json) · [eval_v2.json](../../task6_bert/results/eval_v2.json)

![图1 窗口固定后批量解码的耗时](图表/图1%20窗口固定后批量解码的耗时.png)

[查看图1数据（CSV）](图表/图1%20窗口固定后批量解码的耗时.csv)

![图2 新闻留出集的逐类精确率与召回率](图表/图2%20新闻留出集的逐类精确率与召回率.png)

[查看图2数据（CSV）](图表/图2%20新闻留出集的逐类精确率与召回率.csv)

## 复算

在仓库根目录运行。脚本用 Python 标准库生成同名 CSV，可与上方数据对照。

```shell
python "报告/L2-P68 编程练习题/复算图表数据.py"
```

实验程序与补充材料　[batch_decode.py](../../task6_bert/batch_decode.py) · [train_punct.py](../../task6_bert/train_punct.py) · [完整复现条件](../../docs/复现指南.md)
