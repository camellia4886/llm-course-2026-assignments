# L1-P122 思考题、P125 编程作业

人民日报 n-gram 与剪枝　·　L1 第 109、122、125 页

三元组频次门槛从 1 提至 2，完整用时从 11.5 降至 5.1 秒，六月困惑度从 373.5 升至 892.6。

[下载 Word 报告](L1-P122%20思考题、P125%20编程作业.docx)　·　[课程作业索引](../../README.md)

## 核验结果

对照 min_freq 为 1 和 2 的两行，保留类型从 3,906,388 降至 603,179。train_s 只计三元计数和模型估计；完整运行时间取 total_s。六月困惑度按同一测试口径计算。

原始记录　[ppl_trigram_prune.json](../../task1_ngram/results/ppl_trigram_prune.json) · [kenlm.json](../../task1_ngram/results/kenlm.json) · [nll_check.json](../../course_labs/results/nll_check.json)

![图1 剪枝后的库存、耗时与困惑度](图表/图1%20剪枝后的库存、耗时与困惑度.png)

[查看图1数据（CSV）](图表/图1%20剪枝后的库存、耗时与困惑度.csv)

![图2 累计月份与单位输入的存储成本](图表/图2%20累计月份与单位输入的存储成本.png)

[查看图2数据（CSV）](图表/图2%20累计月份与单位输入的存储成本.csv)

## 复算

在仓库根目录运行。脚本用 Python 标准库生成同名 CSV，可与上方数据对照。

```shell
python "报告/L1-P122 思考题、P125 编程作业/复算图表数据.py"
```

实验程序与补充材料　[ppl_extra.py](../../task1_ngram/ppl_extra.py) · [kenlm_experiment.py](../../task1_ngram/kenlm_experiment.py) · [思考题合集.md](../../思考题合集.md) · [完整复现条件](../../docs/复现指南.md)
