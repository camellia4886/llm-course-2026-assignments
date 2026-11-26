# L2-P95 选择题（nanoGPT）

nanoGPT 的损失与计算量　·　L2 Part 2 第 94–95 页

120 步训练后，同一批六月评估窗口的平均损失从 7.857 降至 4.527 nat/位置，计分位置共 4,096 个。

[下载 Word 报告](L2-P95%20选择题（nanoGPT）.docx)　·　[课程作业索引](../../README.md)

## 扩展报告核验

扩展稿将 8×64 的输入、错位标签、四个注意力头与留出损失一一对应，并核对了困惑度的换算。六月文本中的 335 个表外字符和随机窗口中的 4,096 个计分位置分别使用各自分母。

## 核验结果

先查 heldout_before、heldout_after。六月份留出损失由 7.857 降至 4.527 nat／位置。history 的四个读数取自训练批，图 1 绘制的是后者。

原始记录　[nanogpt_exercise.json](../../course_labs/results/nanogpt_exercise.json)

![图1 训练批次损失的四个记录点](图表/图1%20训练批次损失的四个记录点.png)

[查看图1数据（CSV）](图表/图1%20训练批次损失的四个记录点.csv)

![图2 一个 Transformer 块的运算分布](图表/图2%20一个%20Transformer%20块的运算分布.png)

[查看图2数据（CSV）](图表/图2%20一个%20Transformer%20块的运算分布.csv)

## 复算

在仓库根目录运行。脚本用 Python 标准库生成同名 CSV，可与上方数据对照。

```shell
python "报告/L2-P95 选择题（nanoGPT）/复算图表数据.py"
```

实验程序与补充材料　[nanogpt_exercise.py](../../course_labs/nanogpt_exercise.py) · [完整复现条件](../../docs/复现指南.md)
