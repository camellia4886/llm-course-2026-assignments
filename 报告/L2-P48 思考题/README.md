# L2-P48 思考题

分段检索与阅读预算　·　L2 Part 1 第 48、87、104 页，Part 2 第 231–264 页

限制前三块时词汇切分比定长少命中 6 道中文题和 51 道英文题；改为等字符预算后，分别多命中 3 题和 8 题。

[下载 Word 报告](L2-P48%20思考题.docx)　·　[课程作业索引](../../README.md)

## 核验结果

查 languages.zh.retrieval。954 个问题在“前三块”下定长命中 923 题、词汇切分命中 917 题；按相同字符预算，两者分别命中 915、918 题。

原始记录　[segmentation_benchmark.json](../../course_labs/results/segmentation_benchmark.json)

![图1 词汇切分的优势随阅读口径反转](图表/图1%20词汇切分的优势随阅读口径反转.png)

[查看图1数据（CSV）](图表/图1%20词汇切分的优势随阅读口径反转.csv)

逐项数据　[命中题数](图表/命中题数.csv)

## 复算

在仓库根目录运行。脚本用 Python 标准库生成同名 CSV，可与上方数据对照。

```shell
python "报告/L2-P48 思考题/复算图表数据.py"
```

实验程序与补充材料　[segmentation_benchmark.py](../../course_labs/segmentation_benchmark.py) · [动手实验报告.md](../../course_labs/动手实验报告.md) · [完整复现条件](../../docs/复现指南.md)
