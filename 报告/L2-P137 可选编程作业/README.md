# L2-P137 可选编程作业

词向量与新闻分类　·　L2 Part 1 第 137、171 页，Part 2 第 3 页

六类新闻上，固定分类器和五折划分，用 fastText 替换 Word2Vec 后，逻辑回归和 SVM 的平均宏 F1 分别提高 3.95 和 3.56 个百分点。

[下载 Word 报告](L2-P137%20可选编程作业.docx)　·　[课程作业索引](../../README.md)

## 扩展报告核验

补充核验将两个向量语料分开比较。六类线性 SVM 上，按宏 F1 计算，FastText 相对 Word2Vec 在百科语料中提高 3.56 个百分点，换作跨月新闻语料则出现反向差值。报告列出二、四、六类的配对结果、分词到分类的源码入口，并解释弱标签和表外词过滤如何限制结论。

## 核验结果

筛选 classes=6、clf=SVM 两行。w2v 与 ft 的五折平均 F1 分别为 0.7198 和 0.7554，差为 3.56 个百分点。

原始记录　[classification_v3.json](../../task4_gensim/results/classification_v3.json) · [official_fasttext.json](../../task4_gensim/results/official_fasttext.json)

![图1 在同一分类器上更换词向量](图表/图1%20在同一分类器上更换词向量.png)

[查看图1数据（CSV）](图表/图1%20在同一分类器上更换词向量.csv)

逐项数据　[交叉验证汇总](图表/交叉验证汇总.csv)

## 复算

在仓库根目录运行。脚本用 Python 标准库生成同名 CSV，可与上方数据对照。

```shell
python "报告/L2-P137 可选编程作业/复算图表数据.py"
```

实验程序与补充材料　[run_gensim.py](../../task4_gensim/run_gensim.py) · [official_fasttext.py](../../task4_gensim/official_fasttext.py) · [完整复现条件](../../docs/复现指南.md)
