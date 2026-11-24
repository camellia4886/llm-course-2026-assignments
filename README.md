# 大语言模型与应用实践课程作业

2026 年秋季课程 · 本地提交版 · 2026-11-24 更新。页码使用 PPT 的实际页序。报告引用新版结果时使用带 v2 的文件名，早期 JSON 保留供比较。课程作业根目录之外的同名文件是旧工作目录。

## 交付索引

| 课件 | 交付 | 结果状态 |
|---|---|---|
| L1 第 109、122、125 页 | [NLL 与 n-gram 思考](思考题合集.md) · [人民日报 n-gram 训练](task1_ngram/报告.md) | NLL 三种写法数值校验，Python 与原生 KenLM 的 1/3/6 月训练、独立月份 PPL 与续写 |
| L1 第 22、24、164–191、295 页 | [语言单位、项目案例、证据与 LAS 作答](course_labs/课堂习题补全.md) | 书面作答 |
| L1 第 167–169、192 页 | [商业敏感信息](task8_sensitive/商业敏感信息识别-研究路径.md) · [熟悉项目重做](L1可选-把熟悉项目重做成可检查任务.md) | 教学示例与对照实测 |
| L2 Part 1 第 13、48、87、104、147、230、256、276 页 | [课堂作答](course_labs/课堂习题补全.md) · [Datasets、Tokenizers 与分段动手练习](course_labs/动手实验报告.md) | 第 5、6 章公开测验、FAISS 和中英文自然段落与证据检索，账号进度待核实 |
| L2 Part 1 第 137、171 页，Part 2 第 3 页 | [Gensim 与原生 FastText](task4_gensim/报告.md) | CMRC 独立语料表示学习、12 组 9801 分类及原生 FastText 实验 |
| L2 Part 2 第 68 页 | [BERT 微调与批量解码](task6_bert/报告.md) | 三月留出集测试、回归样例、速度及质量实测 |
| L2 Part 2 第 90–91 页 | [设计下一代 BERT](course_labs/下一代BERT设计.md) | 参数账本、训练目标、三项消融设计与小模型实测 |
| L2 Part 2 第 94–95 页 | [nanoGPT 研读与小型重建](course_labs/动手实验报告.md) | 源码逐函数对照、CPU 训练和因果性验证 |
| L2 Part 2 第 115、123、128、140、157、173、223–226 页 | [资源与论文习题](course_labs/课堂习题补全.md) · [去重及决策模型思考](思考题合集.md) · [一页资源账本](task2_ledger/一页资源账本.pdf) | 算例、论证与可证伪对照 |
| L2 Part 2 第 231–264 页 | [古汉语词表适配](course_labs/动手实验报告.md) | 分词接口实测与权重迁移方案 |
| L3 Part 1 第 15、19、43、52、54、66、87、89–90 页 | [安全与权限验证](course_labs/提示词安全与权限验证.md) · [课堂小题](course_labs/课堂习题补全.md) · [提示经验实验](task16_prompteng/提示经验对照.md) · [唐诗宋词对照](task16_prompteng/唐诗宋词英译提示对比实验.md) | 本地 Qwen 权限建议与应用授权核查、提示规则与英译实测、单模型辅助盲评 |
| L3 Part 2 第 57、79–80、123–126 页 | [在线学习与偏好优化思考](思考题合集.md) · [SFT 掩码检查](course_labs/动手实验报告.md) | 书面作答与实际逐 token 检查 |
| L3 Part 2 第 224、226 页 | [训练配方拆解](task3_survey/训练配方拆解.md) · [五页研究报告](task3_survey/后训练综述.pdf) · [源码追踪](task3_survey/开源代码追踪.md) · [汇报讲稿](task3_survey/10分钟课堂汇报.md) | 教学型 SFT/GRPO 小实验、10 分钟逐页讲稿和备答就绪 |
| L3 Part 2 第 227–237 页 | [领域模型实验](task17_domain/报告.md) · [两道综合思考](course_labs/课堂习题补全.md) | 独立篇章、约束变式和现代回归均有结果 |

## 本地复现

当前使用 Windows Python 3.12、PyTorch 2.11.0 CUDA 12.8、Transformers 5.23.0、Gensim 4.4.0。现有虚拟环境在本目录上一级。在下面的命令中，`python` 为已激活环境的解释器，PowerShell 也可用 `& '../.venv/Scripts/python.exe' -X utf8 -B` 替换。主要依赖见 [requirements.txt](requirements.txt)。原生 FastText 另装 `fasttext-wheel==0.9.2`，生成 PDF 需 ReportLab 与微软雅黑字体。

```text
python task1_ngram/clean_corpus.py
python task1_ngram/run_experiments.py
python task1_ngram/ppl_extra.py
python task1_ngram/smoke.py
python task1_ngram/kenlm_experiment.py --lmplz .kenlm-course-cache/portable/bin/lmplz.exe
python course_labs/nll_check.py
python task2_ledger/calc.py
python task4_gensim/run_gensim.py
python task4_gensim/run_gensim.py --cmrc-train data/public_benchmarks/cmrc2018_train.json
python task4_gensim/official_fasttext.py
python task6_bert/build_data.py
python task6_bert/train_punct.py
python task6_bert/reeval.py
python task6_bert/batch_decode.py
python task17_domain/run_domain_sft.py
python task17_domain/error_examples.py
python l1_optional_checkable/guwen_vs_baihua.py
python task8_sensitive/run_mini_demo.py
python course_labs/data_and_tokenizers.py
python course_labs/faiss_search.py
python course_labs/segment_text.py
python course_labs/segmentation_benchmark.py
python course_labs/nanogpt_exercise.py
python course_labs/next_bert_probe.py
python course_labs/sft_loss_check.py
python course_labs/safety_boundary_check.py
python course_labs/safety_boundary_model.py
python task3_survey/run_tiny_posttrain.py
python "task3_survey/生成课程PDF.py"
python task16_prompteng/run_controlled_v2.py
python task16_prompteng/summarize_v2.py
```

- `clean_corpus.py` 默认使用本工作区已解压的原始语料。其他位置用 `--source-dir` 指定六个原件。直接使用已清理的月度文件可以跳过该步。
- KenLM 使用本机已解包的 Windows x64 工具。其他环境安装 `lmplz` 和 `query` 后，按实际路径传 `--lmplz`，可另用 `--query` 指定评分程序。FAISS 补实验使用 `faiss-cpu==1.13.2` 和同一份本地 BERT 缓存。
- CMRC、DRCD 和 SQuAD 的下载位置与 SHA-256 见 [公开评测数据说明](data/public_benchmarks/README.md)。
- BERT 需要本机缓存的 `bert-base-chinese`，可设 `BERT_BASE_MODEL` 指向本地检查点。任务 17 和 L1 熟悉项目依赖任务 7 新权重。发布文件时按 .gitignore 排除大权重，本目录保留本轮本机权重。
- Qwen 项目默认读取上一级的 `models/qwen2.5-1.5b-instruct`。有独立模型目录时设置 `QWEN_MODEL_DIR`。新版匿名候选、复核评分和合并结果位于 `task16_prompteng/results/v2/`。换权重或采样配置时应重做匿名评分。
- nanoGPT 为按源码设计重建的教学模型，后训练小实验为单 token 教学代理。下一代 BERT 和参数层面的古文扩表提交验证设计和适用的小实验。
- Piazza 提交按要求暂缓。课件要求课堂现场展示运行材料、过程和判断依据，汇报可直接使用五页研究报告与逐页讲稿。
