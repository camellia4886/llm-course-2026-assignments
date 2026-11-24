# 独立训练语料与分段评测数据

2026-11-24 从各项目的发布地址取得原版文件。复现时将文件放在本目录，保留下列文件名。训练与结果报告附带源码和校验值，源数据可从各项目重新取得。

| 文件 | 本地用途 | SHA-256 |
|---|---|---|
| `cmrc2018_train.json` | 只读取正文，用于独立的中文词向量训练 | `5497aa2f81908e31d6b0e27d99b1f90ab63a8f58fa92fffe5d17cf62eba0c212` |
| `DRCD_dev.json` | 中文段落边界及证据检索 | `e236df03861ba241bd7b29628cd48b3b8e339c60137e041a4358b5320ed61928` |
| `squad_dev_v11.json` | 英文段落边界及证据检索 | `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9` |

- CMRC 2018 由 Cui 等发布。 [项目](https://github.com/ymcui/cmrc2018) · [训练数据](https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_train.json)
- DRCD 由台达团队发布，本文使用繁体中文开发集。 [项目](https://github.com/DRCKnowledgeTeam/DRCD) · [开发数据](https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/master/DRCD_dev.json)
- SQuAD 1.1 使用 Rajpurkar 等发布的开发集。 [项目](https://rajpurkar.github.io/SQuAD-explorer/) · [开发数据](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
