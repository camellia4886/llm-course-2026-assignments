"""合并盲评结果：anonymized.json + review_scores.json + mapping.json -> prompt_experiment.json

review_scores.json 格式（评分者只读 anonymized.json 后产出）：
  {"C01": {"checklist": {"relation": 1, "imagery": 1, "terminology": 1,
                          "structure": 1, "fidelity": 1, "format": 0},
           "faithful": 4, "free": 4, "notes": "..."}}
checklist 各项 1=通过 0=未通过：
  relation   否定/疑问/转折关系保持
  imagery    意象保留（替换/丢失记 0）
  terminology 术语一致性（同一输出内译法一致）
  structure  行数/分句对应
  fidelity   无增译/漏译
  format     押韵/标点/大小写符合约定
本脚本不做任何评分判断，只做合并与按版本聚合。
"""
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
R = BASE / "results"
ITEMS = ["relation", "imagery", "terminology", "structure", "fidelity", "format"]


def main():
    mapping = json.loads((R / "mapping.json").read_text(encoding="utf-8"))
    anon = {r["id"]: r for r in json.loads((R / "anonymized.json").read_text(encoding="utf-8"))}
    review = json.loads((R / "review_scores.json").read_text(encoding="utf-8"))

    # 合并：每个 ID -> {poem, version, seed, output, cost, review}
    merged = {}
    for cid, m in mapping.items():
        a = anon[cid]
        rv = review[cid]
        merged[cid] = {**m, "text": a["text"],
                       "checklist": rv["checklist"], "faithful": rv["faithful"],
                       "free": rv["free"], "notes": rv.get("notes", "")}

    out = {"model": "Qwen/Qwen2.5-1.5B-Instruct",
           "config": {"max_new_tokens": 96, "temperature": 1.0, "top_p": 0.95},
           "per_output": merged}

    # 按版本聚合（主实验两首诗 × 3 seed；泛化诗 1 seed）
    for poem in ("静夜思", "赠别", "相思"):
        n_seeds = 3 if poem != "相思" else 1
        agg = {}
        for v in ("V0", "V1", "V2", "V3"):
            rows = [r for r in merged.values()
                    if r["poem"] == poem and r["version"] == v]
            assert len(rows) == n_seeds, (poem, v, len(rows))
            items = {it: {"passed": sum(r["checklist"][it] for r in rows),
                          "total": n_seeds} for it in ITEMS}
            v0 = [r for r in merged.values()
                  if r["poem"] == poem and r["version"] == "V0"]
            fixed = added = 0
            for it in ITEMS:
                for rx, r0 in zip(rows, v0):
                    if rx["checklist"][it] and not r0["checklist"][it]:
                        fixed += 1
                    if not rx["checklist"][it] and r0["checklist"][it]:
                        added += 1
            agg[v] = {
                "items": items,
                "error_count": sum(n_seeds - items[it]["passed"] for it in ITEMS),
                "vs_V0": {"fixed": fixed, "new_errors": added},
                "cost": {"prompt_tokens": sum(r["cost"]["prompt_tokens"] for r in rows),
                         "completion_tokens": sum(r["cost"]["completion_tokens"] for r in rows),
                         "gen_secs": round(sum(r["cost"]["gen_secs"] for r in rows), 2)},
                "faithful_avg": round(sum(r["faithful"] for r in rows) / n_seeds, 2),
                "free_avg": round(sum(r["free"] for r in rows) / n_seeds, 2),
            }
        out[f"aggregate_{poem}"] = agg

    (R / "prompt_experiment.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved results/prompt_experiment.json")
    for poem in ("静夜思", "赠别"):
        print(f"\n== {poem} ==")
        for v in ("V0", "V1", "V2", "V3"):
            a = out[f"aggregate_{poem}"][v]
            print(f"{v}: 错误数={a['error_count']} 改对={a['vs_V0']['fixed']} "
                  f"新增错误={a['vs_V0']['new_errors']} "
                  f"忠实均分={a['faithful_avg']} 自由均分={a['free_avg']} "
                  f"token={a['cost']['prompt_tokens']}+{a['cost']['completion_tokens']}")


if __name__ == "__main__":
    main()
