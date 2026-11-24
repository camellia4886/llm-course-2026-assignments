"""Join frozen anonymous reviews to conditions and compute paired score changes."""
import json
import re
from pathlib import Path
from statistics import mean

BASE = Path(__file__).parent / "results" / "v2"
ITEMS = ("relation", "imagery", "fidelity", "format")


def read(file):
    return json.loads((BASE / file).read_text(encoding="utf-8"))


def main():
    protocol = read("locked_protocol.json")
    poem_info = {poem["id"]: poem for poem in protocol["poems"]}
    raw = read("raw.json")
    review = read("review_scores.json")
    scores = review["scores"]
    if set(scores) != {row["id"] for row in raw}:
        raise ValueError("Review IDs must match raw candidates exactly")
    for row in raw:
        row.update(scores[row["id"]])
        lines = [line for line in row["output"].splitlines() if line.strip()]
        content_only = not re.search(r"(?i)(here is|here's|note:|this poem|[\u3400-\u9fff])", row["output"])
        row["format"] = int(content_only and len(lines) == len(poem_info[row["poem"]]["lines"]))
    baseline = {(row["poem"], row["seed"]): row for row in raw if row["version"] == "V0"}
    summary = {}
    for split in ("development", "holdout"):
        versions = {}
        for version in protocol["versions"]:
            rows = [row for row in raw if row["split"] == split and row["version"] == version]
            matched = [(row, baseline[row["poem"], row["seed"]]) for row in rows]
            versions[version] = {
                "candidates": len(rows),
                "passed": {item: sum(row[item] for row in rows) for item in ITEMS},
                "errors": sum(not row[item] for row in rows for item in ITEMS),
                "fixed_vs_V0": sum(row[item] > base[item] for row, base in matched for item in ITEMS),
                "new_vs_V0": sum(row[item] < base[item] for row, base in matched for item in ITEMS),
                "faithful": mean(row["faithful"] for row in rows),
                "free": mean(row["free"] for row in rows),
                "input_tokens": sum(row["input_tokens"] for row in rows),
                "output_tokens": sum(row["output_tokens"] for row in rows),
                "seconds": round(sum(row["seconds"] for row in rows), 3),
                "length_caps": sum(row["hit_length_cap"] for row in rows),
            }
        summary[split] = versions
    rules = read("prompt_design_cases.json")
    by_condition = {}
    for role in (False, True):
        for standard in (False, True):
            rows = [row for row in rules if row["role"] == role and row["standard"] == standard]
            by_condition[f"role={int(role)},standard={int(standard)}"] = {
                "correct": sum(row["correct"] for row in rows), "cases": len(rows),
                "input_tokens": sum(row["input_tokens"] for row in rows),
                "output_tokens": sum(row["output_tokens"] for row in rows),
                "seconds": round(sum(row["seconds"] for row in rows), 3),
            }
    output = {"protocol_sha256": protocol["sha256"], "poetry": summary,
              "prompt_design": by_condition, "review_method": review["review_method"], "rows": raw}
    (BASE / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"poetry": summary, "rules": by_condition}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
