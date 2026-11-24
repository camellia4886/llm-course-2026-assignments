"""Generate the locked poetry runs and a separate prompt-design experiment."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).parent
OUT = BASE / "results" / "v2"
MODEL = os.environ.get("QWEN_MODEL_DIR", str(BASE.parents[1] / "models" / "qwen2.5-1.5b-instruct"))
CONFIG = {"temperature": 0.55, "top_p": 0.9, "max_new_tokens": 192, "do_sample": True}
SEEDS = (20261124, 20261125)
VERSIONS = ("V0", "V1", "V2", "V3")

POEMS = [
    {"id": "P1", "split": "development", "genre": "唐诗", "title": "静夜思", "author": "李白",
     "lines": ["床前明月光", "疑是地上霜", "举头望明月", "低头思故乡"]},
    {"id": "P2", "split": "development", "genre": "宋词", "title": "如梦令·常记溪亭日暮", "author": "李清照",
     "lines": ["常记溪亭日暮", "沉醉不知归路", "兴尽晚回舟", "误入藕花深处", "争渡，争渡", "惊起一滩鸥鹭"]},
    {"id": "P3", "split": "holdout", "genre": "唐诗", "title": "相思", "author": "王维",
     "lines": ["红豆生南国", "春来发几枝", "愿君多采撷", "此物最相思"]},
    {"id": "P4", "split": "holdout", "genre": "宋词", "title": "卜算子·我住长江头", "author": "李之仪",
     "lines": ["我住长江头", "君住长江尾", "日日思君不见君", "共饮长江水",
               "此水几时休", "此恨何时已", "只愿君心似我心", "定不负相思意"]},
]
STANDARDS = ("Output only the translation in English, with one line per source line. "
             "Retain the speaker, direction, negation, questions, and revisions of expectation. "
             "Keep the concrete images and meaningful repetition. Do not fabricate details. "
             "Use concise poetic English. Meaning takes priority over rhyme.")
REFERENCE = ("Reference note: In classical Chinese verse the speaker can be implicit. "
             "Resolve viewpoint only as the text supports it. Parallel clauses often preserve "
             "contrasts of place, time or emotion. Repetition can carry rhythm and force. "
             "English may name a subject only where the source licenses it.")
BOUNDARY = ("Boundary example, unrelated to the target: 虽归未见君 -> Though I have returned, "
            "I have not seen you. Avoid I returned and saw you, which reverses the negative. "
            "Example: 隔岸灯明 -> Across the bank the lamps are bright. "
            "Do not add an unseen village or a reason for the lamps.")

CASES = [
    ("R1", "已登记者无需再交证明；未登记者须在周三前交。申请人甲已登记。甲须交证明吗？", "NO"),
    ("R2", "除主管批准外，档案不得外借。甲未取得批准。甲能外借档案吗？", "NO"),
    ("R3", "当天值班者可从东门通行。甲当天值班。甲可从东门通行吗？", "YES"),
    ("R4", "申请必须在周五前提交，周五当天不算。甲周五当天提交。甲按期提交了吗？", "NO"),
    ("R5", "须同时持有效证件和预约单。甲持有效证件，是否有预约单没有说明。甲符合要求吗？", "UNKNOWN"),
    ("R6", "成员均可借阅。甲未说明是否为成员。甲一定可以借阅吗？", "UNKNOWN"),
]


def poetry_prompt(poem, version):
    base = "Translate the following classical Chinese verse into English.\n" + "\n".join(poem["lines"])
    if version == "V0":
        return base
    extra = {"V1": "", "V2": REFERENCE, "V3": BOUNDARY}[version]
    return "\n".join(part for part in (base, STANDARDS, extra) if part)


def rule_prompt(case, use_role, use_standard):
    role = ("你是一位负责逐条核对规则的档案审核员。" if use_role
            else "请你完成下面这道关于档案规则的题目。")
    standard = ("只依据材料，检查否定、例外、期限与必要条件，缺少条件就选 UNKNOWN。" if use_standard
                else "先读完这一段材料，然后根据自己的判断选择你认为正确的选项。")
    return f"{role}\n{standard}\n{case}\n只输出 YES、NO 或 UNKNOWN 一个词。"


def save(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@torch.inference_mode()
def generate(model, tokenizer, prompt, seed, config):
    torch.manual_seed(seed)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    started = time.perf_counter()
    generated = model.generate(**encoded, pad_token_id=tokenizer.eos_token_id, **config)
    elapsed = time.perf_counter() - started
    prefix = encoded["input_ids"].shape[-1]
    new = generated[0, prefix:]
    return tokenizer.decode(new, skip_special_tokens=True).strip(), {
        "input_tokens": prefix, "output_tokens": len(new), "seconds": round(elapsed, 4),
        "hit_length_cap": len(new) == config["max_new_tokens"],
    }


def poetry_runs(model, tokenizer):
    protocol = {"versions": VERSIONS, "seeds": SEEDS, "generation": CONFIG, "poems": POEMS,
                "standards": STANDARDS, "reference": REFERENCE, "boundary": BOUNDARY}
    encoded = json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()
    protocol["sha256"] = hashlib.sha256(encoded).hexdigest()
    save("locked_protocol.json", protocol)
    raw = []
    for poem in POEMS:
        for version in VERSIONS:
            for seed in SEEDS:
                prompt = poetry_prompt(poem, version)
                text, cost = generate(model, tokenizer, prompt, seed, CONFIG)
                raw.append({"poem": poem["id"], "split": poem["split"], "version": version,
                            "seed": seed, "prompt": prompt, "output": text, **cost})
        print("poetry completed", poem["id"], flush=True)
    order = list(range(len(raw)))
    random.Random(601).shuffle(order)
    rows = []
    for index, raw_index in enumerate(order, 1):
        row = raw[raw_index]
        rows.append({"id": f"B{index:02}", "poem": row["poem"], "output": row["output"]})
        row["id"] = f"B{index:02}"
    save("anonymized.json", rows)
    save("raw.json", raw)


def rules_runs(model, tokenizer):
    config = {**CONFIG, "max_new_tokens": 12}
    rows = []
    for case_id, case, gold in CASES:
        for role in (False, True):
            for standard in (False, True):
                prompt = rule_prompt(case, role, standard)
                output, cost = generate(model, tokenizer, prompt, SEEDS[0], config)
                normalized = output.strip().upper().rstrip("。.")
                rows.append({"case": case_id, "role": role, "standard": standard,
                             "gold": gold, "raw_output": output, "correct": normalized == gold,
                             "prompt": prompt, **cost})
    save("prompt_design_cases.json", rows)
    print("rule experiment completed", sum(row["correct"] for row in rows), "/", len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("poetry", "rules"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, local_files_only=True).to(device).eval()
    save("environment.json", {"model": Path(MODEL).name, "device": device, "dtype": str(dtype),
                              "torch_version": str(torch.__version__),
                              "gpu": torch.cuda.get_device_name() if device == "cuda" else None})
    if args.only in (None, "poetry"):
        poetry_runs(model, tokenizer)
    if args.only in (None, "rules"):
        rules_runs(model, tokenizer)


if __name__ == "__main__":
    main()
