"""任务16：唐诗宋词英译提示对比实验 —— 本地小模型真实运行（2026-11-21）

对应 task16_prompteng/唐诗宋词英译提示对比实验.md §5 待运行实验清单：
- 同一模型（Qwen2.5-1.5B-Instruct，本地 HF 缓存）、同一配置（temperature/top_p/max_new_tokens 全部一致）
- 两首诗（静夜思、赠别·其二）× 四版提示（V0 最简 / V1 任务标准 / V2 参考资料 / V3 边界示例）× 3 个 seed
- 泛化测试：王维《相思》（未参与提示设计的材料）× 四版提示 × 1 个 seed
- 输出匿名随机编号（results/anonymized.json），评分者只读匿名文件盲评
  （results/review_scores.json），映射表（results/mapping.json）评分完成后才合并。

生成配置：max_new_tokens=96, do_sample=True, temperature=1.0, top_p=0.95。
模型与参考资料（已核实来源，见报告文档"来源"节）：
- 静夜思 V2 参考：许渊冲版 + Herbert Giles 版（FOB 论坛 9 译本页）
- 赠别 V2 参考：Wang Yongli 版（新浪/腾讯 2026-07 译辑）+ Bynner 首行
- 相思 V2 参考：许渊冲版（《许渊冲译王维诗选》2021，第 68 页）
"""
import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
# 模型目录：优先环境变量 QWEN_MODEL_DIR（本地目录），否则从 HF 缓存加载
MODEL_PATH = os.environ.get("QWEN_MODEL_DIR", "Qwen/Qwen2.5-1.5B-Instruct")
LOAD_KW = {} if Path(MODEL_PATH).exists() else {"local_files_only": True}
MAX_NEW, TEMPERATURE, TOP_P = 96, 1.0, 0.95
SEEDS = [20261121, 20261122, 20261123]

POEMS = {
    "静夜思": "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "赠别": "多情却似总无情，唯觉樽前笑不成。蜡烛有心还惜别，替人垂泪到天明。",
    "相思": "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
}

REFS = {
    "静夜思": ("许渊冲：Abed, I see a silver light, / I wonder if it's frost aground. / "
              "Looking up, I find the moon bright; / Bowing, in homesickness I'm drowned.\n"
              "Herbert Giles：I wake, and moonbeams play around my bed, / Glittering like "
              "hoar-frost to my wandering eyes; / Up towards the glorious moon I raise my head, / "
              "Then lay me down---and thoughts of home arise."),
    "赠别": ("Wang Yongli：My heart brims o'er with love, yet cold I seem, / Before the cup "
            "no smile can light my gleam. / Candles hold hearts that grieve our parting hour; / "
            "They shed long tears till dawn's first light doth flower.\n"
            "Bynner 首行：How can a deep love seem a deep love? / How can it smile, at a "
            "farewell feast?"),
    "相思": ("许渊冲：The red beans grow in southern land. / How many load in spring the trees? / "
            "Gather them till full is your hand, / They would revive fond memories."),
}

TERMS = {
    "静夜思": "明月→the bright moon；故乡→home/homeland；霜→hoarfrost",
    "赠别": "樽→cup/wine cup；蜡烛→candle；泪→tears",
    "相思": "红豆→red beans；相思→longing/love",
}

STANDARDS = {
    "静夜思": ("要求：1) 保留原诗的意象（月光、霜、故乡）；2) 保持行数与句间关系（"
              "'疑是'的疑问关系不得改为陈述）；3) 尽量押尾韵并控制每行长度；"
              "4) 不添加原诗没有的信息；5) 术语译法统一。"),
    "赠别": ("要求：1) 保留原诗的意象（多情、酒杯、蜡烛、泪）；2) 保持行数与句间关系（"
            "'多情却似总无情'的转折关系、'笑不成'的否定不得改变）；3) 尽量押尾韵并控制每行长度；"
            "4) 不添加原诗没有的信息；5) 术语译法统一。"),
    "相思": ("要求：1) 保留原诗的意象（红豆、春枝）；2) 保持行数与句间关系（'发几枝'的疑问、"
            "'此物最相思'的因果）；3) 尽量押尾韵并控制每行长度；4) 不添加原诗没有的信息；"
            "5) 术语译法统一。"),
}

BAD_EXAMPLES = {
    "静夜思": "坏例：It is hoarfrost on the ground.——把'疑是'的疑问改成了陈述，属于疑问关系错误，请避免。",
    "赠别": "坏例：She is cold and unfeeling to me.——把'多情却似总无情'译成真的无情，丢失了转折关系，请避免。",
    "相思": "坏例：I love you very much.——把含蓄的'相思'直白喊出，丢失原诗的含蓄，请避免。",
}


def build_prompt(poem, version):
    text = POEMS[poem]
    v0 = f"把下面的中文诗翻译成英文：{text}"
    if version == "V0":
        return v0
    if version == "V1":
        return v0 + "\n" + STANDARDS[poem]
    if version == "V2":
        return (v0 + "\n" + STANDARDS[poem] +
                f"\n参考以下译法风格（注意其处理方式，不必照抄）：\n{REFS[poem]}\n"
                f"术语表：{TERMS[poem]}")
    if version == "V3":
        return (v0 + "\n" + STANDARDS[poem] +
                f"\n好例（保持原文句间关系的示范）：{REFS[poem].splitlines()[0]}\n"
                f"{BAD_EXAMPLES[poem]}")
    raise ValueError(version)


def generate(model, tok, prompt, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                             temperature=TEMPERATURE, top_p=TOP_P,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    secs = time.time() - t0
    new_ids = out[0][enc["input_ids"].shape[1]:]
    text = tok.decode(new_ids, skip_special_tokens=True).strip()
    return text, {"prompt_tokens": int(enc["input_ids"].shape[1]),
                  "completion_tokens": int(new_ids.shape[0]),
                  "gen_secs": round(secs, 3)}


def main():
    print(f"loading {MODEL_PATH} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, **LOAD_KW)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", **LOAD_KW)
    model.eval()
    device = str(next(model.parameters()).device)

    rows = []          # {poem, version, seed, prompt, output, cost}
    for poem in POEMS:
        for version in ("V0", "V1", "V2", "V3"):
            n_seeds = SEEDS if poem != "相思" else SEEDS[:1]
            for seed in n_seeds:
                prompt = build_prompt(poem, version)
                output, cost = generate(model, tok, prompt, seed)
                rows.append({"poem": poem, "version": version, "seed": seed,
                             "prompt": prompt, "output": output, **cost})
                print(f"[{poem} {version} seed={seed}] {len(output)} chars, "
                      f"{cost['gen_secs']}s")

    # 匿名化：随机编号，评分者只读 anonymized.json（不含任何可推断版本的信息，
    # prompt_tokens 会泄露版本，故成本信息只存 mapping.json，盲评完成后才合并）
    rng = random.Random(20261121)
    ids = [f"C{i:02d}" for i in range(len(rows))]
    rng.shuffle(ids)
    mapping = {ids[i]: {**{k: rows[i][k] for k in ("poem", "version", "seed")},
                        "cost": {k: rows[i][k] for k in ("prompt_tokens",
                                                         "completion_tokens", "gen_secs")}}
               for i in range(len(rows))}
    anonymized = [{"id": ids[i],
                   "poem_tag": rows[i]["poem"],      # 评分需要知道是哪首诗（诗名不含版本信息）
                   "text": rows[i]["output"]}
                  for i in range(len(rows))]
    (RESULTS / "mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "anonymized.json").write_text(
        json.dumps(anonymized, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngenerated {len(rows)} outputs, device={device}")
    print("匿名文件 results/anonymized.json 已生成，映射存 results/mapping.json（盲评完成前勿看）")
    print("请评分者只读 anonymized.json，把评分写入 results/review_scores.json")


if __name__ == "__main__":
    main()
