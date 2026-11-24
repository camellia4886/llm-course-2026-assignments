# -*- coding: utf-8 -*-
"""商业敏感信息识别的最小可运行演示（识别层 + 执行层分离）。

对应报告《商业敏感信息识别-研究路径.md》§5.1 管线与 §5.4 样本对：
- 识别层：纯规则引擎（正则 + 关键词），输出 类别 + 证据位置；
- 执行层：类别 × 接收者授权 × 政策矩阵 → 放行 / 复核 / 阻断，附完整依据链；
- 跑关键条件变化样本对（A/B）与控制样本对（C/D），检验两层各自是否在场。

诚实边界：这是演示管线设计的"最小实现"，规则集与授权矩阵为教学构造，
不是可上线的 DLP 系统；结论只说明设计逻辑，不代表生产系统表现。
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------- 识别层：规则引擎 ----------

PATTERNS = {
    "报价/价格": [
        (re.compile(r"\d+(?:\.\d+)?\s*(?:万|亿)?\s*(?:元|美元|欧元|人民币)"), "金额"),
        (re.compile(r"报价|单价|价格|折扣|成交价"), "报价词"),
    ],
    "客户名单": [
        (re.compile(r"1[3-9]\d{9}"), "手机号"),
        (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "邮箱"),
        (re.compile(r"客户名单|通讯录|联系方式"), "名单词"),
    ],
    "研发资料": [
        (re.compile(r"源代码|配方|原型|算法|未发布|专利草稿|研发文档"), "研发词"),
    ],
    "合同条款": [
        # 引用模式排除：'按/依照/根据…合同' 是引用合同依据而非发送合同文本
        (re.compile(r"(?:按|依照|根据|按照)合同(?:条款)?"), "引用模式-不命中"),
        (re.compile(r"合同|条款|违约金|保密协议|付款条件"), "合同词"),
    ],
    "财务数据": [
        (re.compile(r"财报|营收|利润|预算|现金流"), "财务词"),
    ],
    "人事信息": [
        (re.compile(r"\d{17}[\dXx]"), "身份证号"),
        (re.compile(r"工资|薪酬|绩效|考评"), "人事词"),
    ],
}

# 允许放行标记（出现在普通语境时不应命中，演示关键词误报用）
FALSE_POSITIVE_HINTS = ["新闻", "报道", "通知中提到", "文章里"]


def recognize(text: str):
    """识别层：返回 (命中的类别列表, 证据片段列表)。"""
    hits = []
    for cat, rules in PATTERNS.items():
        for rx, name in rules:
            if name == "引用模式-不命中":
                continue  # 该模式只用于从下方命中中剔除引用片段
            for m in rx.finditer(text):
                hits.append({"category": cat, "rule": name,
                             "evidence": m.group(0), "span": m.span()})
    # 剔除引用模式覆盖的合同词命中（"按合同条款发送"≠发送合同文本）
    ref_rx = PATTERNS["合同条款"][0][0]
    hits = [h for h in hits
            if not (h["category"] == "合同条款" and
                    any(rm.span()[0] <= h["span"][0] < rm.span()[1]
                        for rm in ref_rx.finditer(text)))]
    return hits


# ---------- 执行层：授权矩阵 ----------

# 处置：allow=放行, review=人工复核, block=阻断
# 行=接收者，列=敏感类别
POLICY = {
    #                    报价  客户名单  研发  合同  财务  人事
    "项目组内": {"报价/价格": "allow", "客户名单": "allow", "研发资料": "allow",
                 "合同条款": "allow", "财务数据": "allow", "人事信息": "allow"},
    "组外员工": {"报价/价格": "allow", "客户名单": "review", "研发资料": "review",
                 "合同条款": "review", "财务数据": "block", "人事信息": "block"},
    "合作方":   {"报价/价格": "allow", "客户名单": "block", "研发资料": "block",
                 "合同条款": "review", "财务数据": "block", "人事信息": "block"},
    "外部":     {"报价/价格": "block", "客户名单": "block", "研发资料": "block",
                 "合同条款": "block", "财务数据": "block", "人事信息": "block"},
}

POLICY_TEXT = {
    "项目组内": "组织政策 P1：项目组内成员对全部项目数据有访问授权",
    "组外员工": "组织政策 P2：组外员工仅对报价有访问授权，财务/人事禁止",
    "合作方": "组织政策 P3：合作方仅限合同约定的报价信息",
    "外部": "组织政策 P4：外部联系人无任何业务数据访问授权",
}


def execute(hits, receiver: str):
    """执行层：按授权矩阵给出处置与依据链。"""
    if not hits:
        return {"decision": "allow",
                "basis": [f"识别层无命中 → 按 {POLICY_TEXT[receiver]} 放行",
                          "依据链：内容不含敏感类别（无类别、无证据）"]}
    decisions = {POLICY[receiver].get(h["category"], "review") for h in hits}
    # 取最严格处置：block > review > allow
    decision = "block" if "block" in decisions else (
        "review" if "review" in decisions else "allow")
    basis = [f"识别层命中 {h['category']}（证据：“{h['evidence']}”，位置 {h['span']}）"
             for h in hits]
    basis.append(f"接收者={receiver} → {POLICY_TEXT[receiver]}")
    basis.append("依据链：类别 → 证据位置 → 接收者授权 → 政策条款")
    return {"decision": decision, "basis": basis}


# ---------- 样本集（报告 §5.4 + 补充） ----------

SAMPLES = [
    # 关键条件变化样本对：同一内容、不同接收者
    {"id": "A", "text": "本轮报价为 100 万元。", "receiver": "项目组内",
     "expect": "allow", "note": "关键样本对：内容相同、接收者不同"},
    {"id": "B", "text": "本轮报价为 100 万元。", "receiver": "外部",
     "expect": "block", "note": "关键样本对：内容相同、接收者不同"},
    # 控制样本对：情境不同、结论应相同
    {"id": "C", "text": "下周例会改到周三下午。", "receiver": "项目组内",
     "expect": "allow", "note": "控制样本对：无敏感内容"},
    {"id": "D", "text": "下周例会改到周三下午。", "receiver": "外部",
     "expect": "allow", "note": "控制样本对：无敏感内容"},
    # 补充样本
    {"id": "E", "text": "新闻里报道某公司工资薪酬情况，与我们无关。",
     "receiver": "外部", "expect": "allow",
     "note": "关键词误报演示：无我方敏感内容，但'工资/薪酬'关键词被规则命中"},
    {"id": "F", "text": "客户联系手机 13800138000，邮箱 a@example.com。",
     "receiver": "组外员工", "expect": "review", "note": "客户名单 → 组外员工复核"},
    {"id": "G", "text": "本季度营收 2.3 亿元，员工绩效表见附件。",
     "receiver": "组外员工", "expect": "block", "note": "财务+人事 → 组外员工阻断"},
    {"id": "H", "text": "报价单 45 万元已按合同条款发送。", "receiver": "合作方",
     "expect": "allow", "note": "合作方对报价有授权，合同词触发复核"},
]


def run():
    results = []
    for s in SAMPLES:
        hits = recognize(s["text"])
        out = execute(hits, s["receiver"])
        row = {"id": s["id"], "text": s["text"], "receiver": s["receiver"],
               "note": s["note"], "expect": s["expect"],
               "recognized": hits, **out,
               "match_expect": (s["expect"] is None or out["decision"] == s["expect"])}
        results.append(row)
        print(f"[{s['id']}] {s['text']}  → {s['receiver']}  →  {out['decision']}"
              f"  (预期 {s['expect'] or '—'})  {'OK' if row['match_expect'] else 'MISMATCH'}")

    # 关键对 / 控制对判定（§5.4 的判断方法）
    a = next(r for r in results if r["id"] == "A")
    b = next(r for r in results if r["id"] == "B")
    c = next(r for r in results if r["id"] == "C")
    d = next(r for r in results if r["id"] == "D")
    checks = {
        "关键对_A/B_处置不同": a["decision"] != b["decision"],
        "关键对_执行层在场": a["decision"] == "allow" and b["decision"] == "block",
        "控制对_C/D_均放行": c["decision"] == "allow" and d["decision"] == "allow",
        "控制对_渠道未被误当内容": d["decision"] == "allow",
        "误报演示_E_被规则命中": any(h["category"] == "人事信息" for h in
                              next(r for r in results if r["id"] == "E")["recognized"]),
        "引用排除_H_合作方放行": next(r for r in results if r["id"] == "H")["decision"] == "allow",
    }
    print("\n判定：")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "mini_demo.json"
    metrics = {
        "false_allow": sum(row["decision"] == "allow" and row["expect"] != "allow" for row in results),
        "false_block_or_review": sum(row["decision"] != "allow" and row["expect"] == "allow" for row in results),
        "expected_allow": sum(row["expect"] == "allow" for row in results),
        "expected_block_or_review": sum(row["expect"] != "allow" for row in results),
    }
    out_path.write_text(json.dumps(
        {"samples": results, "checks": checks, "policy": POLICY,
         "policy_text": POLICY_TEXT, "decision_metrics": metrics},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out_path}")


if __name__ == "__main__":
    run()
