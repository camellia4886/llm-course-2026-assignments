"""按资源账本设定核算不同请求配置的 KV 缓存。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_data import write_tables


def tables():
    # task2_ledger/calc.py 的 GQA 配置，K 和 V 分开保存。
    layers, kv_heads, head_width, element_bytes = 32, 8, 128, 2
    bytes_per_position = 2 * layers * kv_heads * head_width * element_bytes
    rows = []
    for batch, seq_len in [(1, 8192), (2, 8192), (4, 32768)]:
        label = f"{batch} 路 × {seq_len // 1024}K"
        rows.append((label, batch * seq_len * bytes_per_position / 2**30))
    yield "图1 相同权重下的 KV 缓存", ["项目", "缓存"], rows


if __name__ == "__main__":
    write_tables(__file__, tables())
