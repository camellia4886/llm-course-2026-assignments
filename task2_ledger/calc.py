# L2 练习与验收：资源账本计算（与 slide 226 答案交叉验证）
GB = 10**9
GiB = 2**30

L, d, h, h_kv, dh, b_kv = 32, 4096, 32, 8, 128, 2
V, N, D = 32000, 7e9, 1.4e11

# 1. 嵌入参数
emb = V * d
print(f"嵌入参数(共享): {emb:,} = {emb/1e6:.1f}M, 字节(BF16) {emb*2/GB:.3f} GB")
print(f"嵌入参数(不共享x2): {emb*2:,} = {emb*2/1e6:.1f}M, 字节 {emb*4/GB:.3f} GB")
print(f"占N比例: {emb/N*100:.2f}%")

# 块参数交叉检查
N_block = 12 * L * d**2
print(f"\n块参数 12Ld^2 = {N_block:,} = {N_block/1e9:.3f}B;  +嵌入 = {N_block+emb:,} = {(N_block+emb)/1e9:.3f}B")

# 2. 训练 FLOPs
C = 6 * N * D
print(f"\n训练FLOPs 6ND = {C:.4e} = {C/1e21:.2f}e21")
# 长上下文修正: F = 6N_block D + 12LDsd (完整方阵口径)
C2 = 6 * N_block * D + 12 * L * D * 32768 * d
print(f"修正(含注意力, s=32768): 6N_block·D={6*N_block*D:.3e}, 12LDsd={12*L*D*32768*d:.3e}, 合计={C2:.4e}")
print(f"因果内核配对减半后: {6*N_block*D + 6*L*D*32768*d:.4e}")

# 3. 权重字节
print(f"\n权重字节: N*2 = {N*2/GB:.3f} GB (十进制) = {N*2/GiB:.3f} GiB")

# 4. KV 缓存: 2 * L * h_kv * dh * B * s * b
def kv(B, s, heads=h_kv):
    return 2 * L * heads * dh * B * s * b_kv
for B, s in [(1, 8192), (1, 32768), (4, 32768)]:
    k = kv(B, s)
    print(f"KV B={B} s={s}: {k/GiB:.3f} GiB = {k/GB:.3f} GB")
k_mha = kv(1, 8192, heads=h)
print(f"对比 MHA(h=32) B=1 s=8192: {k_mha/GiB:.3f} GiB (GQA 的 {h/h_kv} 倍)")
print(f"\n主项(权重+KVs8192): {N*2/GB + kv(1,8192)/GB:.3f} GB")
print(f"主项(权重+KV B=4 s=32768): {N*2/GB + kv(4,32768)/GB:.3f} GB")
