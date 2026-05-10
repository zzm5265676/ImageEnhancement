import math

def rgb_to_hsv(r, g, b):
    """
    标准 RGB → HSV（严格按论文公式）
    输入：r, g, b ∈ [0, 1]
    输出：h ∈ [0, 6], s ∈ [0,1], v ∈ [0,1]
    """
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    delta = maxc - minc

    if maxc == 0:
        s = 0.0
        h = 0.0
    else:
        s = delta / maxc
        if maxc == r:
            h = (g - b) / delta
        elif maxc == g:
            h = 2 + (b - r) / delta
        else:
            h = 4 + (r - g) / delta
        h = h % 6
    return h, s, v

def rgb_to_hvi(r, g, b, k=1.0):
    """
    严格按论文 HVI: A New Color Space for Low-light Image Enhancement
    输入：r, g, b ∈ [0,1]
    输出：H_hat, V_hat, I_max（即 HVI）
    """
    # 1. 先算 HSV
    h_hsv, s_hsv, v_hsv = rgb_to_hsv(r, g, b)
    I_max = max(r, g, b)

    # 2. 极化 H（论文公式）
    h_polar = math.cos(math.pi * h_hsv / 3.0)
    v_polar = math.sin(math.pi * h_hsv / 3.0)

    # 3. 可学习强度塌缩 C_k（论文核心公式）
    eps = 1e-8
    Ck = math.sin(math.pi * I_max / 2.0) ** (1.0 / (k + eps))

    # 4. 计算最终 HV（论文 Eq.5）
    H_hat = Ck * s_hsv * h_polar
    V_hat = Ck * s_hsv * v_polar

    return H_hat, V_hat, I_max

def pixel_to_hsv_hvi(r, g, b, k=1.0):
    # 自动归一化到 0~1
    max_val = max(r, g, b)
    if max_val > 1.0:
        r /= 255.0
        g /= 255.0
        b /= 255.0

    h, s, v = rgb_to_hsv(r, g, b)
    H_hat, V_hat, I_max = rgb_to_hvi(r, g, b, k)

    print("=== 输入像素 ===")
    print(f"R = {r:.4f}, G = {g:.4f}, B = {b:.4f}")
    print("\n=== 输出 HSV ===")
    print(f"H (0~6) = {h:.4f}")
    print(f"S (0~1) = {s:.4f}")
    print(f"V (0~1) = {v:.4f}")
    print("\n=== 输出 HVI（论文）===")
    print(f"Ĥ = {H_hat:.4f}")
    print(f"V̂ = {V_hat:.4f}")
    print(f"Imax = {I_max:.4f}")
    print(f"k = {k}")

# ———————————————————— 你在这里改像素 ————————————————————
if __name__ == "__main__":
    # 输入像素：支持 0~255 或 0~1
    R = 235
    G = 125
    B = 233

    # 计算
    pixel_to_hsv_hvi(R, G, B, k=1.0)