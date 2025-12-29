import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# ==========================================
# 0. 設定: 世界のダイナミクスを定義
# ==========================================
np.random.seed(42) # 再現性のため
T_max = 10.0       # シミュレーション時間
dt = 0.01          # 時間刻み
time = np.arange(0, T_max, dt)
n_steps = len(time)
split_idx = int(n_steps * 0.6) # 60%を過去(学習用)、40%を未来(予測用)とする

# パラダイムシフトのパラメータ μ(t)
# 負(安定)から正(発振)へ変化する -> "分岐"が発生する
mu_t = np.linspace(-0.5, 0.8, n_steps)

# その他の物理パラメータ
omega = 5.0  # 歴史の繰り返しの速さ (周波数)
beta = 1.0   # 歴史は完全に同じではない (非線形な周波数シフト)
noise_level = 0.15 # ガベージイン (テキストデータのノイズ)

# ==========================================
# 1. 真の姿 (Ground Truth) の生成
# ==========================================
# Landau-Stuart方程式 (確率微分方程式) の数値積分
z = np.zeros(n_steps, dtype=complex)
z[0] = 0.01 + 0.0j # 初期微動

for i in range(n_steps - 1):
    # 決定論的なダイナミクス step
    dz_dt = (mu_t[i] + 1j * omega) * z[i] - (1 + 1j * beta) * (np.abs(z[i])**2) * z[i]
    # ノイズ項 (Wiener過程)
    dW = noise_level * (np.random.randn() + 1j * np.random.randn()) * np.sqrt(dt)

    z[i+1] = z[i] + dz_dt * dt + dW

# 可視化のために実部を取り出す (例: あるトピックの社会的関心度)
truth_signal = np.real(z)
train_time = time[:split_idx]
train_signal = truth_signal[:split_idx] # 過去のデータ

# ==========================================
# 2. Model A: 現在のAI (Linear Stack Approximator)
# ==========================================
# 「過去のノイズ」に過剰適合する高次多項式モデル
# (ディープニューラルネットのパラメータの多さを高次数で表現)
degree = 18 # 非常に高い次数 = 膨大なパラメータ
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(train_time.reshape(-1, 1), train_signal)

# 未来予測
prediction_A = poly_model.predict(time.reshape(-1, 1))

# ==========================================
# 3. Model B: 提案手法 (Dynamic Landau-Stuart)
# ==========================================
# ノイズのない理想的なダイナミクス軌道
# (方程式の構造を知っている強み)
z_clean = np.zeros(n_steps, dtype=complex)
z_clean[0] = 0.01
for i in range(n_steps - 1):
    # ノイズ項 dW がない
    dz_dt = (mu_t[i] + 1j * omega) * z_clean[i] - (1 + 1j * beta) * (np.abs(z_clean[i])**2) * z_clean[i]
    z_clean[i+1] = z_clean[i] + dz_dt * dt

prediction_B = np.real(z_clean)

# ==========================================
# 4. 可視化 (Visualization)
# ==========================================
plt.figure(figsize=(12, 6))

# 領域の表示
plt.axvspan(time[0], time[split_idx], color='gray', alpha=0.1, label='PAST (Training Data with Garbage)')
plt.axvspan(time[split_idx], time[-1], color='yellow', alpha=0.1, label='FUTURE (Bifurcation & Unknown)')
plt.axvline(time[split_idx], color='k', linestyle='--', linewidth=1)

# プロット
plt.plot(time, truth_signal, 'k.', markersize=2, alpha=0.3, label='Ground Truth (Noisy Reality)')
plt.plot(time, prediction_A, 'r-', linewidth=2, label='Model A: Current AI (Linear Stack / Overfitting)')
plt.plot(time, prediction_B, 'b-', linewidth=3, label='Model B: Proposed (Landau-Stuart Dynamics)')

# グラフの装飾
plt.title("Comparison: Linear Approximation vs. Nonlinear Dynamics at a Paradigm Shift", fontsize=14)
plt.xlabel("Time (Evolution of Knowledge)", fontsize=12)
plt.ylabel("Causal Confidence (Amplitude)", fontsize=12)
plt.ylim(-1.5, 2.0) # Model Aの発散が酷すぎるのでY軸を制限
plt.legend(loc='upper left', frameon=True)
plt.grid(True, which='both', linestyle='--')

plt.tight_layout()
# 画像を保存
plt.savefig("causal_dynamics_comparison.png", dpi=300)
print("Graph saved to: causal_dynamics_comparison.png")
plt.show()
