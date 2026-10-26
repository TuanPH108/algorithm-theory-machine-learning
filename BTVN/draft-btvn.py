# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # BT1 – Bias-Variance Tradeoff
#
# **Học viên:** <Họ tên> &nbsp;·&nbsp; **MSHV:** `MSHV` *(thay vào ô seed bên dưới và tên file)*
#
# Mô phỏng ví dụ trang 58–68 của slides: hàm mục tiêu $f(x)=\sin(1+x^2)$, nhiễu $\varepsilon\sim\mathcal N(0,\sigma=0.03)$, học bằng polynomial regression bậc 1–9.

# %% [markdown]
# ## 1. Phân tích bài toán dưới góc nhìn toán học
#
# ### 1.1 Mô hình sinh dữ liệu
# Dữ liệu được sinh i.i.d. từ phân phối đồng thời $P(x,y)=P(x)\,P(y\mid x)$ với
# $$x\sim\mathcal U[0,1],\qquad y\mid x\sim\mathcal N\big(f(x),\sigma^2\big),\qquad f(x)=\sin(1+x^2),\ \sigma=0.03 .$$
# Tương đương $y=f(x)+\varepsilon$, $\varepsilon\perp x$, $\mathbb E[\varepsilon]=0$, $\mathrm{Var}(\varepsilon)=\sigma^2=9\times10^{-4}$.
# Lưu ý `np.random.normal(0, 0.03)` nhận **độ lệch chuẩn**, không phải phương sai.
#
# Một bộ dữ liệu $D=\{(x_n,y_n)\}_{n=1}^N$ là **biến ngẫu nhiên**: mỗi lần sinh lại $D$ ta được một mô hình học $g^{(D)}$ khác. Ba bộ $D_1,D_2,D_3$ chính là ba mẫu của biến ngẫu nhiên này, nên so sánh $g^{(D_1)},g^{(D_2)},g^{(D_3)}$ là cách quan sát trực tiếp sự dao động (variance) của thuật toán học.
#
# ### 1.2 Mục tiêu học và hàm tối ưu Bayes
# Với hàm lỗi bình phương, rủi ro (out-of-sample error) của một giả thuyết $h$ là
# $$E_{out}(h)=\mathbb E_{x,y}\big[(h(x)-y)^2\big].$$
# Cố định $x$, khai triển $y=f(x)+\varepsilon$:
# $$\mathbb E_y\big[(h(x)-y)^2\mid x\big]=(h(x)-f(x))^2-2(h(x)-f(x))\underbrace{\mathbb E[\varepsilon]}_{0}+\mathbb E[\varepsilon^2]=(h(x)-f(x))^2+\sigma^2 .$$
# Do đó $E_{out}(h)=\mathbb E_x\big[(h(x)-f(x))^2\big]+\sigma^2$. Hàm tốt nhất có thể là $h^\ast=f=\mathbb E[y\mid x]$ (hàm hồi quy Bayes), và ngay cả nó cũng có $E_{out}(f)=\sigma^2=9\times10^{-4}$. Đây là **sàn lỗi không thể giảm**: mọi $E_{out}$ đo được trong thực nghiệm phải $\gtrsim 9\times10^{-4}$.
#
# ### 1.3 Khảo sát hàm mục tiêu $f(x)=\sin(1+x^2)$ trên $[0,1]$
# - $f'(x)=2x\cos(1+x^2)$ triệt tiêu tại $x=0$ và $1+x^2=\pi/2\Rightarrow x^\ast=\sqrt{\pi/2-1}\approx0.7555$.
# - $f(0)=\sin1\approx0.8415$, $f(x^\ast)=1$ (cực đại), $f(1)=\sin2\approx0.9093$. Hàm tăng trên $[0,x^\ast]$, giảm trên $[x^\ast,1]$, tức là có **một đỉnh cong** mà đường thẳng không thể biểu diễn.
# - $f''(x)=2\cos(1+x^2)-4x^2\sin(1+x^2)$ đổi dấu trong $[0,1]$ (có điểm uốn) → cần ít nhất bậc 3 để bắt được cả độ cong lẫn điểm uốn.
# - Đặt $u=x^2$: $\sin(1+u)=\sin1+u\cos1-\tfrac{u^2}{2}\sin1-\tfrac{u^3}{6}\cos1+\dots$, nên $f$ là **hàm giải tích** với hệ số Taylor giảm theo giai thừa. Theo lý thuyết xấp xỉ, sai số xấp xỉ đa thức tốt nhất bậc $k$ giảm theo cấp số nhân khi $k$ tăng.
# - Độ biến thiên của tín hiệu: $\mathrm{Var}_x[f(x)]\approx3.08\times10^{-3}$, so với nhiễu $\sigma^2=9\times10^{-4}$ thì tỉ số tín hiệu/nhiễu chỉ $\approx3.4$. Tín hiệu yếu nên mô hình phức tạp rất dễ "học cả nhiễu".
#
# ### 1.4 Tập giả thuyết $\mathcal H_k$ và sai số xấp xỉ
# $$\mathcal H_k=\Big\{h_w(x)=\sum_{j=0}^{k}w_jx^j\Big\},\quad k=1,\dots,9,\qquad \mathcal H_1\subset\mathcal H_2\subset\dots\subset\mathcal H_9 .$$
# Mỗi $\mathcal H_k$ là **mô hình tuyến tính theo tham số** trên đặc trưng $\Phi_k(x)=(1,x,\dots,x^k)$, có $p=k+1$ tham số và $d_{VC}=k+1$.
# Giả thuyết tốt nhất trong lớp là phép chiếu trực giao (trong $L^2(P_x)$) của $f$ lên $\mathcal H_k$:
# $$h_k^\ast=\arg\min_{h\in\mathcal H_k}\mathbb E_x[(h(x)-f(x))^2],\qquad b_k^2:=\mathbb E_x[(h_k^\ast(x)-f(x))^2]\ \ (\text{bias}^2\text{ tiệm cận}).$$
# Tính số (cầu phương Gauss–Legendre, không cần dữ liệu):
#
# | $k$ | 0 | 1 | 2 | 3 | 4 | 5 | 9 |
# |---|---|---|---|---|---|---|---|
# | $b_k^2$ | $3.1\times10^{-3}$ | $7.8\times10^{-4}$ | $3.6\times10^{-4}$ | $6.2\times10^{-6}$ | $1.2\times10^{-8}$ | $4.3\times10^{-9}$ | $5\times10^{-16}$ |
# | so với $\sigma^2=9\times10^{-4}$ | lớn hơn | **≈ ngang** | ≈ 40% | ≪ | ≪ | ≪ | ≪ |
#
# **Hệ quả quan trọng:** bậc 1 có bias² xấp xỉ bằng chính phương sai nhiễu nên $E_{out}$ của nó không thể xuống dưới $\approx\sigma^2+b_1^2\approx1.7\times10^{-3}$ dù có bao nhiêu dữ liệu. Từ bậc 3 trở lên bias gần như bằng 0, nên mọi khác biệt giữa bậc 3 và bậc 9 đều đến từ **variance**.
#
# ### 1.5 Thuật toán học: bình phương tối thiểu (ERM)
# Với ma trận Vandermonde $X\in\mathbb R^{N\times p}$, $X_{nj}=x_n^j$:
# $$\hat w=\arg\min_w\frac1N\|Xw-y\|^2\ \Rightarrow\ X^\top X\hat w=X^\top y\ \Rightarrow\ \hat w=X^{+}y .$$
# - Nếu $p\ge N$ (ở đây $k=9$, $N=10$): hệ có nghiệm **nội suy** $X\hat w=y$, nên $E_{in}=0$. Mô hình "thuộc lòng" cả nhiễu.
# - **Ổn định số:** với $x\sim\mathcal U[0,1]$, $\frac1N X^\top X\to\big[\tfrac1{i+j+1}\big]_{ij}$ là **ma trận Hilbert**, có số điều kiện $\approx1.9\times10^1$ ($k=1$), $1.6\times10^4$ ($k=3$), $1.5\times10^7$ ($k=5$), $1.6\times10^{13}$ ($k=9$). Giải trực tiếp phương trình chuẩn với bậc cao sẽ mất độ chính xác, nên code dùng `numpy.polynomial.Polynomial.fit`: hàm này đổi biến $x\mapsto[-1,1]$ rồi giải least squares bằng SVD.

# %% [markdown]
# ## 2. Phân tích bài toán dưới góc nhìn học máy
#
# ### 2.1 $E_{in}$, $E_{out}$ và ước lượng $\hat E_{out}$
# $$E_{in}(g)=\frac1N\sum_{n=1}^N(g(x_n)-y_n)^2,\qquad \hat E_{out}(g)=\frac1M\sum_{m=1}^M(g(x'_m)-y'_m)^2 .$$
# - $E_{in}$ là ước lượng **lạc quan** (có độ chệch xuống) vì $g$ được chọn chính trên các điểm đó.
# - Vì tập test độc lập với $g$, $\hat E_{out}$ là ước lượng **không chệch** của $E_{out}(g)$. Khi $g\approx f$ thì $(g-y)^2\approx\varepsilon^2$ với $\mathrm{Var}(\varepsilon^2)=2\sigma^4$, nên
# $$\mathrm{sd}\big(\hat E_{out}\big)\approx\sigma^2\sqrt{2/M}:\quad M=10\Rightarrow\pm45\%,\quad M=100\Rightarrow\pm14\%,\quad M=1000\Rightarrow\pm4.5\% .$$
#   Với tập test 10 điểm, chênh lệch $\hat E_{out}$ giữa các bậc thấp chỉ có ý nghĩa khi lớn hơn khoảng ±45%. Đây cũng là lý do đề yêu cầu thử thêm 100 và 1000 điểm.
#
# ### 2.2 Phân rã Bias–Variance
# Gọi $\bar g(x)=\mathbb E_D[g^{(D)}(x)]$ là "mô hình trung bình". Với $x$ cố định:
# $$\mathbb E_D\big[(g^{(D)}(x)-f(x))^2\big]=\mathbb E_D\big[(g^{(D)}-\bar g+\bar g-f)^2\big]=\underbrace{\mathbb E_D[(g^{(D)}(x)-\bar g(x))^2]}_{\text{var}(x)}+\underbrace{(\bar g(x)-f(x))^2}_{\text{bias}^2(x)}$$
# (số hạng chéo bằng 0 vì $\mathbb E_D[g^{(D)}-\bar g]=0$). Kết hợp với mục 1.2 và lấy kỳ vọng theo $x$:
# $$\boxed{\ \mathbb E_D\big[E_{out}(g^{(D)})\big]=\underbrace{\mathbb E_x[(\bar g-f)^2]}_{\text{bias}^2}+\underbrace{\mathbb E_x\mathbb E_D[(g^{(D)}-\bar g)^2]}_{\text{variance}}+\underbrace{\sigma^2}_{\text{nhiễu}}\ }$$
# - **Bias** đo khả năng của *tập giả thuyết*: dù có vô số tập dữ liệu thì trung bình các mô hình vẫn lệch khỏi $f$ bao xa. Khi $N\to\infty$, $\bar g\to h_k^\ast$ nên bias² $\to b_k^2$ (mục 1.4), **không phụ thuộc $N$**.
# - **Variance** đo độ nhạy của mô hình với *tập dữ liệu cụ thể*: phụ thuộc vào tỉ lệ giữa độ phức tạp $p=k+1$ và $N$.
#
# ### 2.3 Ước lượng lý thuyết cho least squares
# Với hồi quy tuyến tính $p$ tham số, phần dư mà $\mathcal H_k$ không biểu diễn được ($f-h_k^\ast$, có độ lớn $b_k^2$) đóng vai trò như nhiễu bổ sung. Các xấp xỉ chuẩn (chính xác khi thiết kế Gauss, tiệm cận khi $N\gg p$) là:
# $$\mathbb E[E_{in}]\approx(\sigma^2+b_k^2)\Big(1-\frac pN\Big),\qquad \text{variance}\approx(\sigma^2+b_k^2)\,\frac{p}{N-p-1},\qquad \mathbb E[E_{out}]\approx\sigma^2+b_k^2+\text{variance}.$$
# Hệ quả: generalization gap $\approx2(\sigma^2+b_k^2)\,p/N$; variance **tỉ lệ $1/N$** và tăng theo $p$; khi $p\ge N-1$ variance kỳ vọng **không hữu hạn** (mô hình nội suy).
#
# Dự đoán từ công thức (chưa chạy thực nghiệm):
#
# | $k$ | var ($N$=10) | var ($N$=100) | var ($N$=1000) | $\mathbb E[E_{out}]$ ($N$=10 / 100 / 1000) |
# |---|---|---|---|---|
# | 1 | $4.8\times10^{-4}$ | $3.5\times10^{-5}$ | $3.4\times10^{-6}$ | $2.2\times10^{-3}$ / $1.72\times10^{-3}$ / $1.69\times10^{-3}$ |
# | 2 | $6.3\times10^{-4}$ | $3.9\times10^{-5}$ | $3.8\times10^{-6}$ | $1.9\times10^{-3}$ / $1.29\times10^{-3}$ / $1.26\times10^{-3}$ |
# | 3 | $7.3\times10^{-4}$ | $3.8\times10^{-5}$ | $3.6\times10^{-6}$ | $1.6\times10^{-3}$ / $9.4\times10^{-4}$ / $9.1\times10^{-4}$ |
# | 5 | $1.8\times10^{-3}$ | $5.8\times10^{-5}$ | $5.4\times10^{-6}$ | $2.7\times10^{-3}$ / $9.6\times10^{-4}$ / $9.05\times10^{-4}$ |
# | 7 | $7.2\times10^{-3}$ | $7.9\times10^{-5}$ | $7.3\times10^{-6}$ | $8.1\times10^{-3}$ / $9.8\times10^{-4}$ / $9.07\times10^{-4}$ |
# | 8, 9 | $\infty$ | $\approx10^{-4}$ | $\approx9\times10^{-6}$ | $\infty$ / $\approx1.0\times10^{-3}$ / $\approx9.1\times10^{-4}$ |
#
# ⇒ Bậc tối ưu dự đoán: $k^\ast\approx3$ khi $N=10$; $k^\ast\approx3$–$4$ khi $N=100$; khi $N=1000$, mọi $k\ge3$ gần như bằng nhau ($\approx\sigma^2$).
# *Lưu ý:* với $N=10$ và $x$ phân phối đều, công thức trên **đánh giá thấp** variance của bậc cao. Mô hình bậc cao ngoại suy rất mạnh ở các khoảng trống gần biên, làm phân phối lỗi có đuôi dày. Thực nghiệm sẽ kiểm tra điều này.
#
# ### 2.4 Góc nhìn VC / approximation–generalization tradeoff
# $$E_{out}(g)\le E_{in}(g)+\mathcal O\Big(\sqrt{\tfrac{d_{VC}}{N}\log N}\Big),\qquad d_{VC}=k+1 .$$
# Tăng $k$ làm giảm $E_{in}$ (xấp xỉ tốt hơn) nhưng tăng số hạng phức tạp (tổng quát hoá kém hơn). Quy tắc kinh nghiệm $N\gtrsim10\,d_{VC}$ cho thấy: với $N=10$ chỉ nên dùng mô hình rất đơn giản; với $N=100$ dùng được tới $k\approx9$; với $N=1000$ thì $k=9$ hoàn toàn an toàn.
#
# ### 2.5 Các giả thuyết cần kiểm chứng bằng thực nghiệm
# | | Giả thuyết | Kiểm chứng bằng |
# |---|---|---|
# | H1 | $E_{in}$ giảm đơn điệu theo $k$ ($\mathcal H_k$ lồng nhau); $E_{in}=0$ khi $k=9$, $N=10$ | Bảng lỗi |
# | H2 | $\hat E_{out}$ có dạng chữ U theo $k$; đáy dịch về phía $k$ lớn khi $N$ tăng | Bảng lỗi, biểu đồ lỗi |
# | H3 | Bias² phụ thuộc $k$ nhưng **không phụ thuộc $N$** ($\to b_k^2$) | Monte-Carlo, so với bảng 1.4 |
# | H4 | Variance $\propto p/N$: giảm khoảng 10 lần khi $N$ tăng 10 lần | Monte-Carlo, so với bảng 2.3 |
# | H5 | Khi $N$ lớn: $E_{in}\approx\hat E_{out}\to\sigma^2+b_k^2$ | Bảng tổng hợp, $N=1000$ |

# %% [markdown]
# ## 3. Kiến thức nền và kế hoạch triển khai
#
# **Kiến thức nền cần xem lại:** xác suất (kỳ vọng có điều kiện, phương sai, luật số lớn); hồi quy tuyến tính và phương trình chuẩn, pseudo-inverse; xấp xỉ đa thức và phép chiếu $L^2$; số điều kiện của ma trận (Vandermonde/Hilbert); $E_{in}/E_{out}$, chiều VC, overfitting/underfitting; phân rã bias–variance; learning curve. Về NumPy: `np.random.uniform`, `np.random.normal(loc, scale=σ)`, `numpy.polynomial.Polynomial.fit`.
#
# **Kế hoạch:**
# 1. Cố định seed theo MSHV (ngay sau `import numpy`, trước khi sinh dữ liệu).
# 2. Viết hàm `make_dataset(N)`, `fit_poly`, `mse` theo đúng mô hình ở mục 1.
# 3. **Thực nghiệm chính** cho $N=10,100,1000$: sinh $D_1,D_2,D_3$; huấn luyện bậc 1–9; lập bảng $E_{in}/\hat E_{out}$ (trang 66); vẽ hình 3×3 (trang 68); vẽ lỗi theo bậc. Kiểm chứng H1, H2, H5.
# 4. **Kiểm chứng định lượng:** Monte-Carlo $R=500$ tập dữ liệu cho mỗi $N$ để ước lượng bias², variance, rồi đặt cạnh dự đoán lý thuyết $b_k^2$ và $(\sigma^2+b_k^2)p/(N-p-1)$. Kiểm chứng H3, H4.
# 5. Kết luận, trả lời 2 câu hỏi của đề.

# %% [markdown]
# ## 4. Cài đặt
#
# ### 4.1 Import thư viện và cố định seed

# %%
import numpy as np
import random

seed = 12345678   # <-- THAY 12345678 BẰNG MSHV (dãy chữ số mã số học viên)
np.random.seed(seed)
random.seed(seed)

import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import os

# Chạy được cả trong Jupyter/VS Code (hiện bảng, hình inline) lẫn `python BT1_MSHV.py` (in bảng, lưu hình)
try:
    from IPython import get_ipython
    IN_NOTEBOOK = get_ipython() is not None and "IPKernelApp" in get_ipython().config
except ImportError:
    IN_NOTEBOOK = False
if IN_NOTEBOOK:
    from IPython.display import display
else:
    plt.switch_backend("Agg")
    def display(obj):
        if hasattr(obj, "data") and hasattr(obj, "format"):   # pandas Styler
            with pd.option_context("display.float_format", "{:.2e}".format, "display.width", 200):
                print(obj.data.to_string())
        else:
            with pd.option_context("display.width", 200, "display.max_columns", 50):
                print(obj)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
_fig_count = 0

def save_show():
    # Lưu hình hiện tại vào figures/ rồi hiển thị (trong notebook)
    global _fig_count
    _fig_count += 1
    plt.savefig(os.path.join(FIG_DIR, f"fig_{_fig_count:02d}.png"), dpi=120, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close("all")

pd.set_option("display.float_format", lambda v: f"{v:.6f}" if abs(v) < 1e4 else f"{v:.3e}")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

# %% [markdown]
# ### 4.2 Hàm tiện ích

# %%
SIGMA = 0.03                 # độ lệch chuẩn của nhiễu
DEGREES = list(range(1, 10))  # bậc đa thức 1..9
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]  # D1, D2, D3

def f(x):
    return np.sin(1 + x**2)

def make_dataset(n):
    # Sinh 1 bộ dữ liệu gồm train (n điểm) và test (n điểm) độc lập
    x_tr = np.random.uniform(0, 1, n)
    y_tr = f(x_tr) + np.random.normal(0, SIGMA, n)
    x_te = np.random.uniform(0, 1, n)
    y_te = f(x_te) + np.random.normal(0, SIGMA, n)
    return dict(x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)

def fit_poly(x, y, degree):
    # Least squares trên miền đã chuẩn hoá -> ổn định số khi bậc cao
    return Polynomial.fit(x, y, degree)

def mse(model, x, y):
    return np.mean((model(x) - y) ** 2)

def run_experiment(n, n_sets=3):
    # Sinh n_sets bộ dữ liệu kích thước n, huấn luyện 9 mô hình cho mỗi bộ
    datasets, models, rows = [], [], []
    for i in range(n_sets):
        D = make_dataset(n)
        ms = {k: fit_poly(D["x_train"], D["y_train"], k) for k in DEGREES}
        for k, g in ms.items():
            rows.append(dict(dataset=f"D{i+1}", degree=k,
                             E_in=mse(g, D["x_train"], D["y_train"]),
                             E_out_hat=mse(g, D["x_test"], D["y_test"])))
        datasets.append(D); models.append(ms)
    return datasets, models, pd.DataFrame(rows)

def show_tables(df):
    # Mỗi bộ dữ liệu một bảng: hàng = bậc, cột = E_in, Ê_out (định dạng như slide trang 66)
    t = df.pivot(index="degree", columns="dataset", values=["E_in", "E_out_hat"])
    t = t.swaplevel(axis=1).sort_index(axis=1, level=0)
    t = t.rename(columns={"E_in": "E_in", "E_out_hat": "Ê_out"}, level=1)
    t.index.name = "Degree"
    display(t)

def plot_fits(datasets, models, n, ylim=(0.70, 1.10)):
    # Hình lớn 3x3: mỗi ô là 1 bậc, gồm f(x) và 3 mô hình từ D1, D2, D3
    xs = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), DEGREES):
        ax.plot(xs, f(xs), "k--", lw=2, label="f(x) = sin(1+x²)")
        for i, (D, ms) in enumerate(zip(datasets, models)):
            ax.plot(xs, ms[k](xs), color=COLORS[i], lw=1.6, label=f"g(D{i+1})")
            if n <= 100:
                ax.scatter(D["x_train"], D["y_train"], s=14 if n == 10 else 5,
                           color=COLORS[i], alpha=0.8 if n == 10 else 0.35)
        ax.set_title(f"Bậc {k}")
        ax.set_ylim(*ylim)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Polynomial regression bậc 1–9 trên D1, D2, D3 (N_train = {n})", fontsize=14)
    fig.tight_layout()
    save_show()

def plot_errors(df, n):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, (name, sub) in zip(axes, df.groupby("dataset")):
        ax.semilogy(sub.degree, sub.E_in.clip(lower=1e-12), "o-", label="E_in")
        ax.semilogy(sub.degree, sub.E_out_hat, "s-", label="Ê_out")
        ax.axhline(SIGMA**2, color="gray", ls=":", label="σ² = 9e-4")
        ax.set_title(f"{name} (N = {n})"); ax.set_xlabel("Bậc đa thức"); ax.set_xticks(DEGREES)
    axes[0].set_ylabel("MSE (log)"); axes[0].legend()
    fig.tight_layout(); save_show()


# %% [markdown]
# ## 5. Thực nghiệm với N = 10 (đúng như đề)
#
# ### Bảng $E_{in}$ / $\hat E_{out}$ (tương tự trang 66)
#
# *Tham chiếu từ slides (trang 66, một bộ dữ liệu $N=10$ của giảng viên):* bậc 1: $E_{in}=0.00147$, $\hat E_{out}=0.00215$; bậc 2: $0.00093/0.00137$; bậc 3: $0.00029/0.00081$; bậc 4: $0.00012/0.00116$; bậc 5: $0.00011/0.00110$; bậc 6: $0.00007/0.00108$; bậc 7: $0.00004/0.00118$; bậc 8: $0.00003/0.00151$; bậc 9: $0.00000/0.00231$. Giá trị cụ thể phụ thuộc seed, nhưng xu hướng phải giống: $E_{in}$ giảm về 0, $\hat E_{out}$ thấp nhất quanh bậc 3 rồi tăng lại.

# %%
datasets_10, models_10, df_10 = run_experiment(10)
show_tables(df_10)

# %% [markdown]
# ### Hình 3×3 (tương tự trang 68)

# %%
plot_fits(datasets_10, models_10, 10)

# %% [markdown]
# ### $E_{in}$ và $\hat E_{out}$ theo bậc

# %%
plot_errors(df_10, 10)

# %% [markdown]
# ### Nhận xét (N = 10)
#
#
# **Bảng lỗi:**
# - $E_{in}$ **giảm đơn điệu** khi tăng bậc trên cả 3 bộ, và bằng $0$ ở bậc 9: 10 tham số khớp chính xác 10 điểm.
# - $\hat E_{out}$ có dạng chữ U rất rõ: nhỏ nhất ở bậc 2–3 (D1: bậc 3, $\approx 9.2\times10^{-4}$; D2: bậc 2, $\approx 2.9\times10^{-3}$; D3: bậc 3, $\approx 1.7\times10^{-3}$), rồi **bùng nổ** từ bậc 4–5 trở lên (bậc 9 lên tới $10^4$–$10^9$).
# - Khoảng cách $\hat E_{out}-E_{in}$ (generalization gap) tăng rất nhanh theo bậc → **overfitting**.
#
# **Hình 3×3:**
# - **Bậc 1 (mô hình đơn giản):** ba đường $g^{(D_1)},g^{(D_2)},g^{(D_3)}$ gần như trùng nhau (**variance thấp**) nhưng đều là đường thẳng, không bắt được đỉnh cong của $f$ quanh $x\approx0.76$ (**bias cao**).
# - **Bậc 2–3:** bám sát $f$, ba đường vẫn khá gần nhau → cân bằng bias/variance tốt nhất, khớp với $\hat E_{out}$ nhỏ nhất.
# - **Bậc 4–9 (mô hình phức tạp):** đi qua (gần như) mọi điểm train nên "trung bình" có thể sát $f$ (**bias thấp**), nhưng mỗi bộ dữ liệu cho một đường dao động hoàn toàn khác, nhất là ở các khoảng trống giữa điểm và gần biên $x=0$, $x=1$ (**variance rất cao**).
#
# ⇒ Với $N=10$, độ phức tạp mô hình phải tương xứng với lượng dữ liệu: tăng bậc làm giảm bias nhưng variance tăng nhanh hơn nhiều.
#

# %% [markdown]
# ## 6. Thực nghiệm với N = 100
#
# ### Bảng $E_{in}$ / $\hat E_{out}$ (tương tự trang 66)

# %%
datasets_100, models_100, df_100 = run_experiment(100)
show_tables(df_100)

# %% [markdown]
# ### Hình 3×3 (tương tự trang 68)

# %%
plot_fits(datasets_100, models_100, 100)

# %% [markdown]
# ### $E_{in}$ và $\hat E_{out}$ theo bậc

# %%
plot_errors(df_100, 100)

# %% [markdown]
# ### Nhận xét (N = 100)
#
#
# - $E_{in}$ vẫn giảm theo bậc nhưng **chậm** (từ $\approx1.5$–$2.0\times10^{-3}$ ở bậc 1 xuống $\approx0.6$–$1.0\times10^{-3}$ ở bậc 9) và không thể về 0 vì 100 điểm có nhiễu.
# - $\hat E_{out}$ **không còn bùng nổ**: thấp nhất khoảng bậc 3–6 ($\approx 9.6\times10^{-4}$–$1.0\times10^{-3}$, rất gần $\sigma^2=9\times10^{-4}$), chỉ tăng nhẹ ở bậc 7–9 (D1, D3) → variance của mô hình bậc cao vẫn còn nhưng nhỏ.
# - Trên hình 3×3, ba đường của bậc 3–9 gần như chồng lên nhau và lên $f$; chỉ còn hơi "vểnh" ở hai biên với bậc 8–9. Bậc 1–2 vẫn lệch khỏi $f$ **giống hệt như khi $N=10$** → bias không đổi.
#

# %% [markdown]
# ## 7. Thực nghiệm với N = 1000
#
# ### Bảng $E_{in}$ / $\hat E_{out}$ (tương tự trang 66)

# %%
datasets_1000, models_1000, df_1000 = run_experiment(1000)
show_tables(df_1000)

# %% [markdown]
# ### Hình 3×3 (tương tự trang 68)

# %%
plot_fits(datasets_1000, models_1000, 1000)

# %% [markdown]
# ### $E_{in}$ và $\hat E_{out}$ theo bậc

# %%
plot_errors(df_1000, 1000)

# %% [markdown]
# ### Nhận xét (N = 1000)
#
#
# - $E_{in}\approx\hat E_{out}$ với mọi bậc (gap gần như 0) → không còn overfitting, kể cả bậc 9.
# - Từ bậc 3 trở lên cả hai lỗi đều $\approx 8.6$–$9.6\times10^{-4}\approx\sigma^2$: mô hình đã chạm sàn nhiễu, phần còn lại là irreducible error.
# - Bậc 1 vẫn có lỗi $\approx1.7\times10^{-3}\approx\sigma^2+$bias$^2$ (≈ $9\times10^{-4}+8\times10^{-4}$); bậc 2 $\approx1.25\times10^{-3}$. Dữ liệu nhiều gấp 100 lần **không** cải thiện được hai mô hình này → đó là lỗi do bias.
# - Trên hình 3×3, ba đường của mỗi bậc trùng khít nhau (variance ≈ 0).
#

# %% [markdown]
# ### So sánh tổng hợp ba kích thước N (trung bình trên D1–D3)

# %%
summary = pd.concat({n: d.groupby("degree")[["E_in", "E_out_hat"]].mean()
                     for n, d in [(10, df_10), (100, df_100), (1000, df_1000)]}, axis=1)
summary.columns = [f"{c} (N={n})" for n, c in summary.columns]
display(summary)

# %% [markdown]
# ## 8. Kiểm chứng định lượng: ước lượng Bias² và Variance bằng Monte-Carlo
#
# Ba bộ dữ liệu là quá ít để ước lượng $\bar g$. Ở đây lặp $R=500$ lần: mỗi lần sinh một tập train kích thước $N$, fit mô hình bậc $k$, dự đoán trên lưới 200 điểm $x\in[0,1]$. Từ đó:
# $\bar g(x)\approx\frac1R\sum_r g^{(r)}(x)$, $\;\text{bias}^2\approx\text{mean}_x(\bar g-f)^2$, $\;\text{var}\approx\text{mean}_x\,\text{Var}_r\,g^{(r)}(x)$.
#
# Với $N=10$ và bậc cao, một vài mô hình "bùng nổ" làm trung bình rất lớn, nên dùng trục log.

# %%
R = 500
x_grid = np.linspace(0, 1, 200)

def bias_variance(n, degree, R=R):
    preds = np.empty((R, x_grid.size))
    for r in range(R):
        x = np.random.uniform(0, 1, n)
        y = f(x) + np.random.normal(0, SIGMA, n)
        preds[r] = fit_poly(x, y, degree)(x_grid)
    g_bar = preds.mean(axis=0)
    bias2 = np.mean((g_bar - f(x_grid)) ** 2)
    var = np.mean(preds.var(axis=0))
    return bias2, var, preds

bv_rows, sample_preds = [], {}
for n in [10, 100, 1000]:
    for k in DEGREES:
        b2, v, preds = bias_variance(n, k)
        bv_rows.append(dict(N=n, degree=k, bias2=b2, variance=v))
        if k in (1, 3, 9):
            sample_preds[(n, k)] = preds[:50]
bv = pd.DataFrame(bv_rows)
bv["bias2+var+σ²"] = bv.bias2 + bv.variance + SIGMA**2
display(bv.pivot(index="degree", columns="N", values=["bias2", "variance"]).style.format("{:.2e}"))

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
for ax, n in zip(axes, [10, 100, 1000]):
    s = bv[bv.N == n]
    ax.semilogy(s.degree, s.bias2.clip(lower=1e-12), "o-", label="Bias²")
    ax.semilogy(s.degree, s.variance, "s-", label="Variance")
    ax.axhline(SIGMA**2, color="gray", ls=":", label="σ²")
    ax.set_title(f"N = {n}"); ax.set_xlabel("Bậc đa thức"); ax.set_xticks(DEGREES)
axes[0].set_ylabel("giá trị (log)"); axes[0].legend()
fig.suptitle("Bias² và Variance theo độ phức tạp mô hình", fontsize=13)
fig.tight_layout(); save_show()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for k, mk in zip([1, 3, 5, 9], "o^sD"):
    s = bv[bv.degree == k]
    axes[0].loglog(s.N, s.bias2.clip(lower=1e-12), mk + "-", label=f"bậc {k}")
    axes[1].loglog(s.N, s.variance, mk + "-", label=f"bậc {k}")
axes[0].set_title("Bias² theo N"); axes[1].set_title("Variance theo N")
for ax in axes:
    ax.set_xlabel("N (kích thước tập train)"); ax.legend(); ax.set_xticks([10, 100, 1000])
fig.tight_layout(); save_show()

# %% [markdown]
# ### So sánh thực nghiệm với dự đoán lý thuyết (mục 1.4 và 2.3)
# $b_k^2$ được tính bằng phép chiếu $L^2$ của $f$ lên $\mathcal H_k$ (cầu phương Gauss–Legendre), còn variance dự đoán là $(\sigma^2+b_k^2)\,p/(N-p-1)$ với $p=k+1$.

# %%
gl_x, gl_w = np.polynomial.legendre.leggauss(200)
gl_x, gl_w = (gl_x + 1) / 2, gl_w / 2          # chuyển về U[0,1]

def asymptotic_bias2(k):
    V = np.vander(2 * gl_x - 1, k + 1)
    sw = np.sqrt(gl_w)
    c, *_ = np.linalg.lstsq(V * sw[:, None], f(gl_x) * sw, rcond=None)
    return np.sum(gl_w * (V @ c - f(gl_x)) ** 2)

theory = []
for n in [10, 100, 1000]:
    for k in DEGREES:
        b2, p = asymptotic_bias2(k), k + 1
        var_th = (SIGMA**2 + b2) * p / (n - p - 1) if n - p - 1 > 0 else np.inf
        theory.append(dict(N=n, degree=k, bias2_theory=b2, var_theory=var_th))
cmp_df = bv.merge(pd.DataFrame(theory), on=["N", "degree"])
display(cmp_df.pivot(index="degree", columns="N",
        values=["bias2", "bias2_theory", "variance", "var_theory"]).style.format("{:.2e}"))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
for ax, n in zip(axes, [10, 100, 1000]):
    s = cmp_df[cmp_df.N == n]
    ax.semilogy(s.degree, s.variance, "s-", color="tab:orange", label="Variance (Monte-Carlo)")
    ax.semilogy(s.degree, s.var_theory.replace(np.inf, np.nan), "s--", color="tab:orange", alpha=0.6, label="Variance (lý thuyết)")
    ax.semilogy(s.degree, s.bias2.clip(lower=1e-16), "o-", color="tab:blue", label="Bias² (Monte-Carlo)")
    ax.semilogy(s.degree, s.bias2_theory.clip(lower=1e-16), "o--", color="tab:blue", alpha=0.6, label="Bias² tiệm cận b_k²")
    ax.axhline(SIGMA**2, color="gray", ls=":", label="σ²")
    ax.set_title(f"N = {n}"); ax.set_xlabel("Bậc đa thức"); ax.set_xticks(DEGREES)
axes[0].set_ylabel("giá trị (log)"); axes[0].legend(fontsize=8)
fig.suptitle("Thực nghiệm (nét liền) và dự đoán lý thuyết (nét đứt)", fontsize=13)
fig.tight_layout(); save_show()

# %%
# "Chùm" 50 mô hình từ 50 tập dữ liệu khác nhau: hàng = bậc, cột = N
fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
for i, k in enumerate([1, 3, 9]):
    for j, n in enumerate([10, 100, 1000]):
        ax = axes[i, j]
        P = sample_preds[(n, k)]
        for p in P:
            ax.plot(x_grid, p, color="tab:blue", alpha=0.15, lw=1)
        ax.plot(x_grid, P.mean(axis=0), color="tab:orange", lw=2, label="trung bình ḡ (50 mô hình)")
        ax.plot(x_grid, f(x_grid), "k--", lw=2, label="f(x)")
        ax.set_ylim(0.70, 1.10)
        ax.set_title(f"Bậc {k}, N = {n}")
axes[0, 0].legend(loc="lower right", fontsize=8)
fig.suptitle("Độ rộng của chùm = variance, khoảng cách ḡ–f = bias", fontsize=13)
fig.tight_layout(); save_show()

# %% [markdown]
# ## 9. Trả lời câu hỏi và kết luận
#
#
# ### Bảng tóm tắt từ Monte-Carlo ($R=500$)
#
# | Bậc | Bias² (N=10 / 100 / 1000) | Variance (N=10 / 100 / 1000) |
# |---|---|---|
# | 1 | $8.2\times10^{-4}$ / $8.05\times10^{-4}$ / $8.05\times10^{-4}$ | $4.7\times10^{-4}$ / $3.8\times10^{-5}$ / $4.0\times10^{-6}$ |
# | 2 | $4.3\times10^{-4}$ / $3.7\times10^{-4}$ / $3.7\times10^{-4}$ | $1.1\times10^{-3}$ / $4.6\times10^{-5}$ / $4.3\times10^{-6}$ |
# | 3 | $4.5\times10^{-5}$ / $6.5\times10^{-6}$ / $6.4\times10^{-6}$ | $5.9\times10^{-3}$ / $3.9\times10^{-5}$ / $3.6\times10^{-6}$ |
# | 9 | (*) / $1.4\times10^{-7}$ / $2.5\times10^{-8}$ | $6.6\times10^{12}$ / $1.4\times10^{-4}$ / $1.0\times10^{-5}$ |
#
# (*) Với $N=10$, bậc cao, ước lượng bias² trên bị "nhiễm" bởi variance: $\bar g$ chỉ là trung bình của $R$ mô hình nên $\mathbb E[\widehat{\text{bias}^2}] = \text{bias}^2 + \text{var}/R$. Bậc 9: $\text{var}/R \approx 6.6\times10^{12}/500 \approx 1.3\times10^{10}$, cùng bậc độ lớn với giá trị đo được ($2.3\times10^{10}$). Đây là hiệu ứng của variance khổng lồ, không phải bias thật.
#
# *(Các số trên lấy từ lần chạy với seed tạm `12345678`; khi thay MSHV, giá trị cụ thể thay đổi nhẹ nhưng xu hướng giữ nguyên.)*
#
# ### Đối chiếu với các giả thuyết ở mục 2.5
# - **H1 đúng:** $E_{in}$ giảm đơn điệu theo bậc trên mọi bộ dữ liệu, và $E_{in}=0$ tại bậc 9 khi $N=10$.
# - **H2 đúng:** $\hat E_{out}$ có đáy ở bậc 2–3 khi $N=10$, ở bậc 3–6 khi $N=100$, và phẳng từ bậc 3 khi $N=1000$.
# - **H3 đúng:** bias² Monte-Carlo khi $N=100, 1000$ trùng với $b_k^2$ lý thuyết. Bậc 1: $8.05\times10^{-4}$ so với $7.8\times10^{-4}$, phần chênh nhỏ do lưới 200 điểm đều đặt trọng số lớn hơn ở hai biên. Bậc 2: $3.67$ so với $3.56\times10^{-4}$. Bậc 3: $6.4$ so với $6.2\times10^{-6}$. Với bậc ≥ 4, giá trị đo được ($\sim10^{-8}$) chính là sàn Monte-Carlo var$/R$, vì bias thật nhỏ hơn thế.
# - **H4 đúng:** variance đo được khớp công thức $(\sigma^2+b_k^2)p/(N-p-1)$ với sai lệch không quá 35% khi $N=100, 1000$. Với $N=10$, công thức đúng cho bậc 1 ($4.7$ so với $4.8\times10^{-4}$) nhưng đánh giá thấp mạnh từ bậc 3 trở lên, đúng như cảnh báo ở mục 2.3: các mô hình ngoại suy ở biên tạo ra đuôi dày.
# - **H5 đúng:** khi $N=1000$ thì $E_{in}\approx\hat E_{out}$, bậc 1 đạt $\approx1.7\times10^{-3}\approx\sigma^2+b_1^2$, bậc ≥ 3 đạt $\approx\sigma^2$.
#
# ### Nhận xét chung về bias và variance
# - **Mô hình đơn giản (bậc 1–2):** bias cao, variance thấp → underfitting. Lỗi không giảm được bằng cách thêm dữ liệu.
# - **Mô hình phức tạp (bậc cao):** bias ≈ 0 (tập $\mathcal H_9$ đủ sức biểu diễn $f$ rất sát), nhưng variance lớn, đặc biệt khi $N$ nhỏ → overfitting.
# - **Mô hình tốt nhất** là mô hình cân bằng hai thành phần; điểm cân bằng phụ thuộc vào $N$: bậc 2–3 khi $N=10$, bậc 3–6 khi $N=100$, gần như mọi bậc $\ge3$ khi $N=1000$.
# - **Ở vùng biên** ($x$ gần 0 hoặc 1) variance lớn nhất vì ít điểm "giữ" đa thức ở đó (thấy rõ trên hình chùm mô hình).
#
# ### Câu hỏi 1: Tăng kích thước tập train có giảm được bias không?
# **Không (về bản chất).** Bias² của bậc 1 là $\approx8.05\times10^{-4}$ ở cả $N=100$ và $N=1000$ (và $8.2\times10^{-4}$ ở $N=10$); bậc 2 giữ $\approx3.7\times10^{-4}$. Trên đồ thị "Bias² theo N" các đường gần như nằm ngang, và trên hình chùm mô hình, đường trung bình $\bar g$ của bậc 1 luôn là cùng một đường thẳng lệch khỏi $f$ dù $N$ là 10, 100 hay 1000. Bias được quyết định bởi **tập giả thuyết** $\mathcal H$: một đường thẳng không thể uốn cong theo $\sin(1+x^2)$ dù có bao nhiêu dữ liệu. Muốn giảm bias phải tăng độ phức tạp mô hình (hoặc đổi đặc trưng). Thay đổi nhỏ giữa $N=10$ và $N\ge100$ chỉ là hiệu ứng mẫu nhỏ/ước lượng Monte-Carlo, không phải bias thực sự giảm.
#
# ### Câu hỏi 2: Tăng kích thước tập train có giảm được variance không?
# **Có, và giảm rất mạnh.** Variance giảm xấp xỉ tỉ lệ $1/N$: bậc 1 từ $4.7\times10^{-4}\to3.8\times10^{-5}\to4.0\times10^{-6}$ (mỗi lần ×10 dữ liệu thì variance /≈10); bậc 9 từ $\sim10^{12}$ (không dùng được) xuống $1.4\times10^{-4}$ rồi $1.0\times10^{-5}$. Hệ quả thấy trên bảng lỗi và hình 3×3: với $N=1000$, ba mô hình từ $D_1,D_2,D_3$ trùng nhau, $E_{in}\approx\hat E_{out}$ và $\hat E_{out}\to\text{bias}^2+\sigma^2$.
#
# ### Kết luận
# $$\mathbb E_D[E_{out}] = \text{bias}^2 + \text{variance} + \sigma^2$$
# Thêm dữ liệu chỉ tác động lên **variance**; bias là giới hạn của tập giả thuyết, còn $\sigma^2=9\times10^{-4}$ là sàn không thể vượt qua. Vì vậy khi có ít dữ liệu nên chọn mô hình đơn giản; khi có nhiều dữ liệu có thể dùng mô hình phức tạp hơn để giảm bias mà không sợ variance.
#
