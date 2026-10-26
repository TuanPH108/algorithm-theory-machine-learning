# 06 — Công cụ: numpy, matplotlib, pandas

Các đoạn dưới đây là **mảnh ghép tham khảo**, không phải lời giải hoàn chỉnh.

## 1. Môi trường

```bash
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib pandas jupyter
.venv/bin/jupyter notebook
```

## 2. Seed — đúng mẫu đề

```python
import numpy as np
import random

seed = 12345678          # <-- thay bằng MSHV (chỉ gồm chữ số)
np.random.seed(seed)
random.seed(seed)
```

`np.random.seed` chỉ điều khiển **global RandomState**. Các lệnh sau bị ảnh hưởng: `np.random.uniform`, `np.random.normal`, `np.random.rand`, …
Các lệnh sau **không** bị ảnh hưởng: `np.random.default_rng()` (không truyền seed), `np.random.Generator(...)`.

## 3. Sinh dữ liệu

```python
SIGMA = 0.03                       # độ lệch chuẩn

def f(x):
    return np.sin(1 + x**2)

def make_split(n):
    x = np.random.uniform(0, 1, n)
    y = f(x) + np.random.normal(0, SIGMA, n)   # scale = σ, KHÔNG phải σ²
    return x, y
```

Kiểm tra nhanh: `np.std(y - f(x))` ≈ 0.03 khi n lớn.

## 4. Fit và đánh giá

```python
c = np.polyfit(x_tr, y_tr, deg)             # hệ số: bậc cao -> thấp
mse = lambda c, x, y: np.mean((np.polyval(c, x) - y) ** 2)
```

Tắt cảnh báo điều kiện kém (ghi chú lý do trong notebook):
```python
import warnings
warnings.simplefilter("ignore", np.exceptions.RankWarning)   # numpy >= 1.25
# numpy cũ: np.RankWarning
```

## 5. Lưới 3×3

```python
fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
xg = np.linspace(0, 1, 300)
for deg, ax in zip(range(1, 10), axes.flat):
    ax.plot(xg, f(xg), color="gray", lw=2, label="f(x)")
    # ... vẽ các đường fit
    ax.set_title(f"degree {deg} polynomial")
    ax.set_ylim(0.7, 1.1); ax.set_xlim(0, 1)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4)
fig.suptitle(...); fig.tight_layout()
```

Gợi ý trình bày:
- Màu cố định cho $D_1,D_2,D_3$ (ví dụ `["tab:green", "tab:purple", "tab:orange"]`) trong mọi hình.
- Điểm train: `marker="o", facecolors="none"`; test: `marker="*"` (giống slide 60).
- n=1000: `s=4, alpha=0.3` cho scatter.
- Đồ thị lỗi theo bậc: `ax.set_yscale("log")`; vẽ đường ngang `ax.axhline(SIGMA**2, ls=":")` làm mốc sàn nhiễu.

## 6. Bảng kết quả

```python
import pandas as pd
df = pd.DataFrame(rows, columns=["Degree", "E_in", "E_out_hat"]).set_index("Degree")
df.style.format("{:.5f}")      # 5 chữ số như slide 66
```

Bảng gộp 3 bộ: dùng `pd.concat({ "D1": df1, "D2": df2, "D3": df3 }, axis=1)` ⇒ cột MultiIndex. Với giá trị bùng nổ, dùng `"{:.5g}"` hoặc `"{:.3e}"` để không mất thông tin.

## 7. Kiểm tra trước khi nộp

- Kernel → Restart & Run All. Chạy 2 lần, bảng phải giống hệt nhau.
- Không có cell nào phụ thuộc thứ tự chạy thủ công.
- Đổi tên file `BT1_<MSHV>.ipynb`.
