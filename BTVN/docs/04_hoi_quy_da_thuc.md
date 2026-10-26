# 04 — Hồi quy đa thức (slides 60–65)

## 1. Mô hình

$$h_d(x)=b_0+b_1x+b_2x^2+\dots+b_dx^d$$

- **Phi tuyến theo $x$** nhưng **tuyến tính theo tham số** $b$ ⇒ vẫn là hồi quy tuyến tính trên đặc trưng $(1,x,x^2,\dots,x^d)$.
- Số tham số $p=d+1$. Các lớp lồng nhau: $\mathcal H_1\subset\mathcal H_2\subset\dots\subset\mathcal H_9$ ⇒ bậc cao hơn luôn có $E_{in}$ **không lớn hơn** bậc thấp (trên cùng tập train).

Tên gọi: bậc 1 linear, 2 quadratic, 3 cubic, 4 quartic, 5 quintic, …, 9 nonic.

## 2. Fit bằng bình phương tối thiểu

Ma trận thiết kế (Vandermonde) $X\in\mathbb R^{n\times(d+1)}$, $X_{ij}=x_i^{\,j}$:

$$\hat b=\arg\min_b\frac1n\|Xb-y\|^2,\qquad X^TX\hat b=X^Ty\ \ (\text{Normal Equations})$$

- Nếu $n>d+1$ và các $x_i$ phân biệt ⇒ nghiệm duy nhất.
- Nếu $n=d+1$ (10 điểm, bậc 9) ⇒ $X$ vuông khả nghịch ⇒ đa thức **đi qua đúng mọi điểm** (nội suy Lagrange) ⇒ $E_{in}=0$, mô hình "ghi nhớ" cả nhiễu.
- Nếu $n<d+1$ ⇒ vô số nghiệm (không xảy ra trong bài).

## 3. Vấn đề số học (ill-conditioning)

- Cột $x^j$ trên [0,1] rất giống nhau khi $j$ lớn ($x^8$ và $x^9$ gần như trùng) ⇒ $X$ gần suy biến.
- Số điều kiện cond($X$) với 10 điểm ngẫu nhiên, bậc 9 ≈ $10^{10}$; Normal Equations bình phương nó lên ≈ $10^{20}$ > độ chính xác double ($\approx10^{16}$) ⇒ **không giải tay bằng $(X^TX)^{-1}$**.
- `np.polyfit` dùng `lstsq` (SVD) ⇒ đủ ổn định cho bài. Muốn ổn định hơn: `np.polynomial.Polynomial.fit` (tự scale $x$ về $[-1,1]$).

## 4. Vì sao bậc cao dao động mạnh ở hai đầu

- **Hiện tượng Runge**: đa thức nội suy bậc cao dao động mạnh gần biên khoảng.
- **Ngoại suy**: 10 điểm $x\sim U[0,1]$ hiếm khi phủ sát 0 và 1 (kỳ vọng khoảng hở ở mỗi đầu ≈ $1/(n+1)\approx0.09$). Ngoài $[\min x_i,\max x_i]$, đa thức bậc 9 tăng/giảm như $x^9$ với hệ số khổng lồ ⇒ giá trị có thể lên hàng chục, hàng trăm.
- Test point rơi vào vùng này ⇒ $\hat E_{out}$ bùng nổ. Đây là biểu hiện cực đoan của **variance cao**.
- Khi $n$ lớn (100, 1000): dữ liệu phủ dày cả hai đầu, $p\ll n$ ⇒ dao động biến mất.

## 5. Bậc nào "đúng" với $f(x)=\sin(1+x^2)$?

- Khai triển Taylor quanh tâm khoảng cho thấy $f$ trơn, xấp xỉ rất tốt bởi đa thức bậc thấp trên [0,1].
- Pilot (n=1000): bias² bậc 1 ≈ 8·10⁻⁴, bậc 2 ≈ 3.7·10⁻⁴, bậc 3 ≈ 6·10⁻⁶, bậc ≥5 ≈ 10⁻⁸ ⇒ từ bậc 3 trở lên bias² đã nhỏ hơn $\sigma^2$ cả trăm lần.
- Không bậc nào cho bias **đúng bằng 0** ($f$ không phải đa thức) — đúng như slide 69: lớp giả thuyết hạn chế luôn mang bias.

## Áp dụng vào bài tập

```python
c = np.polyfit(x_train, y_train, d)      # hệ số, bậc cao -> thấp
y_hat = np.polyval(c, x)                 # dự đoán
```

- Bậc 9 với n=10 có thể phát `RankWarning` — bình thường; có thể tắt bằng `warnings` nhưng nên ghi chú.
- Khi vẽ, đánh giá đa thức trên `np.linspace(0, 1, 300)` (không chỉ trên các điểm dữ liệu) để thấy dao động giữa các điểm.
