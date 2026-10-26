# 01 — Khung lý thuyết của học có giám sát (slides 2–17)

## 1. Các thành phần

| Thành phần | Ký hiệu | Ý nghĩa | Trong bài tập |
|---|---|---|---|
| Hàm mục tiêu (target function) | $f:\mathcal X\to\mathcal Y$ | Quan hệ thật, **chưa biết** trong thực tế | $f(x)=\sin(1+x^2)$ — ta *biết* vì đây là mô phỏng |
| Dữ liệu học | $D=\{(x_i,y_i)\}_{i=1}^n$ | Mẫu quan sát được | Tập train $n$ điểm |
| Tập giả thuyết (hypothesis set) | $\mathcal H=\{h_1,h_2,\dots\}$ | Các hàm ứng viên mà thuật toán được phép chọn | $\mathcal H_d$: đa thức bậc $d$ |
| Thuật toán học | $\mathcal A$ | Quy tắc chọn $h$ từ $\mathcal H$ dựa vào $D$ | Bình phương tối thiểu (OLS) |
| Mô hình cuối | $\hat f = h^{(D)}$ | Giả thuyết được chọn | Đa thức đã fit |
| Mô hình học (learning model) | $(\mathcal H,\mathcal A)$ | Cặp tập giả thuyết + thuật toán | (đa thức bậc d, OLS) |

Điểm cốt lõi: $\hat f$ **phụ thuộc vào $D$**. Đổi bộ dữ liệu ⇒ đổi $\hat f$. Đây là gốc rễ của khái niệm *variance*.

## 2. Hai loại dự đoán (slides 8–9)

- Dự đoán trên điểm **đã thấy** $x_i\in D_{in}$: khía cạnh *ghi nhớ* (memorizing).
- Dự đoán trên điểm **chưa thấy** $x_0\notin D_{in}$: khía cạnh *tổng quát hóa* (generalization) — đây là điều ta thực sự quan tâm.
- Fit tốt dữ liệu đã thấy là điều kiện **cần**, không **đủ**. Có thể fit hoàn hảo (bậc 9) mà dự đoán dữ liệu mới rất tệ.

## 3. Góc nhìn xác suất (slides 14–16)

- Giả định các $x_i$ là mẫu **i.i.d.** từ phân phối $P(x)$ trên $\mathcal X$. Giả định này là cầu nối giữa lỗi trên mẫu và lỗi trên toàn bộ không gian.
- Bài tập: $P(x)=\text{Uniform}[0,1]$.

### Mục tiêu có nhiễu (noisy target)

$$y=f(x)+\varepsilon,\qquad \mathbb E[\varepsilon]=0,\ \operatorname{Var}(\varepsilon)=\sigma^2$$

- Cùng một $x$ có thể cho các $y$ khác nhau ⇒ thực chất ta học phân phối có điều kiện $P(y\mid x)$; dữ liệu tuân theo $P(x,y)=P(x)P(y\mid x)$.
- Bài tập: $\varepsilon\sim\mathcal N(0,\sigma^2)$ với **$\sigma=0.03$ (độ lệch chuẩn)** ⇒ $\sigma^2=9\times10^{-4}$.
- Hệ quả quan trọng: **không mô hình nào** có lỗi kỳ vọng trên dữ liệu mới nhỏ hơn $\sigma^2$ — ngay cả $h=f$ cũng có $\mathbb E[(f(x)-y)^2]=\sigma^2$.

## 4. Mục tiêu học (slide 15, 32)

$$E_{out}(\hat f)\approx 0 \iff \underbrace{E_{in}(\hat f)\approx 0}_{\text{thực hành}}\ \text{và}\ \underbrace{E_{out}(\hat f)\approx E_{in}(\hat f)}_{\text{lý thuyết / tổng quát hóa}}$$

- Mô hình đơn giản: dễ đạt vế phải (Ein ≈ Eout) nhưng khó đạt vế trái (Ein lớn).
- Mô hình phức tạp: dễ đạt vế trái (Ein ≈ 0) nhưng khó đạt vế phải.
- ⇒ Đây chính là bias–variance tradeoff dưới góc nhìn $E_{in}/E_{out}$.

## Áp dụng vào bài tập

- Vì là mô phỏng nên ta biết $f$ và có thể **vẽ $f$** lên hình để thấy trực tiếp bias (độ lệch có hệ thống của đường fit so với $f$).
- Ta cũng sinh được **nhiều bộ $D$** từ cùng phân phối ⇒ quan sát được $\hat f$ thay đổi theo $D$ (variance) — điều không làm được với dữ liệu thật.
- Với $n$ lớn, kỳ vọng $\hat E_{out}$ của mô hình tốt sẽ chạm "sàn" $\sigma^2=0.0009$.
