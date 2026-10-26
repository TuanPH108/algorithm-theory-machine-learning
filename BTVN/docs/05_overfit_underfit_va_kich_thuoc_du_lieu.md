# 05 — Overfitting, underfitting và ảnh hưởng của kích thước dữ liệu (slides 64–71)

## 1. Định nghĩa

| Hiện tượng | Dấu hiệu | Nguyên nhân chính | Ví dụ slide 66 (n=10) |
|---|---|---|---|
| **Underfit** | $E_{in}$ và $E_{out}$ đều lớn, $E_{in}\approx E_{out}$ | Bias cao — mô hình thiếu năng lực | Bậc 1 (0.00147 / 0.00215), bậc 2 |
| **"Okayfit"** | Cả hai nhỏ, chênh lệch nhỏ | Cân bằng | Bậc 3–5 ($\hat E_{out}$ 0.00081–0.00116) |
| **Overfit** | $E_{in}$ rất nhỏ nhưng $E_{out}$ lớn | Variance cao — mô hình fit cả nhiễu | Bậc 9 (0.00000 / 0.00231) |

Định nghĩa overfit theo slides 67, 69: chọn mô hình có $E_{in}$ nhỏ hơn nhưng lại có $E_{out}$ lớn hơn — **$E_{in}$ không còn là chỉ báo tốt** cho hiệu năng thật.

## 2. Đường cong lỗi theo độ phức tạp (slide 71)

```
Error
 |\
 | \        E_out (hình chữ U)
 |  \__        __/
 |     \______/
 |  \
 |   \____ E_in (giảm đơn điệu)
 |        \__________
 +-----------------------> độ phức tạp (bậc d)
  underfit   okay   overfit
```

- $E_{in}$ giảm đơn điệu (lớp lồng nhau).
- $E_{out}$ giảm (bias giảm) rồi tăng (variance tăng) ⇒ điểm cực tiểu = độ phức tạp tối ưu.

## 3. Ảnh hưởng của số lượng dữ liệu $n$ (slide 70) — cốt lõi của câu hỏi 6

### Variance phụ thuộc mạnh vào $n$
Với OLS có $p$ tham số, xấp xỉ $\mathbb E_x[\text{var}]\approx\sigma^2\,p/n$.
- $n$ tăng 10 lần ⇒ variance giảm ~10 lần.
- Trực quan: các đường fit từ các bộ $D$ khác nhau **hội tụ về một đường** ($\bar h$).

### Bias (gần như) không phụ thuộc vào $n$
- Bias đo $\bar h-f$. Khi $n\to\infty$, mỗi $h^{(D)}$ hội tụ về **xấp xỉ tốt nhất trong lớp** $h^*_{\mathcal H}$ (theo MSE dưới $P(x)$), nên $\bar h\to h^*_{\mathcal H}$ và bias $\to h^*_{\mathcal H}-f$ — một hằng số chỉ phụ thuộc $\mathcal H$ và $f$.
- "Tăng số điểm học không thể sửa một lớp giả thuyết quá đơn giản" (slide 70). Đường thẳng vẫn là đường thẳng dù có 1 triệu điểm.
- Sắc thái: khi $n$ rất nhỏ so với $p$, $\bar h$ có thể khác $h^*_{\mathcal H}$ (và ước lượng thực nghiệm của $\bar h$ bị nhiễu bởi các fit bất ổn), nên bias đo được có thể thay đổi đôi chút theo $n$. Đây là hiệu ứng thứ cấp, không phải cơ chế giảm bias.

### Hệ quả cho việc chọn mô hình
- $n$ nhỏ: variance chi phối ⇒ mô hình **đơn giản** thắng (bậc tối ưu thấp).
- $n$ lớn: variance nhỏ với mọi bậc ⇒ mô hình **linh hoạt** hưởng lợi, bậc tối ưu dịch sang phải; khoảng cách $E_{out}-E_{in}$ co lại.
- Giới hạn dưới vẫn là $\sigma^2$ + bias² của lớp.

### Bảng tóm tắt kỳ vọng (dựa trên pilot)

| | n=10 | n=100 | n=1000 |
|---|---|---|---|
| Bias² bậc 1 | ~8e-4 | ~8e-4 | ~8e-4 |
| Var bậc 1 | ~5e-4 | ~4e-5 | ~4e-6 |
| $\hat E_{out}$ bậc 3 | ~0.0015 (dao động lớn) | ~0.00093 | ~0.00090 |
| $\hat E_{out}$ bậc 9 | rất lớn / bất ổn | ~0.0010 | ~0.00090 |

## 4. Các cách giảm variance khác (slide 54)

- Thêm dữ liệu huấn luyện.
- Giảm số chiều (PCA, chọn đặc trưng).
- Regularization (Ridge, Lasso) — ràng buộc độ lớn hệ số, giảm dao động bậc cao; đổi lại thêm một chút bias.

## Áp dụng vào bài tập — khung trả lời 2 câu hỏi

**Câu 1: Tăng kích thước tập train có giảm bias không?** → Không (đáng kể).
Dẫn chứng nên dùng: (i) hình slide 68 ở n=1000, ô bậc 1: 3 đường trùng nhau nhưng vẫn lệch $f$ ở hai đầu và đỉnh; (ii) $\hat E_{out}$ bậc 1 ≈ 0.0017 ở cả n=100 và 1000, cao hơn $\sigma^2$ một lượng ≈ bias²; (iii) (mở rộng) bảng bias² gần như hằng theo n.

**Câu 2: Tăng kích thước tập train có giảm variance không?** → Có.
Dẫn chứng: (i) độ tản của 3 đường trong mọi ô giảm rõ từ n=10 → 100 → 1000; (ii) std của $\hat E_{out}$ giữa $D_1,D_2,D_3$ giảm; (iii) $\hat E_{out}-E_{in}$ thu hẹp, bậc 9 hết overfit; (iv) (mở rộng) var giảm ~1/n.
