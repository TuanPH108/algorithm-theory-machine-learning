# 02 — Hàm lỗi, $E_{in}$, $E_{out}$, $\hat E_{out}$ (slides 10–15, 59)

## 1. Lỗi từng điểm (loss) và lỗi tổng (cost / risk)

| Lỗi từng điểm | Công thức | Dùng cho |
|---|---|---|
| Squared error | $(\hat y_i-y_i)^2$ | **Hồi quy — dùng trong bài** |
| Absolute error | $\lvert\hat y_i-y_i\rvert$ | Hồi quy bền với outlier |
| Misclassification | $\mathbb 1[\hat y_i\ne y_i]$ | Phân lớp |

Lỗi tổng = trung bình lỗi từng điểm: $E=\frac1n\sum_i \text{err}_i$. Với squared error đây là **MSE**.

## 2. Ba đại lượng cần phân biệt

| Đại lượng | Định nghĩa | Tính được? |
|---|---|---|
| $E_{in}(h)$ | $\frac1n\sum_{(x_i,y_i)\in D_{train}}(h(x_i)-y_i)^2$ | Có |
| $E_{out}(h)$ | $\mathbb E_{(x,y)\sim P}\big[(h(x)-y)^2\big]$ — kỳ vọng trên toàn bộ không gian | **Không** (lý thuyết) |
| $\hat E_{out}(h)$ | $\frac1m\sum_{(x_j,y_j)\in D_{test}}(h(x_j)-y_j)^2$ | Có — ước lượng của $E_{out}$ |

- $D_{test}$ phải **độc lập** với $D_{train}$ và lấy từ cùng phân phối ⇒ $\hat E_{out}$ là ước lượng **không chệch** của $E_{out}(h)$.
- $E_{in}$ là ước lượng **chệch lạc quan** của $E_{out}$: vì $h$ được chọn để làm nhỏ chính $E_{in}$.
- $\hat E_{out}$ có phương sai $\approx \operatorname{Var}[(h(x)-y)^2]/m$ ⇒ với $m=10$ điểm test, ước lượng **rất nhiễu**. Với $m=1000$ thì đáng tin hơn nhiều.

## 3. Giá trị tham chiếu hữu ích (cho OLS, mô hình đủ linh hoạt)

Với hồi quy tuyến tính theo tham số có $p=d+1$ tham số, dữ liệu $n$ điểm, nhiễu $\sigma^2$, nếu lớp giả thuyết chứa (gần đúng) $f$:

$$\mathbb E[E_{in}]\approx\sigma^2\Big(1-\frac{p}{n}\Big),\qquad \mathbb E[E_{out}]\approx\sigma^2\Big(1+\frac{p}{n}\Big)$$

(công thức thứ hai là xấp xỉ khi $n\gg p$). Hệ quả:
- Khoảng cách $E_{out}-E_{in}\approx 2\sigma^2 p/n$: tăng theo độ phức tạp $p$, giảm theo $n$.
- $p=n$ (bậc 9, $n=10$) ⇒ $E_{in}=0$; xấp xỉ cho $E_{out}$ không còn đúng — thực tế $E_{out}$ có thể rất lớn.
- Nếu lớp giả thuyết **thiếu** linh hoạt (bậc 1, 2), cộng thêm số hạng bias² vào cả hai.

Ví dụ $n=1000$: bậc 3 kỳ vọng $E_{in}\approx 0.0009\cdot(1-0.004)\approx0.000896$, $E_{out}\approx0.000904$ — khớp với pilot.

## 4. Thang đo và cách đọc số

- $\sigma^2=0.0009$. Mọi $\hat E_{out}$ "tốt" sẽ nằm quanh 0.0009–0.0015.
- Bậc 1 có bias² ≈ 0.0008 ⇒ $E_{out}$ ≈ 0.0017 dù $n$ lớn đến đâu.
- Khi so sánh nhiều bậc có lỗi chênh nhau hàng chục lần trở lên, dùng **trục log**.

## Áp dụng vào bài tập

- Tính đúng MSE (chia cho số điểm). Không dùng RMSE, không dùng tổng bình phương, không nhân ½.
- $E_{in}$ dùng tập train; $\hat E_{out}$ dùng tập test **của cùng bộ $D_k$**.
- Khi nhận xét n=10, nhớ rằng $\hat E_{out}$ dựa trên 10 điểm nên dao động lớn giữa $D_1,D_2,D_3$ — một phần dao động đó đến từ chính test set chứ không chỉ từ mô hình.
