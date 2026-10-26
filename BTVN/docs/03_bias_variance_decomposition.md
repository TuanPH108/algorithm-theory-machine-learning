# 03 — Phân rã Bias–Variance (slides 18–57)

## 1. Khởi đầu: bias và variance của một ước lượng (estimator)

- Tham số tổng thể $\theta$ (ví dụ chiều cao trung bình). Từ mẫu ngẫu nhiên kích thước $m$ ta tính thống kê $\hat\theta$.
- **$\hat\theta$ là biến ngẫu nhiên**: mẫu khác ⇒ $\hat\theta$ khác. Gọi $\mu_{\hat\theta}=\mathbb E[\hat\theta]$.

$$\text{MSE}(\hat\theta)=\mathbb E[(\hat\theta-\theta)^2]$$

Chứng minh phân rã: đặt $a=\hat\theta-\mu_{\hat\theta}$, $b=\mu_{\hat\theta}-\theta$ (hằng số).

$$\mathbb E[(a+b)^2]=\mathbb E[a^2]+b^2+2b\,\underbrace{\mathbb E[a]}_{=0}
\;\Rightarrow\;
\boxed{\text{MSE}(\hat\theta)=\underbrace{\mathbb E[(\hat\theta-\mu_{\hat\theta})^2]}_{\operatorname{Var}(\hat\theta)}+\underbrace{(\mu_{\hat\theta}-\theta)^2}_{\text{Bias}^2(\hat\theta)}}$$

- **Bias**: xu hướng *có hệ thống* đánh giá cao/thấp $\theta$, tính trung bình trên mọi mẫu.
- **Variance**: mức phân tán của $\hat\theta$ quanh chính trung bình của nó.
- Hình ảnh bia bắn (slide 31): bias = tâm chùm đạn lệch khỏi hồng tâm; variance = chùm đạn tản rộng.

## 2. Chuyển sang học máy: ước lượng là cả một hàm

Ánh xạ: $\hat\theta \leftrightarrow h^{(D)}(x_0)$ (dự đoán tại điểm $x_0$ của mô hình fit trên $D$), $\theta \leftrightarrow f(x_0)$.

**Giả thuyết trung bình** (average hypothesis):
$$\bar h(x_0)=\mathbb E_D\big[h^{(D)}(x_0)\big]$$
— trung bình các mô hình fit trên **mọi** bộ dữ liệu có thể (cùng kích thước $n$).

### Trường hợp không nhiễu (slides 41–43)

$$\mathbb E_D\big[(h^{(D)}(x_0)-f(x_0))^2\big]=\underbrace{\mathbb E_D\big[(h^{(D)}(x_0)-\bar h(x_0))^2\big]}_{\text{variance}}+\underbrace{(\bar h(x_0)-f(x_0))^2}_{\text{bias}^2}$$

(số hạng chéo bằng 0 vì $\mathbb E_D[h^{(D)}-\bar h]=0$).

### Trường hợp có nhiễu (slides 44, 56)

$y_0=f(x_0)+\varepsilon$, $\varepsilon$ độc lập với $D$, $\mathbb E\varepsilon=0$, $\operatorname{Var}\varepsilon=\sigma^2$:

$$\boxed{\mathbb E_{D,\varepsilon}\big[(h^{(D)}(x_0)-y_0)^2\big]=\text{variance}+\text{bias}^2+\sigma^2}$$

Lấy kỳ vọng thêm theo $x_0\sim P(x)$ ⇒ **lỗi kiểm tra kỳ vọng tổng thể**:
$$\mathbb E_{x}\mathbb E_{D,\varepsilon}[\dots]=\mathbb E_x[\text{var}(x)]+\mathbb E_x[\text{bias}^2(x)]+\sigma^2$$

| Thành phần | Bản chất | Giảm bằng cách |
|---|---|---|
| Bias² | Sai số **hệ thống**: lớp $\mathcal H$ quá hạn chế, $\bar h$ không chạm được $f$ | Tăng độ linh hoạt của $\mathcal H$; thông tin "nội bộ" về dạng của $f$ |
| Variance | Độ **nhạy** của mô hình với bộ dữ liệu cụ thể | Thêm dữ liệu, giảm số chiều, regularization, mô hình đơn giản hơn |
| $\sigma^2$ | Nhiễu ngẫu nhiên vốn có | **Không giảm được** bằng mô hình |

## 3. Ba loại MSE lý thuyết (slide 45) — dễ nhầm

1. $\mathbb E_D[(h^{(D)}(x_0)-f(x_0))^2]$: **một điểm** $x_0$, trung bình trên **nhiều bộ** $D$.
2. $\mathbb E_x[(h(x)-f(x))^2]$: **một mô hình** $h$ (đã fit trên 1 bộ $D$), trung bình trên mọi $x$ ⇐ $\hat E_{out}$ trong bài tập ước lượng loại này (cộng $\sigma^2$).
3. $\mathbb E_x\mathbb E_D[\dots]$: trung bình trên **cả** $D$ và $x$ ⇐ đây mới là đại lượng phân rã được thành bias²+var.

⇒ Một con số $\hat E_{out}$ của **một** bộ dữ liệu **không** tách được thành bias và variance. Phải nhìn qua **nhiều bộ** $D$ (đó là lý do đề yêu cầu $D_1,D_2,D_3$).

## 4. Ví dụ trực giác của slides: $H_0$ vs $H_1$ trên $\sin(\pi x)$, $n=2$ (slides 33–39)

- $H_0$: hằng số $h(x)=b$. 500 lần fit ⇒ các đường nằm ngang gần nhau (variance thấp), trung bình $\bar h_0=0$ lệch xa $\sin(\pi x)$ (bias cao).
- $H_1$: đường thẳng qua 2 điểm. Độ dốc lung tung (variance cao), $\bar h_1$ bắt được xu hướng tăng (bias thấp hơn).
- Bài học: với dữ liệu rất ít, mô hình đơn giản có thể **thắng** dù bias cao, vì variance của mô hình phức tạp quá lớn.

## 5. Bức tranh tradeoff (slides 47–54)

- Mô hình phức tạp hơn ⇒ bias giảm, $E_{in}\to0$, nhưng variance tăng ⇒ nguy cơ $E_{out}\gg0$.
- Mô hình đơn giản hơn ⇒ variance thấp, $E_{in}\approx E_{out}$, nhưng cả hai có thể đều lớn.
- Trong thực tế không có $\bar h$ và $f$ ⇒ không đo trực tiếp được bias/variance; mô phỏng (như bài này) là cách duy nhất để "nhìn thấy" chúng.

## 6. Cách ước lượng bằng mô phỏng (dùng cho phần mở rộng)

Cho $R$ bộ dữ liệu $D^{(1)},\dots,D^{(R)}$, lưới điểm $x_1,\dots,x_G$ trong [0,1]:

$$\bar h(x_g)\approx\frac1R\sum_{r}h^{(r)}(x_g),\quad
\widehat{\text{bias}^2}=\frac1G\sum_g(\bar h(x_g)-f(x_g))^2,\quad
\widehat{\text{var}}=\frac1G\sum_g\frac1R\sum_r(h^{(r)}(x_g)-\bar h(x_g))^2$$

Với $R=3$ (đúng đề) ước lượng rất thô; $R\ge100$ cho số ổn định. Kiểm tra: $\overline{\hat E_{out}}\approx\widehat{\text{bias}^2}+\widehat{\text{var}}+\sigma^2$.

## Áp dụng vào bài tập — đọc hình slide 68

- Trong một ô (một bậc $d$): **độ tản giữa 3 đường** của $D_1,D_2,D_3$ ≈ **variance**; **độ lệch của "đường ở giữa" so với $f$** ≈ **bias**.
- Bậc 1–2: 3 đường gần nhau nhưng cùng lệch khỏi $f$ (cùng cắt sai ở hai đầu, không bắt được đỉnh) ⇒ bias cao, variance thấp.
- Bậc 3–5: 3 đường bám $f$ và gần nhau ⇒ cân bằng tốt.
- Bậc 7–9: 3 đường dao động mạnh, khác nhau rõ, đặc biệt ở hai đầu ⇒ variance cao; trung bình của chúng có thể gần $f$ ⇒ bias thấp.
- Câu chốt: "$E_{in}=0$" của bậc 9 **không** có nghĩa bias = 0 hay mô hình tốt.
