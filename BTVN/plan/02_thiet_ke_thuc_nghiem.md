# 02 — Thiết kế thực nghiệm

## 1. Sinh dữ liệu

```
f(x)  = sin(1 + x^2)
x     ~ Uniform[0, 1]                (np.random.uniform(0, 1, n))
eps   ~ Normal(0, 0.03)  # scale = σ  (np.random.normal(0, 0.03, n))
y     = f(x) + eps
```

- Mỗi bộ $D_k$ = (train: n điểm, test: n điểm), sinh **độc lập**.
- Thứ tự sinh cố định: với mỗi n → với mỗi k=1..3 → train rồi test. Giữ nguyên thứ tự này để kết quả tái lập.
- Lưu ý: `np.random.normal(loc, scale)` nhận **độ lệch chuẩn** → truyền `0.03`, **không** phải `0.0009`.
- Hình dạng của $f$: $1+x^2\in[1,2]$, $\sin$ đạt cực đại tại $\pi/2\approx1.571$ ⇒ $x\approx0.756$, $f_{max}=1$; $f(0)=\sin1\approx0.841$, $f(1)=\sin2\approx0.909$. Hàm tăng rồi giảm (1 cực đại) ⇒ bậc 1 không bẻ cong được, bậc 2 (parabol đối xứng) không khớp độ bất đối xứng ⇒ underfit; bậc 3 trở lên đủ linh hoạt.

## 2. Mô hình

- Lớp giả thuyết $\mathcal H_d=\{h(x)=\sum_{j=0}^d b_j x^j\}$, $d=1..9$.
- Thuật toán học: OLS (tối thiểu MSE trên train). Công cụ đề xuất: `np.polyfit(x, y, d)` + `np.polyval` (lý do ở [03](03_pros_cons_lua_chon.md)).
- Với n=10, d=9: số tham số = số điểm ⇒ đa thức **nội suy** ⇒ $E_{in}=0$ (sai số máy).

## 3. Độ đo

$$E_{in}=\frac1{n}\sum_{(x_i,y_i)\in D_{train}}(h(x_i)-y_i)^2,\qquad
\hat E_{out}=\frac1{m}\sum_{(x_j,y_j)\in D_{test}}(h(x_j)-y_j)^2$$

- Đây là MSE (giống slide 66: $E_{in}$ bậc 1 ≈ 0.00147 cùng bậc độ lớn với $\sigma^2=0.0009$). Không dùng RMSE, không dùng tổng.
- Hiển thị 5 chữ số thập phân như slide; với giá trị rất lớn (bùng nổ) dùng định dạng khoa học.

### Đo bias²/variance (phần mở rộng, khuyến nghị)

Với mỗi n và d, lặp $R=200$ (hoặc 500) bộ train, đánh giá trên lưới $x_g$ (200 điểm):

$$\bar h(x_g)=\frac1R\sum_r h^{(r)}(x_g),\quad
\text{bias}^2=\overline{(\bar h(x_g)-f(x_g))^2},\quad
\text{var}=\overline{\frac1R\sum_r (h^{(r)}(x_g)-\bar h(x_g))^2}$$

- Lưới $x_g$ nên lấy trong [0,1] (đúng miền phân phối của x). Với n=10, bậc cao: mean bị chi phối bởi vài fit bùng nổ ⇒ báo cáo thêm **median** hoặc dùng trimmed mean và nói rõ.
- Kiểm tra chéo: $\mathbb E[\hat E_{out}]\approx \text{bias}^2+\text{var}+\sigma^2$.

## 4. Hình vẽ

| Hình | Bố cục | Nội dung | Mỗi n |
|---|---|---|---|
| H0 (slide 60) | 1 ô | $f(x)$ (xám), điểm train (o xanh), test (* đỏ) của $D_1$ | n=10 (n lớn: tùy chọn, alpha thấp) |
| H1 (slide 65) | 3×3 | Mỗi ô 1 bậc: $f$, điểm train/test, đường fit; tiêu đề ghi $E_{in}$, $\hat E_{out}$ | cho $D_1$ (hoặc cả 3) |
| H2 (slide 68) | 3×3 | Mỗi ô 1 bậc: $f$ (xám, nét đứt), 3 đường fit của $D_1,D_2,D_3$ (3 màu cố định) | **bắt buộc**, mỗi n |
| H3 (slide 71) | 1×3 | $E_{in}$, $\hat E_{out}$ theo bậc, mỗi ô 1 giá trị n, trục y log | tổng hợp |
| H4 (mở rộng) | 1×3 | bias², var, $\sigma^2$ theo bậc cho từng n (log-scale) | tổng hợp |

Quy ước chung: `sharex`, `sharey`, `ylim=(0.7, 1.1)`, `xlim=(0,1)`, lưới vẽ đường fit `np.linspace(0,1,300)`, legend 1 lần cho cả figure, `suptitle` ghi n.

## 5. Cấu trúc notebook đề xuất

```
0. Tiêu đề, họ tên, MSHV, mô tả bài toán
1. Import + seed (đúng mẫu đề)
2. Hàm tiện ích: f, generate_dataset, fit_poly, mse, plot_grid_*
3. Thực nghiệm n=10
   3.1 Sinh D1, D2, D3 + hình H0
   3.2 Fit 9 bậc, bảng E_in / Ê_out (từng bộ + bảng gộp)
   3.3 Hình H1 cho D1 (slide 65)
   3.4 Hình H2 (slide 68)
   3.5 Nhận xét bias/variance
4. Thực nghiệm n=100  (lặp 3.1–3.5 bằng cùng hàm)
5. Thực nghiệm n=1000 (lặp 3.1–3.5)
6. So sánh tổng hợp: H3, (mở rộng) H4 + bảng bias²/var
7. Trả lời 2 câu hỏi
8. Kết luận
```

Nguyên tắc: viết **một hàm** `run_experiment(n)` trả về bảng + coefficients, gọi 3 lần ⇒ tránh copy–paste và đảm bảo 3 kịch bản được xử lý giống hệt nhau.
