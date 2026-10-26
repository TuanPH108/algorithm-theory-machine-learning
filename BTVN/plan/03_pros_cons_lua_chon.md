# 03 — Các quyết định kỹ thuật: ưu / nhược điểm

Ký hiệu: ✅ khuyến nghị.

## Q1. Công cụ fit đa thức

| Phương án | Ưu | Nhược |
|---|---|---|
| ✅ `np.polyfit` + `np.polyval` | Ngắn gọn, chỉ cần numpy; giải bằng `lstsq` (SVD) nên ổn hơn giải Normal Equations; có cảnh báo `RankWarning` khi ill-conditioned | Hệ số theo cơ sở $x^j$ thô, ill-conditioned ở bậc cao (cond ≈ 4·10¹⁰ với n=10, d=9). Với x∈[0,1] và d≤9 vẫn chấp nhận được |
| `np.polynomial.Polynomial.fit` | Tự scale miền x về [-1,1] ⇒ điều kiện số tốt hơn nhiều | Hệ số nằm trong miền đã scale (phải gọi `.convert()` nếu muốn in hệ số); ít quen thuộc |
| `sklearn` `PolynomialFeatures` + `LinearRegression` | Đúng "phong cách ML", dễ mở rộng sang Ridge | Dài dòng hơn; cùng vấn đề điều kiện; thêm dependency |
| Tự giải Normal Equations $(X^TX)^{-1}X^Ty$ | Thể hiện hiểu lý thuyết | Bình phương số điều kiện (~10²⁰) ⇒ **sai số số học lớn** ở bậc 8–9. Không nên dùng làm kết quả chính |

**Chọn:** `np.polyfit`. Có thể thêm 1 cell kiểm chứng so với `Polynomial.fit` (sai khác $E_{in}$ không đáng kể) để chứng tỏ kết quả không phải do lỗi số học.

## Q2. Cách sinh số ngẫu nhiên

| Phương án | Ưu | Nhược |
|---|---|---|
| ✅ API toàn cục `np.random.uniform/normal` | **Khớp đúng mẫu seed của đề** (`np.random.seed`) ⇒ tái lập được | API "legacy" |
| `rng = np.random.default_rng(seed)` | API hiện đại, độc lập luồng | `np.random.seed` **không** ảnh hưởng `default_rng()` không truyền seed ⇒ vi phạm tinh thần đề nếu không truyền `seed` vào |
| `random.uniform` / `random.gauss` | Có seed bởi `random.seed` | Chậm, phải vòng lặp; không vector hóa |

## Q3. Hình slide 68: vẽ điểm dữ liệu hay không

| Phương án | Ưu | Nhược |
|---|---|---|
| ✅ Chỉ vẽ đường fit + $f(x)$ | Giống slide, nhìn rõ độ "dao động" giữa các bộ (= variance) | Không thấy dữ liệu |
| Vẽ cả điểm | Thấy nguyên nhân dao động | Rối, đặc biệt n=1000 (3000 điểm/ô) |

Bù lại: hình H1 (slide 65) đã có điểm dữ liệu.

## Q4. Trục y

| Phương án | Ưu | Nhược |
|---|---|---|
| ✅ Cố định `ylim=(0.7,1.1)` | So sánh trực tiếp với slide và giữa các ô/các n | Đường bùng nổ bị cắt — phải **ghi chú** trong nhận xét |
| Auto-scale | Thấy hết biên độ bùng nổ | Một ô bậc 9 có thể có trục y tới ±10³ ⇒ mọi đường khác thành đường thẳng, mất thông tin |

Có thể thêm 1 hình phụ không cắt trục để minh họa mức độ bùng nổ.

## Q5. Số bộ dữ liệu dùng để kết luận bias/variance

| Phương án | Ưu | Nhược |
|---|---|---|
| Chỉ 3 bộ (đúng đề) | Đúng yêu cầu, dễ trình bày | 3 mẫu ⇒ ước lượng variance rất thô; kết luận dễ bị phản bác do may/rủi của seed |
| ✅ 3 bộ (bắt buộc) **+** R=200 bộ (mở rộng) | Có số bias², var định lượng, kiểm chứng $\mathbb E[\hat E_{out}]\approx$ bias²+var+σ²; trả lời R7 chắc chắn | Thêm code/thời gian chạy (vài giây với numpy); phải tách rõ "phần bắt buộc" và "phần mở rộng" |

## Q6. Kích thước test

| Phương án | Ưu | Nhược |
|---|---|---|
| ✅ Đúng đề: test = train (10/100/1000) | Đúng yêu cầu | $\hat E_{out}$ với 10 điểm test có phương sai lớn (chính nó là ước lượng nhiễu) |
| Thêm test lớn (ví dụ 10⁴) để "gần đúng" $E_{out}$ | Tách được nhiễu do test nhỏ khỏi hiệu ứng mô hình | Ngoài phạm vi đề; chỉ nên là ghi chú |

## Q7. Tổ chức code

| Phương án | Ưu | Nhược |
|---|---|---|
| ✅ Hàm `run_experiment(n)` dùng chung | Nhất quán giữa 3 kịch bản, ít lỗi | Cần thiết kế trước giá trị trả về |
| Copy–paste 3 lần | Nhanh lúc đầu | Dễ lệch (sửa 1 chỗ quên chỗ khác), notebook dài |

## Q8. Bảng kết quả

| Phương án | Ưu | Nhược |
|---|---|---|
| ✅ 1 DataFrame gộp: Degree × {D1,D2,D3} × {E_in, Ê_out} (MultiIndex cột) | Nhìn được variance của $\hat E_{out}$ giữa các bộ ngay trên bảng | Hơi rộng |
| 3 bảng riêng giống hệt slide 66 | Bám sát slide | Khó so sánh |

Làm cả hai là hợp lý: 3 bảng riêng (bám đề) + 1 bảng tổng hợp có thêm cột mean/std của $\hat E_{out}$ qua 3 bộ.
