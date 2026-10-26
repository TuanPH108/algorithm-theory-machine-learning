# 08 — Xác định và kiểm định tính ngẫu nhiên

Script: [pilot/rand_check.py](pilot/rand_check.py) (seed minh họa 12345678, không phải MSHV).

"Xác định độ ngẫu nhiên" trong bài này gồm 4 câu hỏi khác nhau:

| # | Câu hỏi | Công cụ |
|---|---|---|
| 1 | Cái gì là ngẫu nhiên? | Liệt kê nguồn ngẫu nhiên |
| 2 | Ngẫu nhiên có **đúng phân phối** đề yêu cầu không? | Kiểm định thống kê + hình |
| 3 | Các phần có **độc lập** với nhau không? | Tương quan, so sánh phân phối |
| 4 | Ngẫu nhiên **ảnh hưởng kết quả bao nhiêu**? | Lặp lại nhiều lần, đo độ dao động |

## 1. Nguồn ngẫu nhiên

Mỗi bộ $D_k$ có đúng 4 nguồn: $x_{train}$, $\varepsilon_{train}$, $x_{test}$, $\varepsilon_{test}$. Mọi thứ còn lại (hệ số đa thức, $E_{in}$, $\hat E_{out}$, đường fit) là **hàm tất định** của 4 nguồn này.

- Số ngẫu nhiên trong numpy là **giả ngẫu nhiên**: cùng seed + cùng thứ tự gọi ⇒ cùng dãy số. Seed **không làm giảm** tính ngẫu nhiên về mặt thống kê; nó chỉ làm kết quả **tái lập được**.
- Kiểm tra tái lập: chạy 2 lần, so sánh mảng (`np.array_equal`) hoặc in vài giá trị đầu / tổng kiểm (`x.sum()`) ở cell sinh dữ liệu.

## 2. Kiểm tra đúng phân phối

| Thành phần | Giả thuyết | Kiểm định | Hình |
|---|---|---|---|
| $x$ | $U[0,1]$ | `scipy.stats.kstest(x, "uniform")` | Histogram (n=1000) |
| $\varepsilon = y - f(x)$ | $\mathcal N(0, 0.03)$ | `kstest(e/0.03, "norm")`; so `e.std(ddof=1)` với 0.03; `e.mean()` ≈ 0 | Histogram + QQ-plot (`stats.probplot`) |

Có thể tính $\varepsilon$ ngược lại từ dữ liệu vì ta biết $f$ — đây là lợi thế của dữ liệu mô phỏng.

## 3. Kiểm tra độc lập

| Cặp | Kiểm tra | Kỳ vọng |
|---|---|---|
| $x$ và $\varepsilon$ (trong train) | `np.corrcoef(x, e)` | ≈ 0; độ lệch chuẩn của hệ số tương quan mẫu ≈ $1/\sqrt{n-1}$ |
| train và test | `stats.ks_2samp(x_tr, x_te)`; không có giá trị trùng | Cùng phân phối, không chung điểm |
| $D_1, D_2, D_3$ | Không có giá trị x trùng; tương quan giữa các mảng ≈ 0 | Độc lập |

## 4. Kết quả minh họa — và bẫy cần tránh

| n | Bộ | KS-uniform p | KS-normal p | std(e) | corr(x, e) |
|---|---|---|---|---|---|
| 10 | D1 | 0.69 | 0.43 | 0.0239 | **+0.45** |
| 10 | D2 | **0.02** | 0.54 | 0.0280 | −0.04 |
| 10 | D3 | 0.53 | 0.05 | 0.0264 | +0.38 |
| 100 | D3 | 0.64 | 0.81 | 0.0278 | −0.21 |
| 1000 | D1–D3 | 0.18–0.72 | 0.12–0.92 | 0.0294–0.0310 | 0.00–0.05 |

**Bẫy 1 — báo động giả là bình thường.** D2 (n=10) có p = 0.02 < 0.05, D1 có corr = +0.45, dù generator hoàn toàn đúng.
- Khi giả thuyết đúng, p < 0.05 vẫn xảy ra 5% số lần (mô phỏng: 4.95%). Làm 9 bộ × 4 kiểm định = 36 phép thử ⇒ **kỳ vọng ~2 kết quả "đáng ngờ" chỉ do may rủi**.
- Với n=10, độ lệch chuẩn của corr ≈ $1/\sqrt9 = 0.33$ ⇒ corr = 0.45 chỉ cách 0 khoảng 1.4 độ lệch chuẩn — không bất thường.
- ⇒ **Không** sinh lại dữ liệu / đổi seed khi một kiểm định "trượt". Làm vậy là chọn lọc dữ liệu, phá đúng tính ngẫu nhiên mình đang muốn bảo vệ.

**Bẫy 2 — n nhỏ không chứng minh được gì, nhưng vẫn bắt được lỗi code.** Kiểm định trên n=10 yếu với sai lệch nhỏ, nhưng **bắt được 100%** các lỗi thường gặp:
- truyền σ² = 0.0009 thay vì σ = 0.03 ⇒ std(e) nhỏ đi 33 lần;
- sinh x sai miền (ví dụ U[0, 0.5]).

⇒ Vai trò của kiểm định ở đây là **phát hiện lỗi cài đặt**, không phải "chứng nhận" dữ liệu ngẫu nhiên. Kiểm định có ý nghĩa thống kê nhất là trên n=1000.

**Bẫy 3 — "ngẫu nhiên" ≠ "đại diện".** Một bộ 10 điểm đúng phân phối vẫn có thể phủ kém [0,1] (xem [06](06_danh_gia_sinh_du_lieu.md)). Đó **chính là** hiện tượng variance mà bài tập muốn thấy — không phải lỗi cần sửa.

## 5. Đo ảnh hưởng của ngẫu nhiên lên kết quả

| Mức | Cách làm | Đại lượng báo cáo |
|---|---|---|
| Trong đề (R=3) | So $E_{in}$, $\hat E_{out}$ giữa $D_1,D_2,D_3$ | mean ± std qua 3 bộ; độ tản của 3 đường fit |
| Mở rộng (R=200) | Lặp nhiều bộ cùng n | Khoảng 5%–95% của $\hat E_{out}$; bias², var |
| Tách nguồn (tùy chọn) | (a) cố định x, chỉ đổi ε; (b) cố định ε-seed, đổi x; (c) dùng test rất lớn (10⁴) để loại ngẫu nhiên của test | Cho biết dao động đến từ **vị trí x** (phủ kém ⇒ bậc cao bùng nổ) hay từ **nhiễu y**, hay chỉ do **test nhỏ** |

Gợi ý câu viết trong notebook: *"Các khác biệt giữa D1, D2, D3 hoàn toàn do ngẫu nhiên của việc lấy mẫu; độ lớn của khác biệt đó theo từng bậc chính là biểu hiện của variance."*

## 6. Đoạn kiểm tra đề xuất trong notebook

```python
from scipy import stats

def check_dataset(x_tr, y_tr, x_te, y_te):
    e = y_tr - f(x_tr)
    return {
        "x∈[0,1]":       bool((x_tr >= 0).all() and (x_tr < 1).all()),
        "KS uniform p":  stats.kstest(x_tr, "uniform").pvalue,
        "mean(e)":       e.mean(),
        "std(e)":        e.std(ddof=1),
        "KS normal p":   stats.kstest(e / SIGMA, "norm").pvalue,
        "corr(x,e)":     np.corrcoef(x_tr, e)[0, 1],
        "KS train/test p": stats.ks_2samp(x_tr, x_te).pvalue,
        "#test ngoài [min,max] x_train": int(((x_te < x_tr.min()) | (x_te > x_tr.max())).sum()),
    }
# Chỉ assert điều kiện cứng (miền x, kích thước, std(e) khi n=1000); p-value chỉ báo cáo, không assert.
```

Nếu không muốn thêm thư viện `scipy`, chỉ cần giữ: miền x, mean/std của e, corr(x, e), và histogram/QQ-plot cho n=1000.
