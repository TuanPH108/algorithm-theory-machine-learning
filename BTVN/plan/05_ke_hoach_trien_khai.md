# 05 — Kế hoạch triển khai

## Lộ trình

| Bước | Việc | Đầu ra | Kiểm tra |
|---|---|---|---|
| 0 | Chuẩn bị môi trường (venv + numpy, matplotlib, pandas, jupyter) | Kernel chạy được | `import numpy` OK |
| 1 | Cell import + seed đúng mẫu đề | Cell 1 | Có đủ `import random` |
| 2 | Hàm `f`, `generate_dataset(n)` dùng `np.random.uniform/normal` | Hàm | n=1000: std(nhiễu) ≈ 0.03; x ∈ [0,1] |
| 3 | Hàm `fit_poly(x,y,d)`, `mse(c,x,y)` | Hàm | n=10, d=9 ⇒ $E_{in}$ < 1e-10 |
| 4 | `run_experiment(n)`: sinh 3 bộ, fit 9 bậc, trả về coefs + DataFrame | Hàm | Bảng 9×6 |
| 5 | Hàm vẽ `plot_fits_single(D, coefs)` (slide 65) và `plot_fits_three(coefs_list)` (slide 68) | Hàm | Lưới 3×3, ylim cố định |
| 6 | Chạy n=10 → bảng + hình + nhận xét | Mục 3 | So xu hướng với slide 66/68 |
| 7 | Chạy n=100, n=1000 → bảng + hình + nhận xét | Mục 4, 5 | 3 đường fit hội tụ dần |
| 8 | Đồ thị $E_{in}$, $\hat E_{out}$ theo bậc cho 3 n (log y) | H3 | Hình chữ U của $\hat E_{out}$ ở n=10 |
| 9 | (Mở rộng) Mô phỏng R=200 bộ: bias², var theo (n, d) | Bảng + H4 | bias² bậc 1 ~ hằng theo n; var ~ 1/n |
| 10 | Viết trả lời 2 câu hỏi + kết luận, trích số liệu | Mục 7, 8 | Mỗi khẳng định có số/hình dẫn chứng |
| 11 | Restart & Run All, đổi tên `BT1_<MSHV>.ipynb` | File nộp | Checklist dưới |

## Giả thuyết kỳ vọng (để đối chiếu khi có kết quả thật)

**n = 10**
- $E_{in}$ giảm đơn điệu theo bậc, về 0 ở bậc 9.
- $\hat E_{out}$ hình chữ U: giảm tới khoảng bậc 3 (≈ 0.001–0.002) rồi tăng, có thể bùng nổ ở bậc 7–9.
- Hình slide 68: bậc 1–3 ba đường gần nhau (variance thấp), bậc 1–2 cùng lệch khỏi $f$ (bias cao); bậc 7–9 ba đường dao động mạnh, khác nhau rõ (variance cao).

**n = 100**
- $E_{in}$ giảm nhẹ theo bậc, tiến về ≈ 0.0009; không còn bằng 0 ở bậc 9.
- $\hat E_{out}$ ≈ $E_{in}$ với mọi bậc ≥ 3; bậc 1–2 vẫn cao hơn (bias).
- 3 đường fit gần như trùng nhau ở mọi bậc; bậc 9 có thể còn rung nhẹ ở hai đầu.

**n = 1000**
- $E_{in}\approx\hat E_{out}\approx 0.0009$ (=σ²) cho bậc ≥ 3; bậc 1 ≈ 0.0017, bậc 2 ≈ 0.0013 — chênh lệch chính là bias².
- 3 đường bậc 1 trùng nhau nhưng vẫn là đường thẳng lệch khỏi $f$ ⇒ minh họa "variance ≈ 0 nhưng bias không đổi".

**Trả lời câu hỏi**
1. Tăng n **không** làm giảm bias (bậc 1: $\hat E_{out}-\sigma^2$ ≈ 8·10⁻⁴ ở mọi n).
2. Tăng n **làm giảm** variance (độ phân tán giữa 3 đường, std của $\hat E_{out}$ qua 3 bộ, và var ước lượng đều giảm ~1/n).

Nếu kết quả thực tế lệch các giả thuyết trên → kiểm tra lại C1, C2 trong [04](04_yeu_to_critical_rui_ro.md) trước khi viết nhận xét.

## Checklist trước khi nộp

- [ ] `seed = <MSHV thật>` (không còn placeholder)
- [ ] Ba dòng seed nằm sau `import numpy` và trước mọi lệnh sinh dữ liệu
- [ ] Không có `default_rng` / `RandomState` không seed
- [ ] `np.random.normal(0, 0.03, ...)`
- [ ] Đủ 3 kịch bản × (bảng + hình 3×3 kiểu slide 68 + nhận xét)
- [ ] Trả lời rõ 2 câu hỏi, có số liệu dẫn chứng
- [ ] Restart & Run All không lỗi, output được lưu trong file
- [ ] Tên file `BT1_<MSHV>.ipynb`
