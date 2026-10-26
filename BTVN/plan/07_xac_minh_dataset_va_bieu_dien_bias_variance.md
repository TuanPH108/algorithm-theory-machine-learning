# 07 — Xác minh 3 bộ dữ liệu & cách biểu diễn bias / variance

Số liệu kiểm chứng: [pilot/viz_check.py](pilot/viz_check.py) (500 lần lặp, seed khác MSHV).

## Phần A — 3 bộ dữ liệu phải thỏa những điều kiện gì?

Bias và variance được định nghĩa qua kỳ vọng $\mathbb E_D[\cdot]$ trên các bộ dữ liệu **lấy từ cùng một phân phối**. Muốn $D_1,D_2,D_3$ là "3 lần rút mẫu" hợp lệ của $\mathbb E_D$, thì **thứ duy nhất được phép khác nhau giữa chúng là sự ngẫu nhiên**.

| # | Quyết định / điều kiện | Vì sao cần cho bias–variance | Cách xác minh trong notebook |
|---|---|---|---|
| V1 | **Cùng phân phối sinh**: cùng $f$, cùng $P(x)=U[0,1]$, cùng $\sigma=0.03$ | Nếu phân phối khác nhau thì độ tản giữa các mô hình không còn là variance mà lẫn cả "khác bài toán" | Một hàm `make_split(n)` duy nhất, gọi 3 lần; không có tham số nào thay đổi theo k |
| V2 | **Độc lập** giữa $D_1,D_2,D_3$ và giữa train/test trong cùng bộ | $\mathbb E_D$ giả định các mẫu i.i.d.; test phụ thuộc train thì $\hat E_{out}$ bị chệch | Mỗi bộ rút số mới từ RNG; assert không có mảng nào dùng chung (`not np.shares_memory`, các giá trị x khác nhau) |
| V3 | **Cùng n** trong một kịch bản | Variance phụ thuộc n; trộn n khác nhau sẽ lẫn hai hiệu ứng | Assert `len(x_tr)==n` cho cả 3 bộ |
| V4 | **Cùng quy trình học**: cùng tập bậc 1–9, cùng `np.polyfit`, không chuẩn hóa riêng theo từng bộ | Bias/variance là thuộc tính của **(H, A, n)**; đổi A giữa các bộ thì sai định nghĩa | Một hàm `fit_poly` dùng chung |
| V5 | **Cùng lưới đánh giá** $x_g$ = `linspace(0,1,300)` cho mọi mô hình | Bias(x), Var(x) định nghĩa **theo từng điểm** x ⇒ phải so dự đoán tại cùng các điểm | Biến toàn cục `XG`, dùng cho cả vẽ lẫn tính toán |
| V6 | **Tái lập được**: seed đúng mẫu, thứ tự rút số cố định, sinh một lần rồi lưu | Người chấm phải nhận lại đúng 3 bộ | Restart & Run All 2 lần ⇒ bảng giống hệt |
| V7 | **Không chọn lọc**: không đổi seed, không bỏ điểm | Chọn lọc làm sai phân phối của $\mathbb E_D$ | Ghi rõ trong notebook: dùng seed = MSHV duy nhất |
| V8 | **Sanity check số liệu** | Bắt lỗi σ ↔ σ², sai miền x | n=1000: `std(y-f(x))∈[0.0287,0.0313]`, `0≤x<1` |
| V9 | **Nhận thức giới hạn R=3** | 3 mẫu chỉ ước lượng $\mathbb E_D$ rất thô | Xem bảng dưới; nhận xét dùng từ "quan sát / xu hướng", không phải "chứng minh" |

### Độ tin cậy khi ước lượng variance từ R bộ dữ liệu (n=10, bậc 3)

| R | Ước lượng var / var thật (khoảng 5%–95%) |
|---|---|
| 3 | 0.06 – 6.2 (trung vị 0.28) |
| 30 | 0.26 – 3.3 |
| 200 | 0.58 – 4.7 |

- Với R=3 ước lượng có thể lệch **gấp hàng chục lần** ⇒ 3 bộ chỉ đủ để **thấy** variance (độ tản của 3 đường), không đủ để **đo**.
- Kể cả với R=200, ở n=10 phân phối dự đoán có **đuôi rất nặng** (thỉnh thoảng có 1 fit bùng nổ) ⇒ trung bình và phương sai bị vài fit chi phối. Nên dùng **thống kê bền**: trung vị thay cho trung bình ($\bar h$), $(\text{IQR}/1.349)^2$ thay cho phương sai, hoặc chỉ tính trên đoạn trong [0.1, 0.9] — và **ghi rõ** đã dùng cách nào.

## Phần B — Bias và variance phân bố thế nào theo x?

Trung bình bias² và var trên từng đoạn x:

| n | bậc | [0, 0.1] bias² / var | [0.1, 0.9] bias² / var | [0.9, 1] bias² / var |
|---|---|---|---|---|
| 10 | 1 | 1.4e-4 / 6.4e-4 | 4.0e-4 / 3.7e-4 | **4.7e-3** / 1.1e-3 |
| 10 | 3 | 2.9e-5 / 1.7e-2 | 1.5e-5 / 4.2e-3 | 5.8e-4 / **6.3e-2** |
| 10 | 9 | 1.4e-3 / **1.4e+3** | 3.0e-5 / 7.6e-1 | 6.4e-3 / **1.5e+3** |
| 100 | 1 | 1.2e-4 / 4.2e-5 | 4.4e-4 / 2.8e-5 | **4.3e-3** / 8.8e-5 |
| 100 | 3 | 1.2e-5 / 8.7e-5 | 5.4e-6 / 2.9e-5 | 1.1e-5 / 8.4e-5 |
| 100 | 9 | 4.5e-7 / 5.1e-4 | 3.6e-7 / 7.5e-5 | 1.6e-7 / 7.1e-4 |

(n=10, bậc 9 dùng $\bar h$ = trung vị và var bền vì phân phối có đuôi nặng.)

Nhận xét rút ra:
- **Bias của bậc 1 tập trung ở đầu phải** (x ≈ 1, sau đỉnh của $f$): đường thẳng không thể "đi xuống" sau đỉnh. Từ n=10 lên 100 con số này **không đổi** (4.7e-3 → 4.3e-3) ⇒ tăng n không giảm bias.
- **Variance tập trung ở hai đầu** (vùng thiếu dữ liệu, ngoại suy); bậc càng cao thì càng tập trung. Từ n=10 lên 100, var ở hai đầu của bậc 9 giảm khoảng **6 bậc độ lớn**.
- ⇒ Biểu diễn bias/variance **theo x** cho thấy nhiều hơn một con số trung bình.

## Phần C — Cách biểu diễn

### C1. Mức bắt buộc (chỉ với 3 bộ) — nâng cấp hình slide 68

Trong mỗi ô (một bậc d), ngoài 3 đường fit và $f$, vẽ thêm:

| Thành phần | Cách vẽ | Biểu diễn cho |
|---|---|---|
| $f(x)$ | Xám đậm, nét liền | Mục tiêu |
| $h^{(D_1)},h^{(D_2)},h^{(D_3)}$ | 3 màu cố định, nét mảnh | Các lần học |
| **$\bar h_3(x)$** = trung bình 3 đường | Đen, nét đứt | Ước lượng thô của $\bar h$ |
| **Dải min–max** của 3 đường | `fill_between`, alpha 0.2 | **Variance**: dải càng rộng, variance càng lớn |
| Khoảng giữa $\bar h_3$ và $f$ | Phần bị bôi màu nhạt khác, hoặc chỉ cần thấy khoảng cách 2 đường | **Bias** |

Dưới hình, thêm bảng số theo từng bậc (mỗi kịch bản n):

| Degree | mean $E_{in}$ | mean $\hat E_{out}$ | std $\hat E_{out}$ (qua 3 bộ) | $\hat E_{out}-E_{in}$ | spread = trung bình theo $x_g$ của var 3 đường | bias² của $\bar h_3$ |
|---|---|---|---|---|---|---|

Gắn nhãn hai cột cuối là **"ước lượng thô, R=3"**.

Ý nghĩa từng cột:
- std $\hat E_{out}$ và spread ↔ **variance**
- $\hat E_{out}-E_{in}$ ↔ **triệu chứng** của variance cao (overfit)
- mean $\hat E_{out}$ bậc 1–2 vượt $\sigma^2$ ngay cả khi n lớn ↔ **bias**

### C2. Mức mở rộng (R = 200 bộ cho mỗi n) — đo bias/variance thật sự

| Hình | Nội dung | Trả lời câu hỏi |
|---|---|---|
| **E1. Đường tradeoff** (1×3, mỗi ô 1 giá trị n, trục y log) | Theo bậc d: bias², var, $\sigma^2$ (đường ngang), tổng bias²+var+σ², và chấm mean $\hat E_{out}$ đo được | Thấy hình chữ U; kiểm chứng phân rã: tổng ≈ $\hat E_{out}$ |
| **E2. So sánh theo n** (1×2) | Trái: bias²(d) với 3 đường của n=10/100/1000. Phải: var(d) với 3 đường | **Câu hỏi 6**: các đường bias² **chồng lên nhau**; các đường var **tụt xuống ~10 lần** mỗi lần n tăng 10 lần |
| **E3. Dải phễu** (lưới 3×3, mỗi n) | $f$, $\bar h$, dải $\bar h\pm2\,\text{sd}$ (hoặc phân vị 5–95%) | Dải rộng ở bậc cao / n nhỏ / hai đầu ⇒ variance; $\bar h$ lệch $f$ ⇒ bias |
| **E4. Theo vị trí x** (chọn bậc 1, 3, 9) | bias²(x) và var(x), trục y log | Bias bậc 1 dồn về x≈1; variance dồn về hai đầu |
| **E5. Heatmap** bậc × n | Màu log10(bias²) và log10(var), 2 heatmap đặt cạnh nhau | Tổng kết trong một hình: bias đổi theo **cột bậc**, var đổi theo **cả bậc lẫn n** |

Gợi ý thực hiện:
- Hàm `simulate(n, R, degrees)` trả về mảng dự đoán kích thước `(R, len(degrees), len(XG))` và mảng $\hat E_{out}$ ⇒ mọi hình E1–E5 đều tính từ mảng này.
- Với n=10 dùng thống kê bền (Phần A); vẽ phân vị thay cho ±2 sd.
- Phần mở rộng nên dùng **luồng ngẫu nhiên riêng** sau khi đã sinh xong $D_1..D_3$ (ví dụ chạy sau cùng trong notebook) để không làm thay đổi 3 bộ dữ liệu bắt buộc.

### C3. Hình nào làm bằng chứng cho phát biểu nào

| Phát biểu | Bằng chứng mức bắt buộc | Bằng chứng mức mở rộng |
|---|---|---|
| Bậc thấp: bias cao, variance thấp | Dải hẹp nhưng $\bar h_3$ lệch $f$ ở đỉnh / x≈1 | E1, E4 |
| Bậc cao (n=10): bias thấp, variance cao | Dải rất rộng; $E_{in}\approx0$, $\hat E_{out}$ lớn, std lớn | E1, E3 |
| Tăng n **không** giảm bias | n=1000, bậc 1: dải gần như bằng 0 nhưng vẫn lệch $f$; mean $\hat E_{out}$ ≈ 0.0017 > σ² ở cả n=100 và 1000 | E2 trái, E5 |
| Tăng n **giảm** variance | Dải co lại ở mọi bậc; std $\hat E_{out}$ và $\hat E_{out}-E_{in}$ giảm | E2 phải, E3, E5 |
| Tăng n làm bậc tối ưu dịch sang phải | Bậc có mean $\hat E_{out}$ nhỏ nhất tăng dần theo n | E1 |
