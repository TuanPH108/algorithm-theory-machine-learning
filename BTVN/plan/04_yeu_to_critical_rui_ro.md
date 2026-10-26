# 04 — Yếu tố critical & rủi ro

## A. Kết quả chạy thử (pilot)

Script: [pilot/pilot_bias_variance.py](pilot/pilot_bias_variance.py). 200 lần lặp mỗi cấu hình, test cùng kích thước train. Đây là **kiểm chứng thiết kế**, không phải kết quả nộp (seed khác MSHV).

| n | d | median $E_{in}$ | median $\hat E_{out}$ | mean $\hat E_{out}$ | bias² | var |
|---|---|---|---|---|---|---|
| 10 | 1 | 0.00119 | 0.00202 | 0.00228 | 8.2e-4 | 5.1e-4 |
| 10 | 3 | 0.00046 | 0.00155 | 0.00290 | 4.5e-5 | 2.5e-3 |
| 10 | 5 | 0.00032 | 0.00357 | 0.269 | — | 2.9e-1 |
| 10 | 9 | 0.00000 | **660.6** | **2.5e11** | (vô nghĩa) | (bùng nổ) |
| 100 | 1 | 0.00164 | 0.00168 | 0.00168 | 8.1e-4 | 3.6e-5 |
| 100 | 3 | 0.00087 | 0.00093 | 0.00094 | 6.8e-6 | 4.2e-5 |
| 100 | 9 | 0.00081 | 0.00101 | 0.00104 | 4.5e-7 | 1.5e-4 |
| 1000 | 1 | 0.00168 | 0.00168 | 0.00169 | 8.1e-4 | 3.6e-6 |
| 1000 | 3 | 0.00090 | 0.00090 | 0.00091 | 6.4e-6 | 3.7e-6 |
| 1000 | 9 | 0.00090 | 0.00090 | 0.00091 | 3.3e-8 | 9.0e-6 |

Những điều pilot xác nhận:
- **Bias² của bậc 1 gần như không đổi** (≈ 8·10⁻⁴) khi n đi từ 10 → 1000. Bậc 2 cũng vậy (≈ 3.7·10⁻⁴). ⇒ Tăng n không chữa được lớp giả thuyết quá đơn giản.
- **Variance giảm ~10 lần mỗi khi n tăng 10 lần** (bậc 1: 5.1e-4 → 3.6e-5 → 3.6e-6), phù hợp lý thuyết $\text{var}\approx\sigma^2(d+1)/n$.
- Khi n lớn, $E_{in}\approx\hat E_{out}\to\sigma^2=0.0009$ với mọi bậc ≥ 3 ⇒ $\sigma^2$ là **sàn không thể giảm**.
- Bảng slide 66 ($\hat E_{out}$ bậc 9 = 0.00231) là một mẫu **"may mắn"**; với dữ liệu ngẫu nhiên, n=10 bậc 9 thường cho $\hat E_{out}$ rất lớn.

## B. Các yếu tố critical

### C1. Seed & tái lập (mức: CAO — ảnh hưởng điểm hình thức)
- Bắt buộc đúng mẫu: `seed = MSHV`, `np.random.seed(seed)`, `random.seed(seed)`, đặt **sau import numpy và trước khi sinh dữ liệu**.
- Không dùng `np.random.default_rng()` (không bị ảnh hưởng bởi `np.random.seed`). Không dùng `sklearn` `random_state=None` ở đâu đó sinh dữ liệu.
- Không chạy cell sinh dữ liệu nhiều lần theo thứ tự lộn xộn; trước khi nộp: **Restart & Run All**.
- MSHV là số; nếu MSHV có chữ, đề yêu cầu "dãy các chữ số" ⇒ chỉ lấy phần số. `np.random.seed` yêu cầu 0 ≤ seed < 2³².

### C2. σ là độ lệch chuẩn (mức: CAO — sai là sai toàn bộ số liệu)
- `np.random.normal(0, 0.03, n)` ✔; `np.random.normal(0, 0.0009, n)` ✘ (nhiễu nhỏ đi 33 lần, $E_{in}$ sẽ ~1e-7).
- Kiểm tra nhanh: với n=1000, `np.std(y - f(x))` ≈ 0.03.

### C3. Ngoại suy bùng nổ của đa thức bậc cao với n nhỏ (mức: CAO — ảnh hưởng hình & nhận xét)
- x_train là 10 điểm ngẫu nhiên; khoảng [min, max] của chúng thường **không phủ hết [0,1]** (kỳ vọng hở ~1/11 ở mỗi đầu). Điểm test rơi vào vùng hở ⇒ đa thức bậc 9 ngoại suy ⇒ giá trị cực lớn.
- Hệ quả: $\hat E_{out}$ có thể lên hàng trăm/nghìn; hình bị "kéo" nếu không cố định `ylim`.
- Xử lý: cố định `ylim`; định dạng số khoa học trong bảng; dùng log-scale cho đồ thị lỗi; **giải thích** hiện tượng trong nhận xét (đây chính là variance cực lớn — điểm cộng, không phải lỗi).
- Không "sửa" bằng cách bỏ điểm test, đổi seed, hay sắp xếp x đều — đó là thay đổi thực nghiệm.

### C4. Điều kiện số (ill-conditioning) ở bậc 8–9 (mức: TRUNG BÌNH)
- cond(Vandermonde 10×10) ≈ 4·10¹⁰ ⇒ hệ số có thể mất vài chữ số chính xác, `np.polyfit` có thể phát `RankWarning`.
- Không dùng Normal Equations thủ công. Có thể kiểm chứng chéo bằng `np.polynomial.Polynomial.fit`.
- Với n=10, d=9: $E_{in}$ thực tế là ~1e-25…1e-30 — hiển thị 0.00000 là đúng.

### C5. Hiểu đúng "bias" và "variance" trong ngữ cảnh 3 bộ dữ liệu (mức: CAO — đây là phần chấm nội dung)
- Bias/variance là thuộc tính của **lớp giả thuyết + thuật toán + n**, định nghĩa qua kỳ vọng trên **mọi** bộ dữ liệu D. Không phải thuộc tính của 1 mô hình đã fit.
- Từ 3 bộ: variance ≈ độ phân tán giữa 3 đường fit trong cùng ô; bias ≈ độ lệch của "đường trung bình" (tưởng tượng) so với $f$.
- $E_{in}$ nhỏ ≠ bias nhỏ: bias đo bằng $\bar h - f$, không phải lỗi trên train. Đừng viết "bậc 9 có $E_{in}=0$ nên bias = 0".
- Khoảng cách $\hat E_{out}-E_{in}$ lớn là **triệu chứng** của variance cao/overfit, không phải định nghĩa.

### C6. Câu hỏi "tăng n có giảm bias không?" — cẩn thận sắc thái (mức: TRUNG BÌNH)
- Câu trả lời chính: **Không** (đáng kể). Bias của bậc 1, 2 giữ nguyên khi n tăng — thể hiện rõ: ở n=1000, 3 đường bậc 1 trùng nhau nhưng **cùng lệch** khỏi $f$ ở hai đầu.
- Sắc thái nên nêu: với n rất nhỏ so với số tham số (n=10, bậc ≥ 5), $\bar h$ thực nghiệm bị nhiễu bởi các fit bùng nổ nên bias ước lượng có thể trông "giảm" khi n tăng — đó là hiệu ứng ước lượng/bất ổn, không phải tăng n làm lớp giả thuyết linh hoạt hơn. Bias "tiệm cận" (best-in-class) không phụ thuộc n.
- Muốn giảm bias: tăng độ phức tạp/đổi lớp giả thuyết (slide 54, 70).

### C7. Câu hỏi "tăng n có giảm variance không?" (mức: TRUNG BÌNH)
- **Có.** Dẫn chứng: 3 đường fit hội tụ về nhau khi n tăng; std của $\hat E_{out}$ giữa 3 bộ giảm; khoảng cách $\hat E_{out}-E_{in}$ thu hẹp; (mở rộng) var giảm ~1/n.
- Hệ quả: với n lớn, mô hình phức tạp (bậc 9) trở nên ổn định và tận dụng được độ linh hoạt (slide 70) ⇒ "bậc tối ưu" dịch sang phải khi n tăng.

### C8. $\hat E_{out}$ trên 10 điểm test là ước lượng rất nhiễu (mức: THẤP–TRUNG BÌNH)
- Với n=10, thứ hạng các bậc theo $\hat E_{out}$ có thể đảo lộn giữa $D_1,D_2,D_3$. Không kết luận "bậc tốt nhất" chỉ từ 1 bộ.
- Nhận xét nên dựa trên xu hướng chung qua 3 bộ (và phần mở rộng).

### C9. Môi trường chạy (mức: TRUNG BÌNH — chặn việc triển khai)
- Python hệ thống trên máy hiện **chưa có numpy** (pilot chạy trong venv tạm). Cần môi trường có `numpy`, `matplotlib`, `pandas`, `jupyter` (ví dụ `python3 -m venv .venv && .venv/bin/pip install numpy matplotlib pandas jupyter`).
- Ghi phiên bản thư viện trong notebook (`np.__version__`) — kết quả số ngẫu nhiên legacy ổn định giữa các phiên bản numpy, nhưng ghi lại vẫn tốt.

### C10. Trình bày (mức: THẤP)
- Màu của $D_1,D_2,D_3$ cố định qua mọi hình; legend rõ; `suptitle` ghi n.
- Với n=1000, H0/H1 nên giảm kích thước marker/alpha.
- Nhận xét viết bằng Markdown ngay dưới mỗi hình/bảng, có trích số liệu cụ thể.

## C. Ma trận rủi ro

| Rủi ro | Khả năng | Tác động | Phòng ngừa |
|---|---|---|---|
| Seed không có tác dụng | Trung bình | Cao | Chỉ dùng `np.random.*` toàn cục; Restart & Run All 2 lần, so sánh bảng |
| Dùng phương sai thay cho độ lệch chuẩn | Trung bình | Cao | Assert `abs(np.std(eps)-0.03) < 0.005` với n=1000 |
| Hình hỏng vì bùng nổ | Cao | Trung bình | `ylim` cố định + ghi chú |
| Kết luận sai khái niệm bias | Trung bình | Cao | Dùng định nghĩa qua $\bar h$; phần mở rộng định lượng |
| Kết quả n=10 khác xa slide 66 | Cao | Thấp | Bình thường — giải thích do ngẫu nhiên; so xu hướng, không so số |
| Sót đổi tên file / MSHV | Thấp | Cao | Checklist cuối ở [05](05_ke_hoach_trien_khai.md) |
