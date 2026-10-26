# PLAN — BT1: Mô phỏng Bias–Variance Tradeoff (slides tr. 58–68)

Thư mục này chứa phần **phân tích trước khi code**. Kiến thức nền nằm ở [../docs/](../docs/README.md).

| File | Nội dung |
|---|---|
| [01_phan_tich_de_bai.md](01_phan_tich_de_bai.md) | Bóc tách đề bài → yêu cầu cụ thể, ánh xạ với từng slide, checklist bài nộp |
| [02_thiet_ke_thuc_nghiem.md](02_thiet_ke_thuc_nghiem.md) | Thiết kế dữ liệu, mô hình, độ đo, hình vẽ, cấu trúc notebook |
| [03_pros_cons_lua_chon.md](03_pros_cons_lua_chon.md) | Các quyết định kỹ thuật và ưu/nhược điểm từng phương án |
| [04_yeu_to_critical_rui_ro.md](04_yeu_to_critical_rui_ro.md) | Các yếu tố critical, rủi ro, kết quả chạy thử (pilot) |
| [05_ke_hoach_trien_khai.md](05_ke_hoach_trien_khai.md) | Lộ trình từng bước, giả thuyết kỳ vọng, checklist trước khi nộp |
| [06_danh_gia_sinh_du_lieu.md](06_danh_gia_sinh_du_lieu.md) | Đánh giá bước tạo 3 bộ dữ liệu: SNR, độ phủ của x, nhiễu, độ nhạy theo seed, các quyết định khi sinh dữ liệu |
| [07_xac_minh_dataset_va_bieu_dien_bias_variance.md](07_xac_minh_dataset_va_bieu_dien_bias_variance.md) | Điều kiện để 3 bộ dữ liệu hợp lệ cho bias–variance; cách biểu diễn bias/variance (bắt buộc + mở rộng) |
| [08_kiem_dinh_tinh_ngau_nhien.md](08_kiem_dinh_tinh_ngau_nhien.md) | Xác định & kiểm định tính ngẫu nhiên: nguồn ngẫu nhiên, kiểm tra phân phối/độc lập, đo ảnh hưởng của ngẫu nhiên lên kết quả |
| [pilot/pilot_bias_variance.py](pilot/pilot_bias_variance.py) | Script chạy thử dùng để kiểm chứng các rủi ro ở file 04 |
| [pilot/input_analysis.py](pilot/input_analysis.py) | Script phân tích input cho file 06 |
| [pilot/viz_check.py](pilot/viz_check.py) | Script kiểm chứng cho file 07 |
| [pilot/rand_check.py](pilot/rand_check.py) | Script kiểm định cho file 08 |

## Tóm tắt 1 phút

- **Bài toán:** $f(x)=\sin(1+x^2)$, $x\sim U[0,1]$, $y=f(x)+\varepsilon$, $\varepsilon\sim\mathcal N(0,\ 0.03^2)$ — **σ = 0.03 là độ lệch chuẩn**, nên phương sai nhiễu $\sigma^2 = 0.0009$ (đây cũng là "sàn" của $\hat E_{out}$).
- **Mô hình:** hồi quy đa thức bậc 1–9, fit bằng bình phương tối thiểu (OLS).
- **Độ đo:** MSE trung bình $\frac1n\sum(\hat y_i-y_i)^2$ trên train ($E_{in}$) và test ($\hat E_{out}$).
- **Thực nghiệm:** 3 kích thước $n\in\{10,100,1000\}$ × 3 bộ dữ liệu $D_1,D_2,D_3$ × 9 bậc.
- **Đầu ra:** bảng $E_{in}/\hat E_{out}$ (như slide 66), lưới 3×3 hình (như slide 65, 68), nhận xét bias/variance, trả lời 2 câu hỏi về ảnh hưởng của $n$.
- **3 rủi ro lớn nhất** (xem file 04):
  1. Dùng `np.random.default_rng()` thì `np.random.seed(MSHV)` **không có tác dụng** → kết quả không tái lập được.
  2. Đa thức bậc cao với n=10 **ngoại suy bùng nổ** ở các điểm test nằm ngoài khoảng của x_train (pilot: median $\hat E_{out}$ của bậc 9 ≈ 660) → bắt buộc cố định `ylim` và giải thích hiện tượng.
  3. Chỉ 3 bộ dữ liệu là bằng chứng **định tính**; muốn kết luận chắc chắn về bias/variance nên bổ sung mô phỏng nhiều bộ dữ liệu (phần mở rộng).
- **Câu trả lời kỳ vọng:** tăng $n$ **không** giảm bias (bias là thuộc tính của lớp giả thuyết); tăng $n$ **có** giảm variance (≈ tỉ lệ $1/n$).
