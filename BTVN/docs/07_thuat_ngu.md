# 07 — Thuật ngữ và ký hiệu

| Ký hiệu / thuật ngữ | Tiếng Việt | Ý nghĩa ngắn |
|---|---|---|
| $f$ — target function | hàm mục tiêu | Quan hệ thật giữa $x$ và $y$; trong bài $\sin(1+x^2)$ |
| $\hat f$, $h^{(D)}$ — fitted model / hypothesis | mô hình đã fit / giả thuyết | Hàm chọn được từ dữ liệu $D$ |
| $\mathcal H$ — hypothesis set | tập giả thuyết | Các hàm ứng viên (ví dụ đa thức bậc $d$) |
| $\mathcal A$ — learning algorithm | thuật toán học | Cách chọn $h$ từ $\mathcal H$ (OLS) |
| $\bar h$ — average hypothesis | giả thuyết trung bình | $\mathbb E_D[h^{(D)}]$ |
| $D_{in}$ / train set | dữ liệu trong mẫu / tập huấn luyện | Dùng để fit |
| $D_{out}$ / test set | dữ liệu ngoài mẫu / tập kiểm tra | Dùng để đánh giá |
| $E_{in}$ — in-sample error | lỗi trong mẫu | MSE trên train |
| $E_{out}$ — out-of-sample error | lỗi ngoài mẫu | MSE kỳ vọng trên toàn phân phối (lý thuyết) |
| $\hat E_{out}$ | ước lượng lỗi ngoài mẫu | MSE trên test |
| $\varepsilon$, $\sigma$ — noise | nhiễu, độ lệch chuẩn nhiễu | $\sigma=0.03$, $\sigma^2=0.0009$ |
| loss / cost / risk | hàm mất mát / chi phí / rủi ro | Lỗi từng điểm / lỗi tổng |
| bias | độ chệch | $\bar h - f$: sai số hệ thống |
| variance | phương sai | $\mathbb E_D[(h^{(D)}-\bar h)^2]$: độ nhạy với dữ liệu |
| irreducible error | lỗi không thể giảm | $\sigma^2$ |
| overfitting | quá khớp | $E_{in}$ nhỏ nhưng $E_{out}$ lớn |
| underfitting | chưa khớp | Cả $E_{in}$, $E_{out}$ lớn do thiếu năng lực |
| i.i.d. | độc lập, cùng phân phối | Giả định lấy mẫu |
| Vandermonde matrix | ma trận Vandermonde | Ma trận đặc trưng $[x^0,\dots,x^d]$ |
| ill-conditioned | điều kiện kém | Hệ gần suy biến, nhạy sai số số học |
| interpolation / extrapolation | nội suy / ngoại suy | Dự đoán trong / ngoài vùng có dữ liệu |
| Runge phenomenon | hiện tượng Runge | Dao động mạnh ở biên của đa thức nội suy bậc cao |
| regularization | chính quy hóa | Ràng buộc mô hình để giảm variance |
