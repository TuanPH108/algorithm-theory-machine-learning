# 01 — Phân tích đề bài

## 1. Bóc tách yêu cầu

| # | Yêu cầu trong đề | Diễn giải cụ thể | Slide tham chiếu |
|---|---|---|---|
| R0 | Đoạn `seed = MSHV; np.random.seed(seed); random.seed(seed)` ở đầu file, sau import numpy, trước khi sinh dữ liệu | Phải `import random` và `import numpy as np`. **Mọi** phép sinh ngẫu nhiên phải đi qua global state của `np.random` (hoặc `random`) để seed có hiệu lực | — |
| R1 | Tạo 3 bộ $D_1,D_2,D_3$; mỗi bộ có train 10 điểm, test 10 điểm | $x\sim U[0,1]$, $y=\sin(1+x^2)+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma=0.03)$. Train và test **độc lập**, cùng phân phối | 58, 59 |
| R2 | Mỗi bộ: train 9 mô hình đa thức bậc 1–9 | Hồi quy đa thức bằng OLS; bậc $d$ có $d+1$ tham số | 60–63, 65 |
| R3 | Tính $E_{in}$ (train), $\hat E_{out}$ (test), lập bảng như slide 66 | Bảng 9 dòng × cột (Degree, $E_{in}$, $\hat E_{out}$), 5 chữ số thập phân như slide | 66 |
| R4 | Làm cho cả 3 bộ để có hình như slide 68; ghép 9 hình nhỏ trong 1 hình lớn | Lưới 3×3, mỗi ô = 1 bậc, vẽ 3 đường fit (của $D_1,D_2,D_3$) + đường $f(x)$ thật | 68 |
| R5 | Nhận xét về bias và variance | Dựa vào bảng và hình: bậc thấp → bias cao/variance thấp; bậc cao → bias thấp/variance cao | 53, 57, 66–70 |
| R6 | Lặp lại R1–R5 với kích thước train/test = 100 và 1000 | Hai kịch bản mới: (100,100), (1000,1000) | 70 |
| R7 | Trả lời: tăng $n$ có giảm bias? có giảm variance? | Phải **dựa vào kết quả thực nghiệm + hình vẽ**, không chỉ trích lý thuyết | 54, 70 |
| R8 | Nộp 1 file `BT1_MSHV.ipynb` chứa code, kết quả, hình, nhận xét | Notebook phải chạy lại từ đầu tới cuối (Restart & Run All) cho cùng kết quả | — |

## 2. Những điểm đề bài **không nói rõ** (cần tự quyết và ghi rõ trong notebook)

| Điểm mơ hồ | Quyết định đề xuất | Lý do |
|---|---|---|
| Hình slide 65 (9 ô, 1 bộ dữ liệu, có điểm train/test) có cần vẽ không? | **Có**, vẽ cho mỗi bộ $D_k$ (hoặc ít nhất $D_1$) | Đề nói "tạo bảng như trang 66" và tham chiếu trang 65; hình này giúp nhận xét over/underfit trực quan |
| Hình slide 68 có vẽ điểm dữ liệu không? | Không vẽ điểm (như slide), chỉ 3 đường fit + $f(x)$ | Ghép 3 bộ điểm vào 1 ô sẽ rối, nhất là n=1000 |
| Giới hạn trục y | Cố định `ylim=(0.7, 1.1)` như slide | Đa thức bậc cao ngoại suy rất xa — xem [04](04_yeu_to_critical_rui_ro.md) |
| "Lặp lại thực nghiệm 1–4" với 100 và 1000: train và test cùng kích thước? | Có: (100,100) và (1000,1000) | Đọc sát nghĩa đề "kích thước tập train và test (100, 1000)" |
| Có cần ước lượng bias²/variance bằng số không? | Có, như **phần mở rộng** (nhiều bộ dữ liệu, ví dụ 200 bộ) | 3 bộ chỉ cho bằng chứng định tính; số liệu định lượng làm câu trả lời R7 thuyết phục hơn |
| MSHV chưa biết | Dùng biến `MSHV` duy nhất ở đầu notebook, tên file đổi theo | Tránh sót khi thay |

## 3. Ánh xạ slide → sản phẩm

- **Slide 58–60:** mô tả dữ liệu → cell sinh dữ liệu + 1 hình "target + in-sample + out-of-sample" (tùy chọn nhưng nên có, giống slide 60).
- **Slide 61–63, 65:** fit từng bậc → lưới 3×3 cho một bộ dữ liệu.
- **Slide 66:** bảng $E_{in}$, $\hat E_{out}$ → `pandas.DataFrame`, có thể gộp 3 bộ thành 1 bảng nhiều cột.
- **Slide 67, 71:** đường cong lỗi theo độ phức tạp → nên vẽ thêm đồ thị $E_{in}$, $\hat E_{out}$ theo bậc (log-scale).
- **Slide 68:** 3 bộ dữ liệu chồng nhau → lưới 3×3 cho mỗi $n$.
- **Slide 69–70:** lý thuyết over/underfit và ảnh hưởng của $n$ → phần nhận xét & trả lời câu hỏi.

## 4. Checklist sản phẩm cuối

- [ ] Cell đầu: import (`numpy`, `random`, …) → `seed = MSHV` → `np.random.seed(seed)` → `random.seed(seed)`
- [ ] Hàm sinh dữ liệu dùng `np.random.uniform` / `np.random.normal` (global state)
- [ ] 3 kịch bản n ∈ {10, 100, 1000}, mỗi kịch bản 3 bộ dữ liệu
- [ ] Mỗi kịch bản: bảng $E_{in}/\hat E_{out}$ cho $D_1,D_2,D_3$
- [ ] Mỗi kịch bản: hình lưới 3×3 kiểu slide 68 (3 đường fit / ô)
- [ ] (Nên có) hình lưới 3×3 kiểu slide 65 và đồ thị lỗi theo bậc
- [ ] Nhận xét bias/variance cho từng kịch bản
- [ ] Trả lời 2 câu hỏi về $n$, có dẫn chứng số liệu và hình
- [ ] (Mở rộng) bảng bias²/variance ước lượng từ nhiều bộ dữ liệu
- [ ] Restart & Run All không lỗi; tên file `BT1_<MSHV>.ipynb`
