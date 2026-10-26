# 06 — Đánh giá bước tạo 3 bộ dữ liệu & phân tích input

Số liệu lấy từ [pilot/input_analysis.py](pilot/input_analysis.py) (mô phỏng 5.000–20.000 lần, seed khác MSHV).

## 1. Đặc tả input

| Thành phần | Đặc tả | Ghi chú |
|---|---|---|
| $x$ | $U[0,1]$, i.i.d. (`np.random.uniform(0,1,n)` trả về giá trị trong $[0,1)$) | Không sắp xếp, không cách đều |
| $f(x)$ | $\sin(1+x^2)$; $f(0)=0.8415$, cực đại $f=1$ tại $x=0.7555$, $f(1)=0.9093$ | Tăng rồi giảm, **không đối xứng** |
| $\varepsilon$ | $\mathcal N(0,\sigma)$, $\sigma=0.03$ ⇒ $\sigma^2=9\times10^{-4}$ | Độc lập với $x$ |
| $y$ | $f(x)+\varepsilon$, ~99.7% nằm trong $[0.751,\ 1.090]$ | `ylim=(0.7,1.1)` chứa gần hết điểm |
| Mỗi $D_k$ | train $n$ điểm + test $n$ điểm, độc lập, cùng phân phối | $k=1,2,3$; $n\in\{10,100,1000\}$ |

## 2. Phát hiện về input

### 2.1 Tín hiệu yếu so với nhiễu ⇒ n=10 khó phân biệt các bậc

| Đại lượng | Giá trị | So với $\sigma^2$ |
|---|---|---|
| $\operatorname{Var}_x f(x)$ (biên độ tín hiệu) | 3.08e-3 | **SNR ≈ 3.4** |
| bias² tốt nhất có thể của bậc 1 | 7.8e-4 | 0.87 σ² |
| bias² tốt nhất có thể của bậc 2 | 3.6e-4 | 0.40 σ² |
| bias² tốt nhất có thể của bậc 3 | 6.2e-6 | 0.007 σ² |
| bias² tốt nhất có thể của bậc ≥ 4 | ~1e-8 | ≈ 0 |

- Phần "độ cong" mà bậc 1 bỏ lỡ (bias² ≈ 7.8e-4) **nhỏ hơn cả phương sai nhiễu**. Với 10 điểm, tín hiệu này dễ bị nhiễu che mất.
- Hệ quả đo được: với n=10, **35% số bộ dữ liệu có $\hat E_{out}$ bậc 1 < bậc 3**; bậc tốt nhất theo $\hat E_{out}$ phân bố: bậc 1: 19%, bậc 2: 21%, bậc 3: 35%, bậc 4: 14%, bậc ≥5: 11%.
- ⇒ Không được kỳ vọng mỗi $D_k$ (n=10) cho đường cong U "đẹp" có đáy ở bậc 3 như slide 66. Nhận xét phải dựa vào **xu hướng qua cả 3 bộ**, không dựa vào cực tiểu của một bộ.
- Với n=100, n=1000: bậc 1–2 thua bậc 3 trong ~100% trường hợp ⇒ bias của bậc 1–2 lộ ra rõ khi variance giảm. Đây là chứng cứ tốt cho câu hỏi "tăng n có giảm bias không".

### 2.2 Độ phủ của $x$ — nguồn gốc bùng nổ bậc cao

| n | Khoảng hở kỳ vọng ở mỗi đầu | Khoảng hở lớn nhất bên trong | P(1 điểm test nằm ngoài [min, max] của x_train) | P(≥1 điểm test nằm ngoài) |
|---|---|---|---|---|
| 10 | 0.091 | 0.257 | 18.2% (= 2/(n+1)) | **76%** |
| 100 | 0.010 | 0.051 | 2.0% | 75% |
| 1000 | 0.001 | 0.007 | 0.2% | 75% |

- n=10: 11% không có điểm train nào trong [0.8, 1] (vùng có đỉnh của $f$), 11% không có điểm nào trong [0, 0.2]; 48% có ít nhất một đoạn dài 0.2 không có điểm nào.
- Bậc 9 với n=10: $\hat E_{out}$ trung vị **≈ 745**; nếu có điểm test ngoài khoảng train ⇒ trung vị ≈ 3.6e3; kể cả khi không có ⇒ trung vị vẫn ≈ 6.6 (dao động trong các khoảng hở lớn bên trong).
- Xác suất gặp được bảng "đẹp" như slide 66 ($\hat E_{out}$ bậc 9 < 0.005) chỉ **≈ 2%**.
- Lưu ý: tỉ lệ "≥1 điểm test ngoài khoảng" ≈ 75% ở mọi n, nhưng khi n lớn, phần ngoại suy chỉ dài ~0.001 và bậc 9 đã ổn định (p ≪ n) nên vô hại. Cái quyết định là **độ rộng khoảng hở so với độ phức tạp mô hình**, không phải việc có ngoại suy hay không.

### 2.3 Nhiễu thực tế của một bộ dữ liệu

Tỉ số (phương sai mẫu của $\varepsilon$) / $\sigma^2$:

| n | trung bình | độ lệch chuẩn | khoảng 5%–95% |
|---|---|---|---|
| 10 | 0.99 | 0.44 | [0.39, 1.81] |
| 100 | 1.00 | 0.14 | [0.78, 1.24] |
| 1000 | 1.00 | 0.04 | [0.93, 1.07] |

- n=10: "sàn nhiễu" thật của một bộ có thể là 0.00035 hay 0.0016. Đừng kết luận gì từ việc $E_{in}$ hay $\hat E_{out}$ của bậc 3 lệch khỏi 0.0009.
- n=10, bậc 1: $\hat E_{out}$ có khoảng 5–95% là [0.00074, 0.00475], tức chênh nhau 6 lần chỉ do mẫu ⇒ số liệu slide 66 (0.00215) chỉ là một giá trị trong khoảng này.
- Sanity check khả thi: với n=1000, `np.std(y - f(x))` phải nằm trong ~[0.0287, 0.0313].

## 3. Đánh giá thiết kế bước sinh dữ liệu

### 3.1 Những gì đề quy định và đã đủ rõ
- Phân phối $x$, $f$, $\sigma$, kích thước, số bộ — rõ ràng.
- Seed toàn cục — rõ ràng (xem C1 trong [04](04_yeu_to_critical_rui_ro.md)).

### 3.2 Những quyết định phải tự đưa ra

| Quyết định | Phương án đề xuất | Lý do / rủi ro nếu làm khác |
|---|---|---|
| **Thứ tự rút số ngẫu nhiên** | Cố định: với n ∈ (10, 100, 1000) → với k = 1..3 → `x_tr`, `eps_tr`, `x_te`, `eps_te` | Global RNG là một dòng duy nhất: đổi thứ tự, chèn một lệnh `np.random.*` khác, hoặc chạy lại một cell thì toàn bộ dữ liệu sau đó đổi theo. Phải viết thứ tự này ra trong notebook |
| **Sinh 1 lần, lưu lại, dùng chung** | Lưu vào dict `data[n][k]`; bảng, hình 65, hình 68 đều đọc từ đó | Nếu mỗi hàm vẽ tự sinh lại dữ liệu thì bảng và hình sẽ là **hai bộ dữ liệu khác nhau** |
| Bộ dữ liệu các n độc lập hay lồng nhau (bộ n=10 là tập con của bộ n=100) | **Độc lập** (bám sát đề) | Lồng nhau làm phép so sánh giữa các n ít nhiễu hơn nhưng không đúng tinh thần "tạo bộ dữ liệu mới"; có thể nhắc như một lựa chọn khác |
| Train/test sinh riêng hay sinh 2n điểm rồi chia đôi | Sinh riêng (về thống kê là tương đương) | Sinh riêng đọc rõ ý hơn và khớp cách đề mô tả |
| Sắp xếp $x$ | Không sắp xếp khi lưu; chỉ sắp xếp/dùng lưới `linspace` khi vẽ | `polyfit` không phụ thuộc thứ tự; vẽ đường nối theo thứ tự ngẫu nhiên sẽ ra hình rối |
| Chọn seed / sinh lại để giống slide | **Không** | Chọn lọc dữ liệu (cherry-picking); bảng giống slide chỉ có xác suất ~2% |
| Loại bỏ điểm test ngoài khoảng train | **Không** | Làm sai lệch $\hat E_{out}$; hiện tượng bùng nổ chính là bằng chứng về variance |

### 3.3 Đề xuất bổ sung cho notebook (tốn ít công, giá trị cao)

1. **Bảng mô tả từng $D_k$** ngay sau khi sinh: min/max của x_train, khoảng hở lớn nhất, **số điểm test nằm ngoài [min, max] của x_train**, std của phần dư `y - f(x)`. Bảng này giải thích *trước* vì sao bậc 9 của bộ nào bùng nổ, nên phần nhận xét có căn cứ.
2. **Assert sanity check**: x ∈ [0,1], kích thước mảng đúng, train ≠ test, |std(eps) − 0.03| < 0.003 ở n=1000.
3. **Hình H0** (kiểu slide 60) cho từng $D_k$ với n=10: thấy ngay các khoảng hở.
4. Khi nhận xét n=10, ghi rõ: kết quả phụ thuộc mạnh vào mẫu (dẫn số liệu ở mục 2), do đó **3 bộ dữ liệu là bằng chứng định tính** ⇒ đây là lý do có phần mở rộng R=200 bộ.

## 4. Kết luận

- Thiết kế input của đề **hợp lý cho mục đích minh họa**: $f$ trơn, có một cực đại không đối xứng nên bậc 1–2 có bias thật; bậc ≥ 3 gần như hết bias; $\sigma$ đủ lớn để overfit hiện rõ ở n=10.
- Điểm cần lưu ý: tín hiệu "độ cong" (bias² bậc 1 ≈ 0.87σ²) **yếu so với nhiễu**, và 10 điểm $x$ ngẫu nhiên **phủ không đều** [0,1] ⇒ kết quả n=10 dao động rất mạnh theo seed; bảng của bạn gần như chắc chắn sẽ khác slide 66, đặc biệt ở bậc 7–9.
- Đi từ n=10 lên 100 và 1000 sẽ cho thấy rõ hai thứ: variance biến mất (các khoảng hở co lại, nhiễu trung bình hóa), còn bias của bậc 1–2 thì vẫn còn. Đó đúng là điều mà hai câu hỏi cuối muốn bạn chỉ ra.
