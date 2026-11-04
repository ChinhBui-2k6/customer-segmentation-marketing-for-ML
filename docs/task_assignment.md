# Phân công công việc — ChinhBui & Tuananh

**Đề tài:** Phân nhóm khách hàng và dự đoán khả năng hưởng ứng chiến dịch marketing.  
**Yêu cầu:** Nhóm 2, mục 2.1–2.4 trong [Projects_Ideas.pdf](../Projects_ideas/Projects_Ideas.pdf).

**Thời gian triển khai:** **04/11/2026–18/11/2026** (hai tuần).  
**Mục tiêu tuần 1:** sẵn sàng nội dung giữa kỳ ngày **11/11/2026**. **Mục tiêu tuần 2:** hoàn thiện toàn bộ ngày **18/11/2026**. Đây là lịch làm việc nội bộ; chưa có ngày nộp chính thức của môn học.

> Đánh dấu `[x]` khi công việc đã hoàn thành và có kết quả kiểm tra. Các ô dưới đây là checklist kế hoạch, chưa xác nhận tiến độ thực tế.

## 1. Nhìn nhanh

| Phần | ChinhBui | Tuananh |
|---|---|---|
| **01 · Tiền xử lý** | Làm sạch, tạo đặc trưng, thống kê | Imputer, mã hóa, chuẩn hóa, chia tập, cấu hình |
| **02 · Giảm chiều** | PCA | Kernel PCA RBF; ghép pipeline PCA còn 1/3 số chiều |
| **03 · Phân cụm** | K-Means | GMM hoặc DBSCAN |
| **04 · Phân loại** | **Naive Bayes + SVM RBF** | **Logistic Regression** |
| **05 · Hồi quy** | KNN Regressor hoặc MLP Regressor | Tạo đầu ra xác suất; Linear Regression hoặc Ridge |
| **Tài liệu** | Slides, phần báo cáo của mình; đọc duyệt bản cuối | Hướng dẫn chạy/dữ liệu, phần báo cáo của mình; ghép và định dạng báo cáo |

**Làm chung:** chốt dữ liệu và hàm đánh giá → so sánh kết quả → viết mở đầu/kết luận → kiểm tra chéo → đóng gói.

## 2. Checklist của ChinhBui

### 01 · Tiền xử lý — mục 2.1

File: [notebook 01](../notebooks/01_preprocessing.ipynb) · [preprocessing.py](../src/preprocessing.py)

- [ ] Thống kê cấu trúc, phân phối, giá trị thiếu và ngoại lai.
- [ ] Xử lý outlier, Marital_Status lạ; bỏ ID, Z_CostContact, Z_Revenue.
- [ ] Đề xuất cách xử lý Income thiếu; tạo và giải thích đặc trưng mới.
- [ ] Bàn giao dữ liệu đã làm sạch, quy tắc xử lý và danh sách đặc trưng cho Tuananh.

### 02 · PCA — mục 2.2

File: [notebook 02](../notebooks/02_dimensionality_reduction.ipynb) · [dim_reduction.py](../src/dim_reduction.py)

- [ ] Triển khai PCA; lập bảng **mean, std, skewness (độ lệch)** của từng thành phần.
- [ ] Báo cáo explained variance và phương sai tích lũy.
- [ ] Vẽ từng cặp trong **4–6 thành phần**, phân biệt theo Response.
- [ ] Vẽ phân phối/boxplot của một số thành phần theo **Response=0/1** và giải thích quan hệ với nhãn.
- [ ] Bàn giao hàm PCA cho Tuananh ghép pipeline dùng chung với `k = max(1, floor(d/3))`.
- [ ] Ghép phương pháp thứ hai của Tuananh và cùng viết so sánh.

### 03 · K-Means — mục 2.3

File: [notebook 03](../notebooks/03_clustering.ipynb) · [clustering.py](../src/clustering.py)

- [ ] Chọn K bằng Elbow và Silhouette.
- [ ] Mô tả khách hàng, độ đo và tỷ lệ Response trong từng cụm.
- [ ] Vẽ màu theo cụm, ký hiệu theo nhãn; so sánh với GMM/DBSCAN của Tuananh.

### 04 · Naive Bayes và SVM — mục 2.4(a–b)

File: [notebook 04](../notebooks/04_classification.ipynb) · [classification.py](../src/classification.py)

- [ ] Triển khai **Gaussian Naive Bayes** và **SVM RBF**.
- [ ] So sánh SVM mặc định với `class_weight='balanced'`; NB mặc định với trọng số mẫu cân bằng chỉ tính từ train.
- [ ] Nêu hạn chế giả định Gaussian trên cột one-hot; giải thích ảnh hưởng và hướng kiểm chứng theo mục 7.
- [ ] Chạy đủ kịch bản phân loại trong mục 4 bên dưới cho cả hai mô hình.
- [ ] Nhận xét overfit; điều chỉnh `var_smoothing` của NB, `C` của SVM khi phù hợp.
- [ ] Xuất bảng độ đo, confusion matrix, ROC/PR và giải thích kết quả.
- [ ] Viết **kết luận riêng cho Naive Bayes và SVM: mô hình có phù hợp không, vì sao**, dựa trên dự đoán–thực tế, lỗi FP/FN, độ đo lớp 1 và chênh lệch train/validation.
- [ ] Bàn giao hàm và kết quả cho Tuananh ghép notebook 04.

### 05 · Mô hình hồi quy thứ nhất — mục 2.4(c)

File: [notebook 05](../notebooks/05_classification_to_regression.ipynb) · [regression.py](../src/regression.py)

- [ ] Nhận đầu ra xác suất và tập chia từ Tuananh.
- [ ] Triển khai **KNeighborsRegressor hoặc MLPRegressor**.
- [ ] Chạy đủ kịch bản hồi quy, xuất RMSE/MAE/R² và nhận xét.
- [ ] Bàn giao hàm và kết quả cho Tuananh ghép notebook 05.

### Tài liệu và hàm dùng chung

- [ ] Tổng hợp các hàm vẽ dùng chung trong [visualization.py](../src/visualization.py).
- [ ] Viết lý thuyết, thực nghiệm và nhận xét cho các mô hình mình làm.
- [ ] Tổng hợp slides trong `slides/`; nhận nội dung từ Tuananh.
- [ ] **Đọc duyệt báo cáo cuối** sau khi Tuananh ghép: nội dung, số liệu, hình/bảng, kết luận và định dạng; gửi lỗi để Tuananh sửa trước khi nộp.

## 3. Checklist của Tuananh

### 01 · Pipeline và chia tập — mục 2.1

File: [notebook 01](../notebooks/01_preprocessing.ipynb) · [preprocessing.py](../src/preprocessing.py) · [config.py](../src/config.py)

- [ ] Nhận dữ liệu và quy tắc làm sạch từ ChinhBui.
- [ ] Xây dựng imputer cho Income, mã hóa category và chuẩn hóa.
- [ ] Tạo các tập chia stratified **80:20, 70:30, 60:40**, dùng seed cố định.
- [ ] Viết cấu hình; bàn giao pipeline và chỉ số dòng của các tập chia.
- [ ] Ghép notebook 01, mô tả dữ liệu sau xử lý và kiểm tra chạy từ đầu đến cuối.

### 02 · Kernel PCA — mục 2.2

File: [notebook 02](../notebooks/02_dimensionality_reduction.ipynb) · [dim_reduction.py](../src/dim_reduction.py)

- [ ] Triển khai **Kernel PCA với kernel RBF** có `transform`; ghi rõ tham số `gamma` và cách chọn.
- [ ] Lập bảng **mean, std, skewness**; vẽ các cặp trong **4–6 thành phần** và quan hệ của một số thành phần với Response.
- [ ] Báo cáo phương sai và tỷ lệ phương sai **trong không gian kernel**, không đồng nhất với phương sai dữ liệu gốc; cách tính ở mục 7.
- [ ] Nhận hàm PCA của ChinhBui; ghép và bàn giao pipeline **PCA còn 1/3 số chiều** dùng chung cho phân loại/hồi quy, fit riêng từng tập chia.
- [ ] Bàn giao Kernel PCA cho ChinhBui ghép notebook 02; cùng so sánh PCA và Kernel PCA.

### 03 · GMM hoặc DBSCAN — mục 2.3

File: [notebook 03](../notebooks/03_clustering.ipynb) · [clustering.py](../src/clustering.py)

- [ ] Triển khai **GMM hoặc DBSCAN** và chọn tham số phù hợp.
- [ ] Mô tả khách hàng, độ đo và tỷ lệ Response trong từng cụm.
- [ ] Vẽ màu theo cụm, ký hiệu theo nhãn; nếu dùng DBSCAN, báo cáo điểm nhiễu.
- [ ] Bàn giao cho ChinhBui ghép notebook 03 và so sánh với K-Means.

### 04 · Logistic Regression — mục 2.4(a–b)

File: [notebook 04](../notebooks/04_classification.ipynb) · [classification.py](../src/classification.py)

- [ ] Triển khai **Logistic Regression** và chạy đủ kịch bản phân loại.
- [ ] So sánh mặc định với `class_weight='balanced'`; đánh giá theo các độ đo ở mục 4.
- [ ] Nhận xét overfit; điều chỉnh `C`, L1/L2 với solver phù hợp.
- [ ] Xuất bảng độ đo, confusion matrix, ROC/PR và giải thích kết quả.
- [ ] Nhận Naive Bayes/SVM từ ChinhBui; ghép notebook 04 và bảng so sánh ba mô hình.
- [ ] Viết **kết luận Logistic có phù hợp không, vì sao**, dựa trên dự đoán–thực tế, lỗi FP/FN, độ đo lớp 1 và chênh lệch train/validation.

### 05 · Tạo đầu ra và mô hình hồi quy thứ hai — mục 2.4(c)

File: [notebook 05](../notebooks/05_classification_to_regression.ipynb) · [regression.py](../src/regression.py)

- [ ] Tạo đầu ra liên tục từ `predict_proba` cho lớp **Response=1** của Logistic.
- [ ] Bàn giao sớm đầu ra, mô hình nguồn và các tập chia cho ChinhBui.
- [ ] Triển khai **LinearRegression hoặc Ridge**, chạy đủ kịch bản hồi quy.
- [ ] Xuất RMSE/MAE/R²; nhận kết quả ChinhBui và ghép bảng so sánh trong notebook 05.

### Tài liệu và hàm dùng chung

- [ ] Tổng hợp các hàm đánh giá trong [evaluation.py](../src/evaluation.py) theo giao diện hai người đã chốt.
- [ ] Viết [DATA_GUIDE.txt](../DATA_GUIDE.txt): link Kaggle, tổ chức dữ liệu, kịch bản thực nghiệm.
- [ ] Viết [README.md](../README.md), cập nhật [requirements.txt](../requirements.txt).
- [ ] Viết báo cáo phần mình và bàn giao nội dung slides cho ChinhBui.
- [ ] **Ghép báo cáo trong `report/`**, thống nhất đánh số mục/hình/bảng, tài liệu tham khảo và định dạng **Times New Roman 13, căn đều, giãn dòng 1.3–1.5**.
- [ ] Gửi bản ghép cho ChinhBui đọc duyệt; sửa các điểm được ghi nhận và xuất bản cuối để nộp.

## 4. Kịch bản thực nghiệm dùng chung

| Bài toán | Đầu vào cần chạy | Tỷ lệ chia | Kết quả cần lưu |
|---|---|---|---|
| **Phân loại · 2.4(a–b)** | Không giảm chiều **và PCA còn 1/3 số chiều** | **80:20 · 70:30 · 60:40** | Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC trên train/validation; confusion matrix, ROC/PR; overfit và kết luận độ phù hợp |
| **Hồi quy · 2.4(c)** | Không giảm chiều **và PCA còn 1/3 số chiều** | **80:20 · 70:30 · 60:40** | RMSE, MAE, R²; bảng so sánh và giải thích |
| **Phân cụm · 2.3** | Đặc trưng **không có Response** | Không bắt buộc ba tỷ lệ trên | Silhouette, hồ sơ/tỷ lệ Response theo cụm, hình phân biệt cụm và nhãn |

- **Cùng điều kiện:** dùng chung tập chia, seed, biểu diễn đầu vào và đầu ra hồi quy khi so sánh mô hình.
- **Chốt biểu diễn:** PCA với `k = max(1, floor(d/3))` là nhánh giảm chiều chung cho 2.4(a) và 2.4(c). Mỗi tập chia có pipeline riêng; nếu dùng cross-validation thì fit lại trong từng fold. Kernel PCA RBF dùng cho phân tích và so sánh ở 2.2, không bắt buộc mở thêm nhánh thực nghiệm 2.4.
- **Nhãn mất cân bằng:** dữ liệu gốc có 334/2.240 mẫu Response=1 (khoảng **14,9%**); thống kê lại sau làm sạch. Luôn báo cáo độ đo cho **lớp dương 1**, không kết luận chỉ bằng Accuracy; ưu tiên phân tích Recall, F1 và PR-AUC.
- **Cách tính:** Precision/Recall/F1 dùng `pos_label=1`; PR-AUC tính diện tích hình thang dưới đường Precision–Recall từ điểm dự đoán, ROC-AUC cũng dùng điểm/xác suất, không dùng nhãn cứng. Nếu báo cáo thêm Average Precision (AP), ghi riêng vì AP không đồng nhất PR-AUC hình thang.
- **Cân bằng lớp:** so sánh bản mặc định với `class_weight='balanced'` cho Logistic/SVM. Gaussian NB không có tham số này; thử `sample_weight` nghịch tần suất lớp tính từ tập train. Chọn cấu hình/ngưỡng trong train bằng cross-validation; giữ nguyên phân phối validation/test và báo cáo ảnh hưởng đến Precision/Recall.
- **Lưu kết quả:** hình vào `results/figures/<bài_toán>/`, bảng vào `results/tables/`; ghi mô hình, tỷ lệ, phương pháp giảm chiều, số chiều và seed.
- **Đủ ba nhóm phân loại:** Gaussian NB (sinh), Logistic (phân biệt, tuyến tính), SVM RBF (phân biệt, phi tuyến). Không có Decision Tree/KNN trong phần phân loại; KNN Regressor vẫn là lựa chọn ở phần hồi quy.
- Các mô hình dùng để đáp ứng yêu cầu đề phải thuộc nội dung đã học. Phương pháp phân cụm thứ hai là phần nhóm chọn làm thêm so với mức tối thiểu một phương pháp.

## 5. Bàn giao và ghép file

**Lịch nội bộ:** bắt đầu **04/11/2026**, hoàn thành trong hai tuần đến **18/11/2026**. Các hạn dưới đây tính đến cuối ngày theo giờ Việt Nam; điều chỉnh nếu lịch nộp chính thức của môn học sớm hơn.

| Bước | Người giao → người nhận | Nội dung bàn giao | Hạn nội bộ | Mở việc tiếp theo |
|---|---|---|---|---|
| **1** | ChinhBui → Tuananh | Dữ liệu sạch, quy tắc, đặc trưng | **06/11/2026** | Ghép pipeline tiền xử lý |
| **2** | Tuananh → ChinhBui | Pipeline, seed, chỉ số tập chia | **07/11/2026** | Chạy các mô hình cùng điều kiện |
| **3** | Tuananh → ChinhBui | Kernel PCA; pipeline PCA còn 1/3 chiều từ hàm ChinhBui bàn giao chậm nhất 08/11 | **09/11/2026** | Ghép notebook 02, thí nghiệm sau giảm chiều |
| **4 · Ưu tiên** | Tuananh → ChinhBui | Xác suất Logistic của lớp 1, mô hình nguồn và chỉ số dòng cho đủ ba tập chia | **08/11/2026** | **Mở phần hồi quy của ChinhBui; không chờ bước 3** |
| **5** | Hai người → người tổng hợp | Hàm, bảng, hình, nhận xét và nội dung báo cáo | **14/11/2026** | Tuananh ghép báo cáo chậm nhất 16/11; ChinhBui duyệt cuối chậm nhất 17/11 |

**Người tổng hợp notebook:** `01`: Tuananh · `02`: ChinhBui · `03`: ChinhBui · `04`: Tuananh · `05`: Tuananh.

**Báo cáo:** Tuananh ghép và định dạng; ChinhBui đọc duyệt cuối; Tuananh sửa và xuất bản hoàn chỉnh ngày **18/11/2026**. Bản giữa kỳ dùng kết quả tuần 1: ghép ngày **09/11**, đọc duyệt ngày **10/11**, sửa và hoàn thiện ngày **11/11**. ChinhBui tổng hợp slides tương ứng với mỗi bản báo cáo.

Người tổng hợp ghép phần của cả hai và kiểm tra chạy từ đầu đến cuối. Thống nhất giao diện hàm trước; tránh sửa đồng thời cùng notebook. Các nhánh mô hình có thể làm song song sau bước 2; ưu tiên bàn giao Logistic sớm để phần hồi quy không phải chờ toàn bộ thí nghiệm phân loại.

## 6. Checklist chung và mốc nộp

### Trước khi làm riêng

- [ ] Chốt schema, chỉ số dòng sau làm sạch, danh sách đặc trưng và nhãn.
- [ ] Chốt seed, ba tập chia, giao diện hàm đánh giá; xác nhận triển khai PCA và Kernel PCA RBF.
- [ ] Chốt cách fit pipeline và cách tạo đầu ra hồi quy theo mục 7.

### Giữa kỳ

**Mục tiêu nội bộ: 11/11/2026 — chưa phải hạn nộp chính thức.** Cả hai bàn giao nội dung và kết quả giữa kỳ trong ngày **09/11** để Tuananh ghép; ChinhBui đọc duyệt ngày **10/11**, cả hai hoàn thiện ngày **11/11**.

- [ ] Hoàn tất tiền xử lý và chuẩn bị các tài liệu yêu cầu ở đầu đề.
- [ ] Hoàn thiện phần giảm chiều/trực quan hóa gồm PCA và Kernel PCA.
- [ ] Có ít nhất hai mô hình phân loại; ưu tiên Naive Bayes và Logistic, kèm kết quả để trình bày.

### Cuối kỳ

**Mục tiêu nội bộ: 18/11/2026 — chưa phải hạn nộp chính thức.** Chốt kết quả và nội dung ngày **14/11**, ghép báo cáo chậm nhất **16/11**, đọc duyệt chậm nhất **17/11**; dành ngày **18/11** để sửa, kiểm tra gói bàn giao và hoàn thiện.

- [ ] Đủ mục 2.1–2.4, đủ các kịch bản và bảng/hình so sánh.
- [ ] Mỗi người viết phần mình; cùng viết mở đầu, so sánh chung và kết luận.
- [ ] Tuananh ghép báo cáo khoảng **15–20 trang A4**, Times New Roman **13**, căn đều, giãn dòng **1.3–1.5**; có giới thiệu, lý thuyết, dữ liệu/thực nghiệm và kết luận.
- [ ] ChinhBui đọc duyệt bản cuối; Tuananh hoàn tất sửa và xuất bản nộp.
- [ ] Hoàn thiện slides, mã nguồn, README và file text hướng dẫn dữ liệu/kịch bản.
- [ ] Chạy kiểm tra chéo notebook của nhau theo hướng dẫn.
- [ ] Mỗi người giải thích và trình bày được ít nhất một mô hình mình làm.
- [ ] Loại dữ liệu khỏi gói nộp, cung cấp link tải; kiểm tra cả ZIP vì `.gitignore` không tự loại file khi nén.

## 7. Ghi chú kỹ thuật khi triển khai

<details>
<summary><strong>Giảm chiều: explained variance và quy ước 1/3 số chiều</strong></summary>

- Chốt **PCA và Kernel PCA RBF**. Đề 2.2 yêu cầu explained variance: PCA đo phương sai trong không gian đặc trưng đầu vào đã tiền xử lý; Kernel PCA đo phương sai trong không gian đặc trưng do kernel tạo ra. Không so sánh trực tiếp hai tỷ lệ như cùng đo lượng thông tin của dữ liệu gốc.
- Với `n` mẫu train và ma trận kernel đã định tâm `Kc`, phương sai thành phần kernel thứ `i` là `lambda_i/(n-1)`; tỷ lệ giữ lại `k` thành phần là `sum(lambda_1...lambda_k)/trace(Kc)` khi mẫu số dương. Kernel RBF là kernel nửa xác định dương; kiểm tra sai số số học nếu có trị riêng âm rất nhỏ.
- Mẫu số dùng **toàn bộ phương sai kernel** (vết Kc, tương đương tổng toàn bộ trị riêng), không dùng tổng trị riêng của riêng các thành phần giữ lại. Báo cáo kernel, gamma và quy tắc định tâm; scikit-learn không cung cấp sẵn `explained_variance_ratio_` cho KernelPCA.
- Không dùng TruncatedSVD để thay phương pháp thứ hai trong kế hoạch này: trên cùng ma trận đã căn giữa, nghiệm SVD liên hệ trực tiếp với PCA, không tạo so sánh tuyến tính–phi tuyến như Kernel PCA RBF. Lưu ý scaler cột số không tự căn giữa các cột one-hot.
- Số chiều gốc `d` là số đặc trưng sau mã hóa trên train, không gồm nhãn. Lấy `max(1, floor(d/3))` và ghi rõ cách làm tròn.
- Biểu diễn 2D để vẽ không thay thế phần phân tích 4–6 thành phần hoặc biểu diễn dùng huấn luyện.
- Tham khảo: [Kernel PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html).

</details>

<details>
<summary><strong>Tiền xử lý, chia tập và tránh rò rỉ dữ liệu</strong></summary>

- ChinhBui xác định quy tắc xử lý Income; Tuananh triển khai imputer. Median/mean chỉ học trên train.
- Tập chia stratified theo Response, seed cố định. Phần giữ lại gọi là validation trong phân loại, test trong mục hồi quy 2.4(c).
- Với đánh giá có giám sát, imputer, encoder, scaler và giảm chiều chỉ fit trên train; validation/test chỉ transform. Khi dò tham số bằng cross-validation, fit lại pipeline trong từng fold train.
- Bàn giao hàm/pipeline cho từng tập chia, không dùng duy nhất ma trận đã fit trên toàn bộ dữ liệu để đánh giá mô hình.
- Phân cụm khám phá có thể fit trên toàn bộ tập khảo sát sau khi bỏ Response; không tái sử dụng transformer đó cho đánh giá có giám sát.
- So sánh Silhouette trên cùng không gian đặc trưng; với DBSCAN, báo cáo số điểm nhiễu và quy tắc xử lý nhiễu khi tính độ đo.

</details>

<details>
<summary><strong>Gaussian Naive Bayes, one-hot và độ đo lớp thiểu số</strong></summary>

- Gaussian NB giả định phân phối Gaussian cho từng đặc trưng theo lớp; các cột one-hot 0/1 không thỏa giả định liên tục này, các cột mã hóa từ cùng category cũng phụ thuộc nhau. ChinhBui phải nêu hạn chế và liên hệ với lỗi thực nghiệm; `var_smoothing` không làm giả định trở nên đúng.
- Giữ Gaussian NB làm mô hình chính theo phân công. Hướng kiểm chứng bổ sung, không bắt buộc: mô hình NB hỗn hợp dùng Gaussian cho cột số và Bernoulli cho cột nhị phân; cộng log-likelihood và chỉ cộng prior lớp **một lần**, đồng thời nêu hạn chế độc lập. Không áp dụng Bernoulli cho các thành phần PCA liên tục và không tự đổi bộ ba mô hình chính.
- GaussianNB hỗ trợ `sample_weight` khi fit, không hỗ trợ `class_weight`. Trọng số cân bằng phải tính riêng trong mỗi fold train; báo cáo rằng chúng thay đổi phân phối lớp dùng để học và có thể ảnh hưởng xác suất dự đoán.
- Dùng PR-AUC hình thang theo quy ước ở mục 4. Nếu bổ sung AP từ `average_precision_score`, đặt cột riêng; các giá trị khác nhau do cách tổng hợp đường PR.
- Tham khảo: [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html), [Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

</details>

<details>
<summary><strong>Tạo đầu ra hồi quy từ Logistic Regression</strong></summary>

- Tuananh tạo xác suất lớp Response=1; mỗi tỷ lệ chia có mô hình nguồn học từ train, không fit bằng nhãn test.
- Khuyến nghị tạo xác suất ngoài fold cho train; mô hình nguồn fit toàn bộ train tạo xác suất test. Nếu áp dụng, imputer/scaler cũng phải fit bên trong từng fold.
- Hai mô hình hồi quy dùng chung vector đầu ra này cho cả đầu vào nguyên bản và giảm chiều.
- Ghi rõ đây là dự đoán điểm của bộ phân loại nguồn. RMSE/MAE/R² không trực tiếp đo độ chính xác của xác suất hưởng ứng thực tế.

</details>
