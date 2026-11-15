# Kịch bản thuyết trình giữa kỳ

Đề tài: Phân nhóm khách hàng và dự đoán hưởng ứng chiến dịch marketing.  
Thành viên: ChinhBui, Tuananh.  
Gợi ý thời lượng: khoảng 12–15 phút cho 16 slide chính, điều chỉnh theo lịch lớp.
Phân loại chính chỉ trình bày NB và SVM. Logistic ở phần phụ lục hoặc kế hoạch cuối kỳ.
Các hình là kết quả thật trong repo; không dùng ảnh minh họa thay biểu đồ thực nghiệm.

## Slide 01 Đề tài và thành viên

**Trên slide**

Phân nhóm khách hàng và dự đoán hưởng ứng chiến dịch marketing  
Báo cáo giữa kỳ Machine Learning  
ChinhBui và Tuananh

**Người nói:** ChinhBui. Khoảng 20 giây.

**Lời dẫn:** Nhóm dùng dữ liệu khách hàng để khảo sát các nhóm hành vi và dự đoán
khả năng hưởng ứng chiến dịch. Hôm nay trình bày quy trình dữ liệu, giảm chiều,
phân cụm và hai mô hình phân loại Naïve Bayes, SVM.

## Slide 02 Mục tiêu và nhãn dự đoán

**Trên slide**

- Phân nhóm hồ sơ khách hàng bằng K-Means.
- Dự đoán `Response=1` cho chiến dịch cuối.
- Đánh giá trước và sau PCA trên cùng tập chia.
- Phạm vi giữa kỳ: NB và SVM.

**Người nói:** ChinhBui. Khoảng 35 giây.

**Lời dẫn:** Nhãn hưởng ứng chỉ dùng trong bài toán phân loại. Khi phân cụm và
giảm chiều, nhóm không đưa nhãn vào phép fit; nhãn được dùng để tô màu và giải
thích kết quả sau đó. Các AcceptedCmp trước được xem là lịch sử có sẵn.

## Slide 03 Dữ liệu khách hàng

**Trên slide**

- Kaggle Customer Personality Analysis.
- 2.240 mẫu, 29 trường ban đầu.
- 334 người hưởng ứng, khoảng 14,9%.
- Biến số và biến phân loại, Income có giá trị thiếu.

**Hình:** `results/figures/preprocessing/development_response_missing.png`.

**Người nói:** ChinhBui. Khoảng 40 giây.

**Lời dẫn:** Bộ dữ liệu mất cân bằng. Nếu chỉ dự đoán không hưởng ứng, Accuracy
đã gần 85%, nhưng không tìm được khách hàng hưởng ứng nào. Vì thế nhóm đọc
Recall, Precision, F1 và đường PR cùng với Accuracy.

**Nguồn:** https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

## Slide 04 Làm sạch và tạo đặc trưng

**Trên slide**

- Loại 4 mẫu theo ngưỡng năm sinh và Income, còn 2.236 mẫu.
- Bỏ ID và hai cột hằng khỏi đặc trưng.
- Giữ Income thiếu đến bước impute trên train.
- Tạo tuổi, thâm niên, tổng chi tiêu và số lần mua.
- Ngày tham chiếu cố định 30/06/2014.

**Người nói:** ChinhBui. Khoảng 45 giây.

**Lời dẫn:** Đây là các quy tắc làm sạch cố định trước khi chia. Median và tham
số chuẩn hóa thuộc bước học từ train. Hồ sơ giống nhau của các khách hàng khác
ID được giữ lại nhưng luôn nằm cùng tập để tránh trùng thông tin giữa train và
validation. Total_Purchases không cộng thêm số lần mua giảm giá như kênh thứ tư.

## Slide 05 Tập chia và chống rò rỉ dữ liệu

**Trên slide**

| Tập | Số mẫu |
|---|---:|
| Test cố định | 448 |
| Development | 1.788 |
| Train / validation 80:20 | 1.430 / 358 |
| Train / validation 70:30 | 1.250 / 538 |
| Train / validation 60:40 | 1.072 / 716 |

Imputer, encoder, scaler và PCA chỉ fit trên train, fit lại trong từng fold CV.

**Người nói:** Tuananh. Khoảng 55 giây.

**Lời dẫn:** Ba tỷ lệ được chia bên trong development. Test cuối giữ riêng cho
giai đoạn sau và chưa dùng đánh giá mô hình giữa kỳ. Chia có phân tầng nhãn và
giữ nhóm trùng. Cả hai biểu diễn dùng cùng dòng train/validation, tránh so sánh
hai mô hình trên các khách hàng khác nhau.

**Nguồn:** `results/tables/data_split_summary.csv`.

## Slide 06 PCA giữ 12 trong 37 chiều

**Trên slide**

- 31 đặc trưng trước mã hóa, 37 đặc trưng sau mã hóa.
- PCA 12 chiều theo quy tắc `floor(d/3)`.
- Giữ khoảng 80,17–80,49% phương sai trên ba train.
- Phương sai lớn không đồng nghĩa dự đoán Response tốt hơn.

**Hình:** `results/figures/dim_reduction/pca_80_20_explained_variance.png`.

**Người nói:** ChinhBui. Khoảng 55 giây.

**Lời dẫn:** PCA học các hướng có phương sai lớn nhất, không nhìn nhãn Response.
Nhóm phân tích mean, std, skewness và các cặp trong sáu thành phần đầu. PCA
dùng để phân loại giữ 12 thành phần, khác biểu diễn hai trục dùng minh họa.

## Slide 07 Thành phần PCA và Response

**Trên slide**

- Mean thành phần train gần 0 sau căn giữa.
- Std và skewness mô tả độ phân tán, độ bất đối xứng.
- Các lớp vẫn có vùng chồng lấn.
- Hệ số chiếu giúp diễn giải nhóm đặc trưng đóng góp.

**Hình:** `results/figures/dim_reduction/pca_80_20_response_boxplots.png`.

**Người nói:** ChinhBui. Khoảng 45 giây.

**Lời dẫn:** PC1 liên quan nhiều tới thu nhập và chi tiêu. Các trục khác mô tả
thêm hộ gia đình và lịch sử chiến dịch. Trục không tương quan trên train chưa
bảo đảm độc lập hay phân phối Gaussian trong từng lớp. Hình đầy đủ 15 cặp của
sáu PC được lưu trong notebook để trả lời khi cần.

## Slide 08 Kernel PCA RBF

**Trên slide**

- Cùng 1.430 mẫu train và cùng preprocessing với PCA.
- Gamma cố định 1/37, giữ 6 thành phần.
- Sáu thành phần giữ 39,21% phương sai kernel.
- KPCA3 có tương quan với Response khoảng 0,298 trên train.

**Hình:** `results/figures/dim_reduction/kpca_explained_variance.png`.

**Người nói:** Tuananh. Khoảng 60 giây.

**Lời dẫn:** RBF đo tương đồng giữa các mẫu rồi Kernel PCA phân tích kernel đã
định tâm. Tỷ lệ phương sai chia cho tổng phương sai toàn kernel, không chia cho
tổng sáu trị riêng giữ lại. Con số 39,21% không cùng thước đo với explained
variance đầu vào của PCA. Gamma không chọn bằng nhãn hoặc validation.

## Slide 09 So sánh PCA và Kernel PCA

**Trên slide**

| Cùng 6 thành phần | PCA | Kernel PCA RBF |
|---|---:|---:|
| Trustworthiness@10 train | 0,977 | 0,948 |
| Trustworthiness@10 validation | 0,967 | 0,930 |

PCA giữ lân cận tốt hơn trong cấu hình đang xét. Chưa kết luận về phân loại bằng Kernel PCA.

**Hình:** `results/figures/dim_reduction/pca_kpca_same_train.png`.

**Người nói:** Tuananh. Khoảng 55 giây.

**Lời dẫn:** Chỉ số đo bảo toàn lân cận so với cùng ma trận 37 chiều, không phải
Accuracy. Kết luận chỉ đúng với sáu thành phần và gamma hiện tại. PCA dễ diễn
giải theo hệ số tuyến tính, Kernel PCA mô tả quan hệ phi tuyến nhưng phụ thuộc
kernel và gamma. Phần phân loại của nhóm vẫn dùng nhánh PCA 12 chiều.

**Nguồn:** `results/tables/pca_kpca_comparison.csv`.

## Slide 10 K-Means chọn hai cụm

**Trên slide**

- Khảo sát K từ 2 đến 10.
- Silhouette train chọn K=2; Elbow gợi ý K=5.
- Chọn theo quy tắc Silhouette đã định trước.
- Response chỉ dùng mô tả cụm sau khi fit.

**Hình:** `results/figures/clustering/kmeans_80_20_elbow_silhouette.png`.

**Người nói:** ChinhBui. Khoảng 45 giây.

**Lời dẫn:** Elbow và Silhouette có thể đưa ra gợi ý khác nhau vì đo các khía
cạnh khác nhau. Nhóm không chọn K từ tỷ lệ Response. Silhouette chưa PCA khoảng
0,266 trên train, cho thấy cụm có cấu trúc nhưng không tách hoàn hảo.

## Slide 11 Hồ sơ hai nhóm khách hàng

**Trên slide**

| Train 80:20 | Cụm 0 | Cụm 1 |
|---|---:|---:|
| Số khách hàng | 882 | 548 |
| Median Income | 38.774,5 | 71.613 |
| Median Total_Spent | 97,5 | 1.170,5 |
| Tỷ lệ hưởng ứng | 9,64% | 23,36% |

**Người nói:** ChinhBui. Khoảng 45 giây.

**Lời dẫn:** Hai cụm khác nhau về mức chi tiêu và thu nhập. Nhóm chi tiêu cao
có tỷ lệ hưởng ứng cao hơn trong dữ liệu quan sát, nhưng mỗi cụm vẫn có cả hai
nhãn. Không gọi cụm là lớp hưởng ứng và không suy ra chi tiêu gây ra hưởng ứng.

**Nguồn:** `results/tables/kmeans_80_20_encoded_train_*`.

## Slide 12 Thiết kế phân loại NB và SVM

**Trên slide**

- Gaussian NB: cách tiếp cận sinh, giả định độc lập theo lớp.
- SVM RBF: cách tiếp cận phân biệt, ranh giới phi tuyến.
- Ba tỷ lệ, hai biểu diễn: 37 chiều và PCA 12 chiều.
- Mặc định, cân bằng lớp, tuned bằng 5-fold group-CV.
- Chọn tham số theo AP, đánh giá ở ngưỡng cố định.

**Người nói:** ChinhBui. Khoảng 60 giây.

**Lời dẫn:** NB dò var_smoothing; SVM dò C và gamma. Trọng số lớp cũng được
kiểm chứng. Mỗi fold fit lại toàn bộ pipeline. AP dùng chọn tham số; PR-AUC
hình thang được báo cáo riêng. SVM xuất margin, không coi margin là xác suất.

## Slide 13 Kết quả validation 80:20

**Trên slide**

| Mô hình tuned | Precision | Recall | F1 | FP / FN |
|---|---:|---:|---:|---:|
| NB chưa PCA | 0,486 | 0,630 | 0,548 | 36 / 20 |
| NB PCA | 0,923 | 0,222 | 0,358 | 1 / 42 |
| SVM chưa PCA | 0,611 | 0,407 | 0,489 | 14 / 32 |
| SVM PCA | 0,621 | 0,333 | 0,434 | 11 / 36 |

**Người nói:** ChinhBui. Khoảng 70 giây.

**Lời dẫn:** NB sau PCA có Precision cao nhưng bỏ sót nhiều người hưởng ứng.
Giảm chiều không tự động cải thiện phân loại. Với SVM, AP hai nhánh gần nhau
nhưng Recall/F1 ở ngưỡng hiện tại khác nhau. Mỗi tỷ lệ còn có kết quả riêng
trong báo cáo; không chọn một tỷ lệ tốt nhất như thể ba validation độc lập.

**Nguồn:** `results/tables/classification/midterm_nb_svm_comparison.csv`.

## Slide 14 Overfit và hiệu chỉnh

**Trên slide**

- SVM tuned chưa PCA: AP train 0,894; validation 0,579.
- SVM tuned PCA: AP train 0,767; validation 0,576.
- C và gamma giúp kiểm soát độ phức tạp, vẫn còn overfit.
- NB có hạn chế Gaussian trên one-hot và phụ thuộc giữa đặc trưng.

**Hình:** `results/figures/classification/nb_svm_80_20_default_balanced_tuned.png`.

**Người nói:** ChinhBui. Khoảng 65 giây.

**Lời dẫn:** Không nói mô hình hết overfit chỉ vì đã tuning. Trong đối chiếu
giữ gamma=scale và không cân bằng, giảm C từ 100 xuống 1 làm chênh lệch CV giảm
và điểm fold đánh giá tăng. Cân bằng lớp là thay đổi chi phí lớp, không đồng
nhất với regularization. var_smoothing không làm one-hot trở thành Gaussian.

## Slide 15 Dự đoán và giá trị sử dụng

**Trên slide**

- FP: liên hệ người không hưởng ứng.
- FN: bỏ sót người hưởng ứng.
- Cân bằng lớp thường tăng Recall và tăng FP.
- Chưa có chi phí kinh doanh để chốt mô hình triển khai.

**Hình:** `results/figures/classification/nb_svm_80_20_actual_vs_predicted.png`.

**Người nói:** ChinhBui. Khoảng 45 giây.

**Lời dẫn:** Hình dùng 80 khách hàng đầu validation theo thứ tự cố định, không
chọn mẫu mô hình đoán đúng. Kết luận dựa trên toàn bộ validation. Mô hình thích
hợp hay không còn phụ thuộc chi phí bỏ sót và chi phí liên hệ nhầm. Nhóm mới
đánh giá thực nghiệm, chưa có kết quả test cuối hay thử nghiệm kinh doanh.

## Slide 16 Kết luận và công việc cuối kỳ

**Trên slide**

- Đã có quy trình dữ liệu, hai phương pháp giảm chiều và K-Means.
- NB/SVM chạy đủ ba tỷ lệ, trước/sau PCA và hiệu chỉnh.
- Giảm chiều không bảo đảm tăng chất lượng phân loại.
- Cuối kỳ: trình bày đầy đủ Logistic và thêm hai mô hình hồi quy từ điểm phân lớp.

**Người nói:** Tuananh. Khoảng 45 giây.

**Lời dẫn:** Logistic đã được tích hợp nhưng phần chính giữa kỳ chỉ tập trung
hai mô hình theo kế hoạch. Phần hồi quy sẽ dự đoán điểm của Logistic, không phải
một nhãn thực tế liên tục đã có. Test cuối vẫn được giữ riêng. Hai thành viên
cần hoàn thiện nhận xét và đánh giá sau góp ý của giảng viên.

## Phụ lục hỏi đáp

**Vì sao dùng thư viện khác cách tự tính trong bài thực hành?**  
Tài liệu của thầy cũng có cách dùng sklearn. Nhóm dùng cùng phương pháp, thêm
pipeline và CV để kiểm soát thực nghiệm. Cần giải thích được thuật toán thay vì
chỉ đọc tên hàm. Không tuyên bố tự cài lõi thuật toán.

**Vì sao không giảm chiều test ngay?**  
Test cuối đang đóng băng. Khi đánh giá cuối cùng, nếu mô hình cần PCA thì test
phải transform bằng pipeline đã học, không fit lại và không ghi đè file test.

**PCA không dùng nhãn, vậy fit toàn dữ liệu có sao không?**  
Phép chiếu vẫn học phân phối X của validation/test. Để đánh giá khả năng khái
quát, nhóm chỉ fit trên train và áp dụng lên phần giữ lại.

**Kernel PCA giữ ít phần trăm hơn thì kém hơn không?**  
Không so trực tiếp hai phần trăm vì thuộc hai không gian. Bảng trustworthiness
dùng cùng không gian tham chiếu nhưng chỉ đánh giá lân cận, không thay phân loại.

**Vì sao chọn K=2 mà Elbow gợi ý 5?**  
Nhóm chọn Silhouette train làm quy tắc chính từ đầu. Elbow là thông tin bổ sung;
K=2 không chứng minh thị trường chỉ có đúng hai nhóm khách hàng.

**Tuned có F1 thấp hơn balanced thì tuning sai không?**  
Mục tiêu tuning là AP trong CV, không phải F1 của validation. Không chọn lại
tham số/ngưỡng từ validation sau khi đã xem kết quả.

**Gaussian NB có phù hợp với one-hot không?**  
One-hot không thỏa giả định Gaussian và các cột cùng category còn phụ thuộc.
Đây là hạn chế thực nghiệm được báo cáo. PCA không tự bảo đảm Gaussian theo lớp.

**Logistic đã xong chưa?**  
Đã có mã và kết quả ba tỷ lệ, hai biểu diễn, hai chế độ trọng số. Xem notebook
04 phần B/C. Phần phân loại chính giữa kỳ chỉ chọn NB và SVM để trình bày sâu.

**Dự báo xác suất thật bằng hồi quy chưa?**  
Chưa triển khai hồi quy. Đầu ra dự kiến là điểm của Logistic, không phải xác
suất thật được quan sát. RMSE thấp trên điểm nguồn không chứng minh xác suất thật đúng.
