# Bàn giao tiền xử lý — ChinhBui → Tuananh

**Cập nhật:** 05/11/2026. **Phạm vi:** mục 2.1, pipeline train-only cho ba tập chia và đầu vào PCA còn 1/3 số chiều.

**Trạng thái:** notebook đã chạy toàn bộ bằng kernel Python qua nbclient, lưu bảng và 7 biểu đồ; **17 kiểm tra tự động đạt**. Cấu hình Jupyter/IPython/Matplotlib dùng thư mục tạm được phép ghi. Test và toàn bộ file splits giữ nguyên nội dung và thời gian sửa đổi. Bản trong Downloads được giữ nguyên; bản đã sửa nằm tại `notebooks/01_preprocessing.ipynb`.

## 1. Kết quả hiện tại

| Chỉ tiêu | Kết quả |
|---|---:|
| Dữ liệu gốc | 2.240 dòng × 29 cột |
| Dòng trùng hoàn toàn / ID trùng | 0 / 0 |
| Dòng bị loại do năm sinh | 3 |
| Dòng bị loại do Income > 200.000 | 1 |
| Dữ liệu bàn giao | **2.236 dòng × 32 cột** |
| Đặc trưng đầu vào | **31: 29 số + 2 category** |
| Nhãn | Response: 1.902 mẫu lớp 0, 334 mẫu lớp 1 (khoảng 14,94%) |
| Giá trị thiếu giữ lại | Income: 24; Family_Size: 4 |
| Nhóm đầu vào có nhiều dòng | **194 nhóm** |
| Dòng trùng đầu vào sau dòng đầu của mỗi nhóm | **203** |

Các con số được xuất tự động vào [preprocessing_summary.json](../results/tables/preprocessing_summary.json). File sạch và splits vẫn chưa impute, encode hoặc scale. Đã tách **448 mẫu test cố định** và **1.788 mẫu development**. Bản train/validation đã biến đổi nằm riêng trong `data/processed/experiments/`; xem [quy tắc chia tập](data_split_protocol.md).

## 2. File nhận bàn giao

| File | Nội dung |
|---|---|
| `data/processed/cleaned_marketing_campaign.csv` | Dữ liệu sạch/tạo đặc trưng, gồm nhãn Response |
| `data/processed/customer_row_metadata.csv` | Ánh xạ dòng và nhóm đầu vào trùng; cùng thứ tự với file dữ liệu |
| `data/processed/splits/` | Test gốc/test sạch, development, ba cặp train/validation, assignments và manifest; chưa chuẩn hóa/giảm chiều |
| [preprocessing_feature_schema.csv](../results/tables/preprocessing_feature_schema.csv) | Tên cột, kiểu dữ liệu, vai trò, số giá trị thiếu và số giá trị khác nhau |
| [preprocessing_raw_profile.csv](../results/tables/preprocessing_raw_profile.csv) / [cleaned_profile.csv](../results/tables/preprocessing_cleaned_profile.csv) | Cấu trúc và tỷ lệ thiếu trước/sau |
| [preprocessing_raw_statistics.csv](../results/tables/preprocessing_raw_statistics.csv) / [cleaned_statistics.csv](../results/tables/preprocessing_cleaned_statistics.csv) | Count, mean, std, min, các tứ phân vị, max của cột số |
| [preprocessing_category_counts.csv](../results/tables/preprocessing_category_counts.csv) | Phân phối category và Response trước/sau |
| [notebook 01](../notebooks/01_preprocessing.ipynb) | Diễn giải, mã thống kê và biểu đồ; phần chuẩn hóa toàn bộ tập cũ đã được thay bằng bước bàn giao |

Các file dữ liệu trong `data/processed/`, kể cả thư mục `splits/`, được `.gitignore` loại khỏi Git. Tuananh cần lấy dữ liệu gốc và chạy lại lệnh ở mục 5 nếu nhận project qua GitHub; không gửi kèm dữ liệu khi nộp bài.

## 3. Quy tắc làm sạch và đặc trưng

| Nội dung | Quy tắc đã áp dụng | Lý do / giới hạn |
|---|---|---|
| Schema | Kiểm tra đủ cột, trường số, nhãn 0/1 và ngày dd-mm-yyyy; báo lỗi rõ khi sai | Không tự suy đoán nhãn hoặc định dạng ngày |
| Năm sinh | Giữ [1920, 2014] | Ngưỡng khảo sát cố định; loại ba năm 1893, 1899, 1900. Năm 1920 ứng với 94 tuổi tại mốc 2014, không phải quy tắc Age > 100 |
| Income | Loại > 200.000; giữ đúng 200.000; âm → NaN; giữ NaN sẵn có | Loại một dòng 666.666. Đây là quyết định xử lý ngoại lai, không phải khẳng định mọi thu nhập cao đều sai; cần nêu phạm vi áp dụng trong báo cáo |
| Imputation | Median của train cho Income và Family_Size | Đã fit riêng từng tập train; khi CV phải fit lại trong từng fold |
| ID / cột hằng | Bỏ ID, Z_CostContact, Z_Revenue khỏi đầu vào | ID còn trong metadata để đối chiếu |
| Marital_Group | Married/Together → Partner; Single/Divorced/Widow/Alone → Single; Absurd/YOLO/nhãn lạ → Unknown | Không gán nhãn vô nghĩa thành Single |
| Education_Level | Giữ nguyên 5 nhãn Education | Không tự suy đoán `2n Cycle` tương đương mức bằng cấp khác |
| Age | 2014 − Year_Birth | Tuổi xấp xỉ theo năm |
| Customer_Tenure_Days | **2014-06-30 − Dt_Customer** | Mốc quy ước cố định ngay sau ngày đăng ký lớn nhất quan sát được; không phải ngày thu thập đã được nguồn xác nhận. Không lấy max ngày riêng cho từng batch |
| Total_Spent | Tổng 6 cột Mnt* | Chi tiêu quan sát, không phải biến đầu ra cần dự báo |
| Total_Purchases | NumWebPurchases + NumCatalogPurchases + NumStorePurchases | Không cộng NumDealsPurchases vì giảm giá không phải kênh độc lập; giữ biến này riêng |
| Total_Accepted_Cmp | Tổng AcceptedCmp1–5 | Không bao gồm Response; giả định đây là lịch sử sẵn có trước chiến dịch mục tiêu |
| Children / Is_Parent | Kidhome + Teenhome / tổng > 0 | Hai biến liên quan trực tiếp |
| Family_Size | Children + 2 nếu Partner, +1 nếu Single, NaN nếu Unknown | Ước tính hộ gia đình; 4 trường hợp Unknown chưa biết số người lớn |

Nguồn ý nghĩa trường và chiến dịch: [Kaggle — Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis). Các tổng và thành phần tạo tương quan mạnh; cần lưu ý khi phân tích PCA, khoảng cách và giả định độc lập của Naive Bayes. Dữ liệu không cung cấp đủ mốc thời gian để khẳng định mọi đặc trưng hành vi đều có trước chiến dịch mục tiêu; cần nêu giả định này khi diễn giải khả năng dự báo.

## 4. Hợp đồng dữ liệu và tránh rò rỉ

**File chính:** một dòng cho mỗi khách hàng được giữ lại; `Response` là nhãn. Không chứa ID hoặc thông tin nhóm. Hai category là `Marital_Group` và `Education_Level`; còn lại là số.

**Metadata:** `processed_row` là vị trí dòng 0-based trong file sạch; `source_row` là vị trí dòng 0-based trong CSV gốc không tính header; `customer_id` là ID gốc; `duplicate_group` là mã nhóm tạo từ toàn bộ X sau tạo đặc trưng, **không sử dụng Response**. Không đưa bất kỳ cột metadata nào vào mô hình. Giữ hai file đồng bộ khi sắp xếp/lọc dữ liệu.

Có 182 dòng lặp hồ sơ gốc khi bỏ ID. Giữ chúng vì khác ID không đủ chứng minh là bản ghi sai. Sau tạo đặc trưng và bỏ nhãn, có 203 dòng lặp đầu vào sau dòng đầu, thuộc 194 nhóm; nhóm có thể chứa các nhãn khác nhau. Tuananh cần:

1. Dùng các tập đã chốt trong `data/processed/splits/`: test cuối 448 dòng, các tỷ lệ 80:20, 70:30, 60:40 chỉ trên development 1.788 dòng. Giữ nguyên nhóm trong cross-validation, không tự chia lại toàn bộ dataset.
2. Kiểm tra `set(train_groups).isdisjoint(validation_groups)`; báo cáo tỷ lệ hàng và nhãn thực tế, vì nhóm không phải lúc nào cũng chia đúng tỷ lệ hàng tuyệt đối.
3. Chỉ fit imputer, encoder, scaler và giảm chiều trên train/fold train. Factory `make_feature_pipeline` trong `src/experiment_preprocessing.py` tạo pipeline mới chưa fit. Notebook đã lưu pipeline fit riêng cho mỗi tập chia; khi tuning phải clone/tạo mới và fit toàn bộ pipeline trong từng fold.
4. Giữ NaN đến lúc fit imputer; loại Response khỏi X. Không dùng lại file `transformed_marketing_campaign.csv` từ phiên bản cũ nếu có trên máy.

Schema hiện có 31 đầu vào trước mã hóa. Nếu train có đủ các mức category, OneHotEncoder giữ toàn bộ mức sẽ tạo 37 đầu vào (29 + 3 + 5); không hard-code con số này vì số mức phải được học từ train.

## 5. Chạy lại và kiểm tra

Từ thư mục gốc project, sau khi đặt file Kaggle tại `data/raw/marketing_campaign.csv`:

```powershell
python -m src.preprocessing
python -m src.data_splitting
python -m unittest discover -s tests -v
jupyter notebook notebooks/01_preprocessing.ipynb
```

Trong Jupyter, chọn **Restart Kernel and Run All**. Notebook chạy được khi thư mục làm việc là gốc project hoặc `notebooks/`; tự tạo các thư mục xuất hình/bảng. Lệnh đầu chỉ xuất dữ liệu/thống kê, không xuất hình; biểu đồ được tạo khi chạy notebook.

**Đã kiểm tra:** 7 bài kiểm tra tiền xử lý, 7 bài kiểm tra chia tập/giữ nguyên test và 3 bài kiểm tra pipeline (validation không thay đổi tham số train, clone fit lại độc lập, từ chối nhãn/metadata trong X). Toàn bộ 10 cell code của notebook đã chạy thành công bằng kernel, không có output lỗi; đã kiểm tra trực quan biểu đồ phân bố và trước/sau chuẩn hóa.

**Việc tiếp theo:** Tuananh rà soát bàn giao và triển khai Kernel PCA; ChinhBui tiếp tục phân tích PCA đầy đủ ở notebook 02. Chưa hoàn thành yêu cầu so sánh hai phương pháp giảm chiều hay các mô hình dự đoán. Test giữ nguyên; chỉ transform trong bộ nhớ bằng pipeline cuối khi đánh giá cuối.

## 6. Đầu ra đã fit trên train

| Kịch bản | Train | Validation | Sau one-hot | PCA | Tổng tỷ lệ phương sai PCA |
|---|---:|---:|---:|---:|---:|
| 80:20 | 1.430 | 358 | 37 | 12 | 80,17% |
| 70:30 | 1.250 | 538 | 37 | 12 | 80,20% |
| 60:40 | 1.072 | 716 | 37 | 12 | 80,49% |

Mỗi thư mục `data/processed/experiments/<tỷ_lệ>/` có `X_train_encoded.csv`, `X_validation_encoded.csv`, hai bản `X_*_pca.csv`, `y_train.csv`, `y_validation.csv`, metadata train/validation và `feature_pipeline.joblib`. Các file cùng vai trò có cùng thứ tự dòng. Pipeline nhận X sạch và trả về PCA; bước `named_steps['preprocessor']` trả về bản không giảm chiều.

Không dùng CSV đã biến đổi để tuning qua các fold mới: dùng X sạch, thêm mô hình vào pipeline và truyền nhóm trùng cho StratifiedGroupKFold. Báo cáo kiểm tra ở [train_preprocessing_audit.json](../results/tables/train_preprocessing_audit.json), bảng kết quả ở [train_preprocessing_summary.csv](../results/tables/train_preprocessing_summary.csv).
