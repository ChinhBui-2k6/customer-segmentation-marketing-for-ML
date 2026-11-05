# Tập test cố định và các thí nghiệm phát triển

**Ngày tạo:** 04/11/2026. **Seed:** 42. **scikit-learn:** 1.9.1. **Mục tiêu:** giữ khoảng 20% dữ liệu làm test cuối; phần còn lại dùng cho các tỷ lệ thực nghiệm trong đề.

## 1. Các tập đã lưu

Tất cả CSV dưới đây nằm trong `data/processed/splits/`, được Git bỏ qua theo quy tắc dữ liệu hiện có.

| File / tập | Số dòng | Lớp 0 | Lớp 1 | Tỷ lệ lớp 1 |
|---|---:|---:|---:|---:|
| `test.csv` | **448** | 381 | 67 | 14,96% |
| `development.csv` | **1.788** | 1.521 | 267 | 14,93% |
| `train_80_20.csv` | 1.430 | 1.217 | 213 | 14,90% |
| `validation_80_20.csv` | 358 | 304 | 54 | 15,08% |
| `train_70_30.csv` | 1.250 | 1.064 | 186 | 14,88% |
| `validation_70_30.csv` | 538 | 457 | 81 | 15,06% |
| `train_60_40.csv` | 1.072 | 912 | 160 | 14,93% |
| `validation_60_40.csv` | 716 | 609 | 107 | 14,94% |

Các tỷ lệ 80:20, 70:30, 60:40 được tính **trong development**, không phải trên cả 2.236 dòng. Ví dụ thí nghiệm 80:20 tương đương khoảng 64% train, 16% validation và 20% test của toàn bộ dữ liệu sạch. Các thí nghiệm dùng lại development nhưng không chứa bất kỳ dòng hoặc nhóm nào của test.

- **`test_raw.csv`:** 448 dòng với **29 cột gốc** lấy từ CSV Kaggle, kể cả ID. Giá trị/cột gốc được giữ nguyên, không impute, encode, scale hay giảm chiều. Đây là bản lưu để đối chiếu, không đưa ID vào mô hình.
- **`test.csv`:** cùng 448 khách hàng, **31 đặc trưng đã tạo + Response**; vẫn giữ category và NaN, chưa fit bất kỳ transformer nào. Đây là đầu vào cho pipeline cuối.
- **`assignments.csv`:** ánh xạ processed_row/source_row/customer_id/duplicate_group, nhãn partition và development_fold. Dòng test có fold = -1. Các vị trí dòng 0-based khớp bộ dữ liệu sạch đầy đủ trong file bàn giao.
- **`manifest.json`:** thông số chia, phiên bản sklearn, danh sách đặc trưng, fingerprint nguồn và checksum SHA-256 của từng file.
- **[data_split_summary.csv](../results/tables/data_split_summary.csv):** thống kê tổng hợp có thể đưa lên Git, không chứa hồ sơ từng khách hàng.

## 2. Cách chia và quy tắc sử dụng

1. Tạo nhóm theo toàn bộ X sau làm sạch/tạo đặc trưng, không dùng Response để tạo nhóm; giữ hồ sơ giống nhau cùng một phía.
2. Dùng `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, lấy **fold đầu tiên** làm test. Giữ nguyên kết quả; không chọn fold dựa trên điểm mô hình.
3. Chỉ trên development, chia 10 fold bằng cùng thuật toán và seed. Gộp 2/3/4 fold đầu làm validation tương ứng 20%/30%/40%; phần còn lại làm train. Validation giữa các thí nghiệm được lồng nhau theo quy tắc cố định.
4. Do nhóm không thể bị tách, tỷ lệ số dòng chỉ xấp xỉ mục tiêu; bảng trên ghi số thực tế. Mọi cặp train–validation và development–test đã được kiểm tra không giao nhóm/dòng, đầy đủ hai lớp và không bỏ sót mẫu.
5. Các notebook giảm chiều, phân cụm để xây mô hình, phân loại và hồi quy tiếp theo phải bắt đầu từ development hoặc các file train/validation đã lưu. Chọn đặc trưng, tham số, ngưỡng hoặc kỹ thuật cân bằng lớp bằng development, không dùng test cuối.
6. Khi cross-validation trong train, tiếp tục dùng group theo `assignments.csv`; không chuyển sang chia ngẫu nhiên từng dòng làm tách nhóm trùng.

Tài liệu thuật toán: [StratifiedGroupKFold — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

**Phạm vi kết luận:** thống kê/EDA ban đầu đã xem toàn bộ dataset trước khi chốt tập chia. Vì vậy đây là holdout được giữ cố định **từ thời điểm này**, không phải dữ liệu hoàn toàn chưa từng được quan sát. Các phân tích phục vụ lựa chọn mô hình từ nay dùng development; notebook 01 giữ vai trò ghi lại quá trình khảo sát/làm sạch ban đầu.

## 3. Test không bị giảm chiều hoặc ghi đè

Không có bước fit/transform trong module chia dữ liệu. Các file test giữ nguyên khi chạy lại. Không dùng test để học median, encoder, scaler, PCA hoặc Kernel PCA.

Sau khi chọn cấu hình bằng development, fit pipeline cuối trên development rồi đánh giá test một lần. Nếu mô hình dùng PCA, `pipeline.predict(X_test)` cần transform một bản trong bộ nhớ bằng PCA đã học trên development; **không fit trên test và không ghi đè `test.csv`/`test_raw.csv`**. Nếu không giảm chiều, pipeline không có PCA. Đầu vào và đầu ra đã biến đổi của các mô hình khác nhau không được lưu thay cho bản test gốc.

Riêng mục 2.4(c), phần giữ lại của ba thí nghiệm trên development vẫn là tập đánh giá hồi quy theo tỷ lệ đề yêu cầu; phân biệt nó với **test cuối 448 mẫu**. Mô hình Logistic tạo điểm chỉ fit trên train tương ứng, dùng điểm ngoài fold cho train và dự đoán điểm validation. Khi đánh giá cuối, mô hình nguồn được fit trên development, không dùng nhãn test; độ đo hồi quy so với điểm nguồn không thay thế đánh giá phân loại theo Response thật.

## 4. Chạy lại

Tại thư mục gốc project:

```powershell
python -m src.preprocessing
python -m src.data_splitting
python -m unittest discover -s tests -v
```

- Lần đầu tạo file chia tập; lần sau xác minh fingerprint/checksum và **không ghi lại các file đã chốt**, kể cả thời gian sửa file.
- Nếu dữ liệu, quy tắc tiền xử lý hoặc seed/tỷ lệ đổi, chương trình báo lỗi thay vì tự thay test. Cần chủ động thống nhất một phiên bản dữ liệu/thư mục xuất khác; không xóa tập cũ để thử nhiều tập test rồi chọn kết quả đẹp.
- Nếu file bị sửa hoặc mất, chương trình báo lỗi. Checksum phát hiện thay đổi khi kiểm tra lại, không phải cơ chế khóa ghi của hệ điều hành.
- Trên máy mới, dùng cùng dữ liệu, cấu hình và phiên bản thư viện ghi trong manifest để tái tạo. Gói dữ liệu vẫn không đưa vào Git.

Ví dụ mở một thí nghiệm, không đụng test:

```python
from src import config
import pandas as pd

train = pd.read_csv(config.SPLITS_DIR / "train_80_20.csv")
validation = pd.read_csv(config.SPLITS_DIR / "validation_80_20.csv")
X_train, y_train = train.drop(columns="Response"), train["Response"]
X_validation = validation.drop(columns="Response")
y_validation = validation["Response"]
# Fit pipeline chỉ trên X_train/y_train; dùng validation để so sánh.
```

Các CSV con giữ thứ tự dòng của tập đầy đủ. Nếu cần groups cho một file train/validation, dùng assignments: development_fold >= 2/3/4 là train; fold từ 0 đến dưới 2/3/4 là validation. Luôn lọc partition=development trước, không đưa fold=-1 của test vào validation.

**Kiểm tra đã thực hiện (05/11/2026):** 17 kiểm tra gồm tiền xử lý, chia tập và pipeline train-only; kiểm tra giữ nguyên dữ liệu gốc/NaN/category/số chiều, không giao nhóm, test không phụ thuộc số thí nghiệm nội bộ, chạy lại không đổi file, từ chối đổi nguồn/seed, phát hiện file bị chỉnh sửa và validation không thay đổi tham số học từ train. Notebook 01 đã chạy toàn bộ, xác nhận checksum và mtime của mọi file splits giữ nguyên.

Pipeline mới ở `src/experiment_preprocessing.py`: `make_feature_pipeline(X_train, reduce=True)` tạo imputer/encoder/scaler/PCA chưa fit, có thể clone trong CV. `reduce=False` tạo bản không PCA. Số chiều PCA là `max(1, floor(d/3))` với d học từ train sau mã hóa. Notebook xuất train/validation đã biến đổi vào `data/processed/experiments/`, không ghi vào thư mục splits và không transform test. Khi tuning, dùng X sạch và fit lại toàn bộ pipeline trong từng fold theo nhóm; không dùng các CSV đã transform sẵn cho CV.
