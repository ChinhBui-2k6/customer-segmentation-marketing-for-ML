# Phân nhóm khách hàng và dự đoán hưởng ứng marketing

Bài tập lớn Machine Learning của **ChinhBui và Tuananh**, sử dụng Customer
Personality Analysis. Nhãn phân loại là `Response` (1: hưởng ứng chiến dịch cuối).

## Phạm vi giữa kỳ

- Tiền xử lý, thống kê và chia dữ liệu cố định.
- PCA và Kernel PCA RBF; so sánh trên cùng train/validation 80:20.
- K-Means và mô tả các nhóm khách hàng.
- Phần trình bày chính: Gaussian Naïve Bayes và SVM RBF, trước/sau PCA,
  ba tỷ lệ train:validation và nhận xét overfit.
- Logistic đã được tích hợp, để đối chiếu bổ sung và hoàn thiện trình bày cuối kỳ.
- Chuyển sang hồi quy ở notebook 05 là việc **cuối kỳ**, chưa triển khai.

## Cài đặt

Chạy lệnh từ thư mục gốc của repo, không chạy trong `src/`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name customer-ml --display-name "Customer ML"
.\.venv\Scripts\python.exe -m jupyter notebook
```

Trong Jupyter, chọn kernel **Customer ML**. Môi trường đã kiểm chứng và phiên bản
cụ thể nằm trong `requirements-tested.txt`; file này là bản ghi môi trường,
không phải cam kết các phiên bản đều cài được trên mọi phiên bản Python.
Máy chạy kết quả hiện tại dùng Python 3.14 trên Windows.

## Dữ liệu

Tải `marketing_campaign.csv` từ
[Kaggle Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis),
đặt vào `data/raw/marketing_campaign.csv`. File dùng dấu tab, không phải dấu phẩy.
Không commit hoặc gửi kèm dữ liệu. Xem [DATA_GUIDE.txt](DATA_GUIDE.txt).

## Thứ tự chạy

| Notebook | Nội dung | Đầu ra chính |
|---|---|---|
| `01_preprocessing.ipynb` | Làm sạch, nhóm trùng, tập chia, pipeline | `data/processed/splits/`, `results/tables/` |
| `02_dimensionality_reduction.ipynb` | PCA, Kernel PCA và so sánh | `pca_*`, `kpca_*`, `pca_kpca_*` |
| `03_clustering.ipynb` | K-Means | `kmeans_*` |
| `04_classification.ipynb` | NB, SVM, Logistic và đối chiếu | `results/tables/classification/` |
| `05_classification_to_regression.ipynb` | Dự kiến cuối kỳ | Chưa thực hiện |

Chọn **Restart Kernel and Run All** trong từng notebook theo thứ tự.
Notebook 04 chạy lại toàn bộ tìm tham số và có thể mất nhiều phút.
Các file `.py` trong `src/` cung cấp hàm; chạy riêng chúng không thay cho notebook.

Cách chạy tự động từ môi trường đã có kernel `python3` trỏ đến cùng Python:

```powershell
python -m ipykernel install --user --name python3
python -X utf8 scripts/run_notebooks.py 01 02 03 04
python -m unittest discover -s tests -v
```

Runner dùng thư mục cache Jupyter tạm, lưu notebook sau khi chạy thành công và
ghi `results/tables/notebook_execution_audit.json`. Không ghi đè tập chia đã chốt.

## Quy tắc thực nghiệm

1. Làm sạch bằng quy tắc cố định và gán nhóm trùng từ đặc trưng, không dùng nhãn.
2. Tách test cuối 448 mẫu. Chia development 1.788 mẫu theo 80:20, 70:30, 60:40.
3. Imputer, encoder, scaler và PCA chỉ fit trên train, và fit lại trong từng fold CV.
4. Dữ liệu “gốc” trong so sánh là dữ liệu đã tiền xử lý, chưa giảm chiều: 37 cột.
   Nhánh PCA còn `floor(37/3)=12` thành phần; cùng mẫu, cùng nhãn giữa hai nhánh.
5. Phân tích PCA–Kernel PCA dùng cùng 6 thành phần trên train 80:20, khác mục đích
   với PCA 12 chiều dùng phân loại. Hai tỷ lệ phương sai thuộc hai không gian khác nhau.
6. Không dùng test cuối để chọn mô hình. Các kết quả giữa kỳ đều là train/validation.

## Đọc kết quả và trình bày

- [Báo cáo nội dung giữa kỳ](report/midterm_report.md).
- [Nội dung và lời trình bày slides](slides/midterm_slides.md).
- [Kiểm tra sẵn sàng giữa kỳ](docs/midterm_readiness.md).
- [Phân công](docs/task_assignment.md) và các file `docs/*_handoff.md`.
- Bảng so sánh giảm chiều: `results/tables/pca_kpca_comparison.csv`.
- Bảng NB/SVM giữa kỳ: `results/tables/classification/midterm_nb_svm_comparison.csv`.
- Bảng ba mô hình: `results/tables/classification/all_models_comparison.csv`.

Accuracy cần đọc cùng Precision, Recall, F1 và PR-AUC do lớp hưởng ứng chỉ khoảng
15%. PR-AUC hình thang và Average Precision (AP) được ghi riêng. Tham số chọn bằng
AP trong CV, không chọn lại từ validation chỉ vì F1 thấp hơn một cấu hình khác.

Logistic `class_weight=default` nghĩa là không cân bằng lớp, nhưng C và L1/L2
vẫn đã được dò bằng CV. Không coi đây là Logistic chưa hiệu chỉnh.

## Đầu ra xác suất và phần cuối kỳ

Mã xuất xác suất Logistic đã có, nhưng không cần chạy để nộp giữa kỳ:

```powershell
python -m src.classification --export-logistic-scores
```

Lệnh ghi xác suất ngoài fold cho train và xác suất validation từ mô hình fit
toàn train vào `data/processed/logistic_scores/`. Lệnh từ chối ghi đè nếu thư mục
đã tồn tại. Đây là dự đoán điểm của mô hình nguồn, không phải xác suất thực tế đã biết.

## Đóng gói

```powershell
python scripts/package_midterm.py
```

Gói nộp chứa mã nguồn, notebook, kết quả tổng hợp, tài liệu và hướng dẫn tải dữ liệu.
Không chứa `data/`, `.git/`, môi trường ảo, cache hay mô hình `.joblib`.
Xem file ZIP trong `dist/`; đọc duyệt nội dung và bổ sung thông tin lớp/MSSV trước khi nộp.
