# Bàn giao PCA — ChinhBui

Cập nhật 05/11/2026. Phần PCA trong [notebook 02](../notebooks/02_dimensionality_reduction.ipynb) đã chạy toàn bộ 7 cell code không lỗi. Cả bộ 20 kiểm tra tự động đạt, gồm 3 kiểm tra mới về PCA. Phần Kernel PCA và so sánh hai phương pháp vẫn chờ Tuananh.

## Kết quả

| Train:validation | Train | Validation | Đầu vào | Giữ lại | Phương sai giữ lại | PC đạt 90% | PC đạt 95% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 80:20 | 1.430 | 358 | 37 | 12 | 80,17% | 18 | 21 |
| 70:30 | 1.250 | 538 | 37 | 12 | 80,20% | 18 | 21 |
| 60:40 | 1.072 | 716 | 37 | 12 | 80,49% | 18 | 21 |

12 được chọn theo quy tắc `max(1, floor(d/3))`, không phải số chiều đủ giữ 90% phương sai. Phương sai được đo trong không gian số chuẩn hóa + one-hot 0/1; không phải phương sai của dữ liệu thô. Không thay đổi k dựa trên nhãn validation.

## Nhận xét từ dữ liệu thực

Các diễn giải dưới đây dùng train 80:20 và chiều trục của lần fit này:

- **PC1 giải thích 31,14% phương sai**, có trọng số lớn ở Total_Spent (0,306), Total_Purchases (0,276), Income (0,266), MntMeatProducts (0,260), NumCatalogPurchases (0,257) và MntWines (0,254). Có thể diễn giải trục này chủ yếu liên quan quy mô chi tiêu, mua sắm và thu nhập.
- **PC2 giải thích 10,47%**, gắn với Teenhome, NumDealsPurchases, Children, Family_Size và Is_Parent. Khác biệt mean Response trên trục này nhỏ: khoảng 0,030 độ lệch chuẩn train, không đủ để xem là trục phân biệt mạnh chỉ dựa vào mean.
- **PC3 giải thích 8,11%**, nổi bật Total_Accepted_Cmp và AcceptedCmp1–5, gợi ý trục lịch sử hưởng ứng. Mean lớp 1 cao hơn lớp 0 khoảng 0,989 độ lệch chuẩn trên train; validation là 0,845 độ lệch chuẩn train. Đây là quan hệ mô tả, không chứng minh nhân quả hay chất lượng phân loại.
- **6 PC đầu giữ 62,93% phương sai**. Chỉ vẽ hai PC đầu sẽ bỏ qua nhiều cấu trúc, nên notebook hiển thị đủ 15 cặp trong 6 PC đầu.
- PC1 và PC4 cũng có chênh lệch mean theo Response trên train, lần lượt khoảng 0,725 và 0,699 độ lệch chuẩn. Các boxplot và scatter vẫn cho thấy chồng lấn giữa hai lớp; không có bằng chứng tách lớp hoàn hảo.
- **PC6 có quan hệ không ổn định giữa train và validation**: chênh lệch mean chuẩn hóa chuyển từ +0,374 sang −0,114 trong cùng hệ trục đã fit trên train. Không nên khái quát quan hệ từ một biểu đồ train, cũng không tự chọn/bỏ thành phần theo quan sát validation này.
- Các PC không tương quan trên train nhưng vẫn có thể lệch, phụ thuộc phi tuyến hoặc không Gaussian. PCA không tự giải quyết mọi giả định của Gaussian NB.
- Trọng số trên biến tổng và các biến thành phần phản ánh dữ liệu có dư thừa; chúng không phải các bằng chứng độc lập. Dấu PCA có thể đảo mà không đổi không gian. Không so trực tiếp dấu của PC từ các tập train khác nhau.

## File và cách dùng

- `src/dim_reduction.py`: `analyze_pca(X_train, X_validation)` fit preprocessing và PCA đầy đủ trên train, trả scores, mean/std/skewness, explained variance, hệ số và bảng tổng hợp. `component_statistics` dùng std mẫu (`ddof=1`).
- `src/experiment_preprocessing.py`: `make_feature_pipeline(X_train, reduce=True)` là pipeline 12 PC dành cho mô hình/CV. Đã kiểm tra kết quả khớp 12 cột đầu của phân tích PCA đầy đủ. Phải thêm mô hình vào pipeline và fit lại toàn bộ từng fold khi tuning theo nhóm.
- `results/tables/pca_<tỷ_lệ>_{variance,train_statistics,validation_statistics,weights}.csv`: bảng đầy đủ cho cả ba tỷ lệ; `pca_experiment_summary.csv` tổng hợp; `pca_response_relationship.csv` liên hệ với nhãn; `pca_80_20_top_weights.csv` trọng số nổi bật.
- `results/figures/dim_reduction/pca_80_20_*.png`: scree/phương sai tích lũy, ma trận cặp 6 PC, boxplot theo nhãn, heatmap hệ số.
- `results/tables/pca_audit.json`: ghi nhận fit trên train, không transform test, toàn bộ splits giữ nguyên nội dung và mtime.

Không tạo bản dữ liệu mới cạnh các bản trong `data/processed/experiments/` của notebook 01. Tập test 448 dòng vẫn giữ nguyên. Chạy lại notebook từ thư mục gốc hoặc `notebooks/`, sau khi notebook 01 đã tạo splits. Kiểm tra bằng `python -B -m unittest discover -s tests -v`.

## Tuananh tiếp tục

Thêm Kernel PCA RBF tại phần B của notebook 02 và hàm vào `src/dim_reduction.py`. Giữ cùng các tập chia, chỉ fit train và chọn gamma trong train. Báo cáo phương sai trong không gian kernel, không so tỷ lệ đó như cùng đơn vị với phương sai PCA. Cả hai hoàn thiện phần C sau khi có kết quả; đánh giá lợi ích cho phân loại ở notebook 04.
