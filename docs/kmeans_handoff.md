# Bàn giao K-Means — ChinhBui

Cập nhật 06/11/2026. Phần A của [notebook 03](../notebooks/03_clustering.ipynb) thực hiện K-Means cho mục 2.3. Phần GMM/DBSCAN và so sánh chung chờ Tuananh.

## Phương pháp

Dùng các tập train/validation đã chốt trong development. Mỗi tỷ lệ 80:20, 70:30, 60:40 có hai bản: không PCA (37 chiều sau impute/encode/scale), PCA (12 chiều). Fit mọi bước trên train; gán validation về tâm đã học. Response và metadata không đi vào không gian cụm. Nhóm trùng không giao giữa train, validation và test.

Thử K=2…10, `n_init=10`, seed 42. Chọn K theo Silhouette train lớn nhất, hòa thì K nhỏ hơn. Heuristic Elbow là vị trí cách xa nhất phía dưới đường nối hai đầu inertia đã chuẩn hóa; nó chỉ hỗ trợ đọc đường cong. Không chọn K theo Response hoặc điểm validation. Không đánh giá hay gán cụm cho test.

## Kết quả thực nghiệm

| Tỷ lệ | Không gian | K chọn | Elbow gợi ý | Silhouette train | Silhouette validation |
|---|---|---:|---:|---:|---:|
| 80:20 | 37 chiều | 2 | 5 | 0,266 | 0,275 |
| 80:20 | PCA 12 chiều | 2 | 5 | 0,312 | 0,323 |
| 70:30 | 37 chiều | 2 | 5 | 0,267 | 0,270 |
| 70:30 | PCA 12 chiều | 2 | 5 | 0,313 | 0,320 |
| 60:40 | 37 chiều | 2 | 5 | 0,272 | 0,260 |
| 60:40 | PCA 12 chiều | 2 | 5 | 0,319 | 0,309 |

Bảng đủ còn có Davies–Bouldin, Calinski–Harabasz, bình phương khoảng cách trung bình đến tâm và số cụm thực sự có mặt. Khi validation chỉ có một cụm, các chỉ số cần ít nhất hai cụm được báo NaN.

## Nhận xét từ dữ liệu thực

- **Chọn K=2 trong cả sáu kịch bản theo Silhouette.** Elbow gợi ý K=5 do inertia giảm đều với mức giảm dần, không có góc gãy duy nhất thật rõ. Ở train 80:20 không PCA, K=2 có Silhouette 0,266, K=4 là 0,203 và K=5 là 0,135. Quy tắc Silhouette ưu tiên hai nhóm lớn; K=5 có thể phân chia chi tiết hơn nhưng độ tách/độ chặt trung bình kém hơn theo tiêu chí này. K=2 không phải khẳng định khách hàng chỉ có hai nhóm kinh doanh đúng duy nhất.
- **Độ tách cụm còn hạn chế.** Silhouette khoảng 0,26–0,32; hình chiếu vẫn có vùng chồng lấn. Không gọi kết quả là phân nhóm hoàn hảo. Validation không giảm mạnh so với train trong các kịch bản này; đây là kiểm tra độ nhất quán nội tại, không thay đánh giá dự đoán Response.
- **PCA có Silhouette cao hơn trong không gian đã giảm chiều**, nhưng hai không gian có hình học khác nhau. Ở kịch bản 80:20, hai bản cho cùng phân hoạch train sau khi hoán vị ID cụm (ARI=1); validation gần nhau nhưng không hoàn toàn giống (ARI≈0,978). ARI giữa hai biểu diễn trên cả sáu phân vùng nằm khoảng 0,978–1. Vì vậy, chỉ số Silhouette cao hơn chưa chứng minh PCA tạo phân nhóm có ích hơn hoặc dự đoán Response tốt hơn. Bảng ARI giữa hai biểu diễn được xuất riêng.
- **Ổn định khi đổi khởi tạo:** chạy seed 7/21/42/84/123 với cùng train, `n_init=10`. ARI so với seed 42 tối thiểu khoảng 0,994 cho bản 37 chiều 80:20 và bằng 1 trong các cấu hình còn lại. Đây là kiểm tra khởi tạo trên cùng dữ liệu, chưa phải bootstrap hay độ ổn định qua các mẫu khách hàng độc lập.
- **Hồ sơ train 80:20, bản 37 chiều:** cụm 0 có 882 khách hàng (61,68%), Income median 38.774,5, Total_Spent median 97,5, Total_Purchases median 6; cụm 1 có 548 khách hàng (38,32%), Income median 71.613, Total_Spent median 1.170,5, Total_Purchases median 20. Có thể mô tả lần lượt là nhóm chi tiêu/mua sắm thấp hơn và nhóm chi tiêu/mua sắm cao hơn. Đây là thống kê tương đối trong bộ dữ liệu, không gán đơn vị tiền tệ chưa xác minh.
- **Response sau phân cụm:** cụm 0 có 85/882 = 9,64%; cụm 1 có 128/548 = 23,36%, so với 14,90% toàn train. Lift của nhóm chi tiêu cao là 1,57. Trên validation, hai nhóm tương ứng có tỷ lệ 8,93% và 25,37% (lift nhóm cao 1,68). Cần ghi số mẫu vì validation chỉ có 54 khách hưởng ứng. Những tỷ lệ này chưa là xác suất cá nhân đã hiệu chuẩn và chưa chứng minh hiệu quả của một chiến dịch mới.
- **ID cụm chỉ là số thứ tự.** Ở 80:20, cụm 0 của bản PCA tương ứng cụm 1 của bản 37 chiều và ngược lại; không so các ID trực tiếp. Biểu đồ dùng màu theo ID của từng mô hình, ký hiệu tròn/vuông theo Response.
- **Giới hạn khoảng cách:** biến tổng có tương quan với biến thành phần, số được chuẩn hóa và one-hot giữ 0/1 nên các nhóm đặc trưng không đóng góp ngang nhau. PCA không loại toàn bộ hạn chế đó; K-Means còn giả định các cụm tương đối tròn trong khoảng cách Euclid.

## File và chạy lại

- `src/clustering.py`: `analyze_kmeans`, `cluster_metrics`, `describe_clusters`, `initialization_stability`. Chạy trực tiếp file chỉ in hướng dẫn mở notebook; import hàm để dùng lại.
- `notebooks/03_clustering.ipynb`: chạy từ đầu đến cuối trong Jupyter; mở được từ gốc project hoặc `notebooks/` sau khi notebook 01 đã tạo splits.
- `results/tables/kmeans_selection_summary.csv`, `kmeans_evaluation.csv`: tóm tắt chọn K và điểm train/validation; các file `kmeans_<tỷ_lệ>_<không_gian>_k_selection.csv` có mọi K đã thử.
- Các bảng `kmeans_<thí_nghiệm>_<phân_vùng>_*`: số mẫu/tỷ lệ Response/lift, mean/median theo đơn vị gốc, tỷ lệ category và số thiếu của từng cụm. `kmeans_cluster_profiles.csv` tổng hợp kích thước và Response.
- `kmeans_initialization_stability.csv` và `kmeans_representation_agreement.csv`: ARI khi đổi seed và khi đổi cách biểu diễn.
- `results/figures/clustering/kmeans_80_20_*.png`: Elbow/Silhouette, Silhouette từng cụm, hồ sơ khách hàng, tỷ lệ hưởng ứng, hình màu theo cụm/ký hiệu theo Response.
- `data/processed/clustering/<tỷ_lệ>_<không_gian>/`: pipeline `.joblib` và metadata/nhãn từng khách hàng (bị Git bỏ qua). Model nhận X sạch không có Response; `.predict` dùng tiền xử lý và tâm đã fit. Không đưa nhãn từng khách hàng vào Git hoặc gói nộp.
- `results/tables/kmeans_audit.json`: xác minh test và mọi file splits giữ nguyên byte và mtime.

## Kiểm tra và bước tiếp theo

Bộ 24 kiểm tra tự động đạt, gồm 4 kiểm tra mới: chọn đúng hai cụm trên dữ liệu tách rõ; thay validation cực lớn không đổi fit/chọn K; hồ sơ khớp theo vị trí dù index khác nhau và từ chối Response trong X; xử lý K sai/chỉ số không xác định. Pipeline lưu/đọc cho nhãn validation giống nhau. Chạy kiểm tra bằng `python -B -m unittest discover -s tests -v`.

Tuananh bổ sung GMM/DBSCAN tại phần B, dùng cùng tập chia và biểu diễn để so sánh ở phần C. Nếu đưa cụm vào mô hình phân loại, fit lại toàn bộ preprocessing/PCA/K-Means trong mỗi fold train, tránh dùng nhãn cụm từ toàn development trước khi CV. ChinhBui có thể tiếp tục Naive Bayes và SVM.
