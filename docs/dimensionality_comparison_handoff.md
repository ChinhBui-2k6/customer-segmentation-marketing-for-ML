# Bàn giao so sánh PCA và Kernel PCA

Đã giữ hàm Kernel PCA từ Tuananh và thay phạm vi fit trong notebook 02 từ
development sang **train 80:20**. Lý do là phần PCA đang báo cáo theo từng train;
hai phương pháp cần cùng mẫu, cùng preprocessing và cùng số chiều khi đối chiếu.

- 1.430 mẫu train và 358 mẫu validation; preprocessing của PCA train được dùng
  cho cả hai phương pháp. Không fit lại preprocessing bằng validation.
- Cùng 6 thành phần. Kernel RBF dùng gamma=1/37, không dò theo nhãn.
- PCA 12 chiều của phân loại vẫn giữ nguyên, không thay bằng Kernel PCA.
- Skewness bảng Kernel PCA dùng cách tính mẫu của pandas, thống nhất với PCA.
- Phương sai kernel chia cho trace toàn bộ kernel đã định tâm.

| Độ đo | PCA | Kernel PCA RBF |
|---|---:|---:|
| Trustworthiness@10 train | 0,9765 | 0,9479 |
| Trustworthiness@10 validation | 0,9667 | 0,9295 |
| Phương sai train giữ lại với 6 thành phần | 62,93% đầu vào đã xử lý | 39,21% kernel |

Trustworthiness tham chiếu cùng ma trận 37 chiều. PCA giữ lân cận tốt hơn trong
cấu hình này; không kết luận tốt hơn về dự đoán Response hoặc về mọi gamma.
Hai tỷ lệ phương sai không cùng thước đo để xếp hạng trực tiếp.

Các kết quả `kpca_*` mới thay cho kết quả cũ trên toàn development. Vì phạm vi
fit thay đổi, không trộn số liệu KPCA cũ 1.788 mẫu với bảng so sánh mới 1.430 mẫu.
Nhận xét trên GitHub ban đầu vẫn xem được trong lịch sử commit.

Đầu ra: `pca_kpca_comparison.csv`, `pca_kpca_audit.json`,
`kpca_component_statistics.csv`, `kpca_validation_statistics.csv` và
`pca_kpca_same_train.png`. Notebook 02 chạy lại thành công, audit ghi hash file
và xác nhận tất cả file splits giữ nguyên nội dung lẫn thời gian sửa.
