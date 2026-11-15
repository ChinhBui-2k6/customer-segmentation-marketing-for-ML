# Kiểm tra nội dung giữa kỳ

Phạm vi theo cuối `Projects_ideas/Projects_Ideas.pdf`: có tài liệu yêu cầu, hoàn
thành tiền xử lý, giảm chiều/trực quan và ít nhất hai mô hình phân loại.
Nhóm chọn trình bày NB và SVM; giữ Logistic làm kết quả bổ sung. Hồi quy để cuối kỳ.

## Nội dung hiện có

| Mục | Nội dung dùng giữa kỳ | Nơi kiểm tra |
|---|---|---|
| 2.1 | Làm sạch, thống kê, chia nhóm, train-only pipeline | Notebook 01 |
| 2.2 | PCA, Kernel PCA RBF, thống kê/hình và so sánh cùng mẫu | Notebook 02 |
| 2.3 | K-Means, chọn K, hồ sơ cụm, Response theo cụm | Notebook 03 |
| 2.4(a–b) | NB/SVM, 3 tỷ lệ, gốc/PCA, regularization, nhận xét | Notebook 04 phần A |
| Bổ sung | Logistic, cùng tỷ lệ/biểu diễn, đối chiếu ba mô hình | Notebook 04 phần B/C |
| Tái lập | Cài đặt, dữ liệu, lệnh chạy | README, DATA_GUIDE, requirements |
| Trình bày | Bản thảo báo cáo và kịch bản slides | report/, slides/ |

## Thay đổi sau khi nhận phần Tuananh

- Giữ nguyên các hàm Kernel PCA và Logistic đã bàn giao.
- Kernel PCA fit train 80:20 thay cho toàn development; dùng chính preprocessing
  của PCA train 80:20. Validation chỉ transform. Gamma cố định 1/37.
- So sánh cùng 6 thành phần; phần dự đoán vẫn dùng PCA 12 chiều.
- Cập nhật skewness của bảng Kernel PCA theo cùng cách tính mẫu như PCA.
- Bổ sung trustworthiness@10 với cùng không gian tham chiếu và bảng so sánh.
- Khôi phục nhận xét Logistic và sửa dòng tiếng Việt lỗi mã hóa sau lần ghép.
- Bảng ba mô hình phân biệt Logistic đã tuning trong từng chế độ trọng số với
  NB/SVM tuned bao gồm cả lựa chọn trọng số bằng CV.

## Cần đọc duyệt trước khi nộp

Kiểm chứng kỹ thuật ngày 15/11/2026: notebook 02 chạy đủ 14 ô code và notebook
04 chạy đủ 11 ô code, không có lỗi. Bộ unittest có 31 kiểm tra đạt. Đã kiểm tra
trực quan hình so sánh PCA–Kernel PCA, pairplot Kernel PCA và confusion matrix
Logistic. Test và các file splits không đổi hash/thời gian sửa.

- [ ] Cả hai đọc và giải thích được phần mình, đặc biệt Kernel PCA và SVM RBF.
- [ ] Điền họ tên chính thức, MSSV, lớp và tên giảng viên nếu trường yêu cầu.
- [ ] Chốt thời lượng thuyết trình và chia người nói từng slide.
- [ ] Định dạng bản thảo báo cáo theo yêu cầu 15–20 trang A4, Times New Roman 13,
      căn đều, giãn dòng 1.3–1.5; bản Markdown chưa xác định được số trang.
- [ ] Chuyển kịch bản slides sang định dạng trình chiếu của nhóm và duyệt hình/chữ.
- [ ] Xác nhận phạm vi phương pháp Kernel PCA với nội dung đã học/giảng viên nếu cần.
- [ ] Kiểm tra file ZIP chỉ có mã, kết quả tổng hợp và tài liệu; không có dữ liệu.

Không đánh dấu hoàn thành phần đọc duyệt thay thành viên. Audit chạy notebook
nằm ở `results/tables/notebook_execution_audit.json`, là bằng chứng kỹ thuật,
không thay việc thành viên đọc và hiểu kết quả.

## Việc cuối kỳ

Trình bày đầy đủ Logistic và so sánh ba nhóm, mục 2.4(c) với hai mô hình hồi quy,
hoàn thiện báo cáo cuối. GMM/DBSCAN là phần mở rộng đã phân công, không bắt buộc
để đủ mức tối thiểu một mô hình phân cụm của đề. Test cuối vẫn đóng băng.
