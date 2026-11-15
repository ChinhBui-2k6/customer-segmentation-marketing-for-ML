# Bàn giao Naive Bayes và SVM — ChinhBui

Cập nhật 07/11/2026. Phần A của [notebook 04](../notebooks/04_classification.ipynb) triển khai mục 2.4(a–b) cho Gaussian NB và SVM RBF. Logistic của Tuananh còn cần bổ sung để đủ ba nhóm phân loại; phần hồi quy 2.4(c) tiếp tục riêng ở notebook 05.

## Thực nghiệm và lựa chọn tham số

12 cấu hình = 2 mô hình × 2 biểu diễn × 3 tỷ lệ trong development. Mỗi cấu hình chạy 3 phiên bản: default, balanced, tuned. Tổng cộng 36 pipeline cuối và 72 dòng độ đo train/validation. Bản encoded đã điền thiếu/mã hóa/chuẩn hóa, chưa giảm chiều (hiện 37 chiều); bản PCA giữ `max(1, floor(d/3))` (hiện 12 chiều).

Mỗi cấu hình dùng train/validation cố định đã chốt. GridSearchCV chỉ nhận train và các fold StratifiedGroupKFold 5-fold (seed 42), không giao nhóm trùng. Median, category, scaler, PCA và trọng số cân bằng đều được học lại trong mỗi fold train; sau chọn tham số, refit toàn train. Không dùng validation để chọn tham số, trọng số hay ngưỡng. Test 448 dòng không fit, transform, dự đoán hoặc dùng để chọn cấu hình.

| Mô hình | Default | Balanced | Grid tìm trong train |
|---|---|---|---|
| Gaussian NB | var_smoothing=1e-9 | Trọng số mẫu nghịch tần suất lớp | var_smoothing: 1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1; balanced False/True |
| SVM RBF | C=1, gamma=scale | class_weight=balanced | C: 0.1,1,10,100; gamma: scale,0.01,0.1; class_weight None/balanced |

NB có 12 ứng viên × 5 fold; SVM có 24 ứng viên × 5 fold trong mỗi cặp tỷ lệ/biểu diễn: tổng 1.080 lượt fit CV, chưa tính refit và baseline. Chọn mean Average Precision cao nhất, vì quan tâm xếp hạng lớp hưởng ứng hiếm. Bảng báo cáo AP riêng với PR-AUC hình thang; không đồng nhất hai khái niệm.

Nếu nhiều ứng viên có mean AP bằng nhau, GridSearchCV lấy ứng viên đầu theo thứ tự grid, không dùng F1 làm tiêu chí phụ. Đặc biệt NB có thể cho cùng thứ tự xếp hạng khi thay prior/trọng số, nhưng nhãn cứng ở ngưỡng 0,5 lại khác; vì vậy các ứng viên đồng AP vẫn có thể khác F1/Recall.

Default và balanced giữ cùng tham số để đối chiếu ảnh hưởng trọng số. Bản tuned tìm đồng thời tham số/trọng số nên thay đổi so với default có thể do cả hai. Các bảng grid cho phép xem riêng tác động var_smoothing hoặc C khi giữ cố định các tham số khác. Không tuyên bố regularization luôn cải thiện mọi độ đo.

## Quy ước đánh giá và diễn giải

- `Response=1` là lớp dương. Báo cáo Accuracy, Precision, Recall, F1, PR-AUC hình thang, AP, ROC-AUC và TN/FP/FN/TP trên cả train và validation. Dữ liệu khoảng 15% lớp 1; Accuracy phải đối chiếu baseline dự đoán toàn 0 và độ đo lớp 1.
- NB dùng posterior với quyết định theo lớp lớn nhất (0,5; trường hợp hòa theo sklearn). SVM dùng ngưỡng decision margin 0; không bật tính xác suất hoặc calibration nội bộ. Margin không phải xác suất.
- PR-AUC: `auc(recall, precision)` từ `precision_recall_curve`; AP: `average_precision_score`. Các chỉ số xếp hạng dùng score liên tục, không dùng nhãn cứng. Các độ đo dựa trên nhãn vẫn ở ngưỡng mặc định đã chốt.
- `train_minus_validation_*` là chênh lệch trên phân vùng ngoài; `cv_train_minus_validation_ap` là mean train AP − mean held-out fold AP của cấu hình được chọn. Đây là bằng chứng để thảo luận overfit, không có ngưỡng kết luận cứng. Validation giữa ba tỷ lệ lồng nhau, không phải ba thực nghiệm độc lập.
- C nhỏ hơn tăng regularization của SVM; gamma nhỏ thường cho ranh giới trơn hơn. Var_smoothing ổn định phương sai NB, không phải L1/L2. Trọng số lớp giải quyết mất cân bằng chi phí học, không phải regularization trực tiếp.
- FP là chi phí liên hệ khách không hưởng ứng; FN là bỏ sót khách thực tế hưởng ứng. Chưa có ngân sách/chi phí cụ thể để chọn ngưỡng tối ưu. Chưa dò ngưỡng từ validation và chưa đánh giá test.

## Nhận xét mô hình từ kịch bản 80:20

Validation có 358 khách hàng, gồm 54 hưởng ứng. Dự đoán toàn 0 đạt Accuracy 84,92% nhưng Recall/F1 lớp 1 bằng 0. Các số dưới đây là validation của cấu hình tuned bằng CV trên train, không phải test.

| Mô hình | Biểu diễn | Precision | Recall | F1 | AP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| Gaussian NB | Không PCA | 0,486 | 0,630 | 0,548 | 0,558 | 36 | 20 |
| Gaussian NB | PCA | 0,923 | 0,222 | 0,358 | 0,563 | 1 | 42 |
| SVM RBF | Không PCA | 0,611 | 0,407 | 0,489 | 0,579 | 14 | 32 |
| SVM RBF | PCA | 0,621 | 0,333 | 0,434 | 0,576 | 11 | 36 |

- **Gaussian NB:** bản không PCA tuned tăng Recall từ 0,426 lên 0,630 và F1 từ 0,511 lên 0,548 so default, nhưng FP tăng 13→36 và AP giảm 0,576→0,558. Không gọi tuning là cải thiện mọi độ đo. Bản PCA tuned có Precision cao nhưng bỏ sót 42/54 người hưởng ứng: không phù hợp nếu mục tiêu là tìm được nhiều khách hưởng ứng ở ngưỡng 0,5 này. AP vẫn khoảng 0,563 cho thấy chất lượng xếp hạng và chất lượng nhãn cứng là hai vấn đề khác nhau.
- **NB và overfit:** AP train của tuned không PCA/PCA lần lượt khoảng 0,472/0,535, validation khoảng 0,558/0,563. Không có dấu hiệu overfit rõ theo khoảng cách AP trên cặp chia này; điều đó không chứng minh mô hình hoàn hảo hoặc giả định đúng. Điểm NB vẫn hạn chế và Recall nhạy với prior/smoothing. Phù hợp làm baseline sinh, nhanh và có tín hiệu xếp hạng; chưa đủ cơ sở chọn làm mô hình cuối hoặc xem xác suất là đã hiệu chuẩn.
- **SVM cân bằng lớp:** bản không PCA tăng Recall 0,278→0,722, F1 0,370→0,574, nhưng FP 12→43 và Precision 0,556→0,476 so default. Bản PCA balanced có Recall 0,778, FP=51, FN=12, F1=0,571. Đây là lựa chọn đánh đổi nếu quan tâm Recall; chưa có chi phí FP/FN cụ thể để kết luận dùng cấu hình nào cuối cùng.
- **SVM tuned và overfit:** không PCA có AP train/validation 0,893/0,579 (gap≈0,315), F1 0,759/0,489 (gap≈0,270). PCA có AP 0,767/0,576 (gap≈0,191), F1 0,618/0,434 (gap≈0,184). Vẫn có dấu hiệu overfit sau tuning; PCA giảm gap trong kịch bản này nhưng AP validation gần như tương đương. Cả hai chọn C=10, gamma=0,01, class_weight=None theo CV-AP, không theo F1 validation.
- **Kết luận riêng SVM:** phù hợp là ứng viên phân biệt phi tuyến để so sánh, có chất lượng xếp hạng đáng chú ý so với tỷ lệ lớp dương khoảng 0,151. Tuy nhiên, gap train–validation và số FN còn cao ở cấu hình tuned. Chưa thể kết luận là mô hình cuối tốt nhất khi Logistic chưa có và test chưa được đánh giá. Cân bằng lớp cải thiện Recall nhưng không khắc phục hết overfit.

### Kiểm chứng tác động regularization trong CV, giữ tham số khác cố định

Từ grid SVM 80:20 **không PCA**, cố định gamma=scale và class_weight=None:

| C | Mean AP fold train | Mean AP fold validation | Gap |
|---:|---:|---:|---:|
| 0,1 | 0,829 | 0,624 | 0,205 |
| 1 | 0,885 | 0,660 | 0,225 |
| 10 | 0,969 | 0,634 | 0,335 |
| 100 | 0,980 | 0,566 | 0,414 |

Giảm C từ 100 xuống 1 làm AP train giảm nhưng AP held-out fold tăng 0,566→0,660, gap giảm 0,414→0,225: bằng chứng trong train cho tác dụng regularization. C=0,1 có gap nhỏ hơn nhưng AP held-out thấp hơn C=1, thể hiện regularization quá mạnh cũng có thể giảm chất lượng. Khi cho gamma thay đổi trong grid, cấu hình C=10/gamma=0,01 được chọn với CV-AP≈0,683, gap CV≈0,218. Không chỉ xét riêng C rồi khẳng định mô hình phức tạp hơn hay kém hơn khi gamma đồng thời thay đổi.

Với NB không PCA, var_smoothing=0,1 cùng balanced=True được chọn; CV-AP≈0,475, chỉ tăng nhẹ so với các smoothing nhỏ. Với PCA, var_smoothing=0,1 có các ứng viên balanced/unbalanced đồng CV-AP≈0,539; thứ tự grid chọn unbalanced. Không dùng kết quả F1 validation để đổi lại cấu hình sau khi xem. Các bảng đầy đủ của 70:30 và 60:40 cần được đọc cùng kết quả 80:20; không suy rộng mọi con số trên sang cả ba tỷ lệ.

### So sánh đủ ba tỷ lệ và hai biểu diễn

Các ô là **F1 / AP trên validation của tuned**:

| Tỷ lệ | NB không PCA | NB PCA | SVM không PCA | SVM PCA |
|---|---|---|---|---|
| 80:20 | 0,548 / 0,558 | 0,358 / 0,563 | 0,489 / 0,579 | 0,434 / 0,576 |
| 70:30 | 0,544 / 0,556 | 0,429 / 0,559 | 0,589 / 0,543 | 0,500 / 0,635 |
| 60:40 | 0,521 / 0,522 | 0,378 / 0,544 | 0,565 / 0,626 | 0,489 / 0,591 |

- NB PCA có AP nhỉnh hơn không PCA ở cả ba tỷ lệ, nhưng F1 thấp hơn do Recall thấp ở ngưỡng posterior hiện tại; không nên kết luận “giảm chiều tốt hơn” chỉ theo một chỉ số. Không PCA tuned có Recall khoảng 0,605–0,692, còn PCA khoảng 0,222–0,296.
- SVM PCA cải thiện AP rõ ở 70:30 (0,543→0,635), nhưng không cải thiện ở 80:20 hoặc 60:40; F1 của nhánh PCA thấp hơn trong cả ba tỷ lệ. Mỗi nhánh tự chọn tham số/trọng số theo CV, nên đây là so sánh toàn pipeline, không quy mọi chênh lệch chỉ cho phép PCA.
- SVM vẫn có gap mean train–held-out fold AP khoảng 0,125–0,218 trong cả sáu cấu hình tuned. NB có gap khoảng −0,006 đến 0,011: chưa thấy dấu hiệu overfit rõ theo AP-CV, nhưng chất lượng mô hình/giả định vẫn cần xem xét. Các chi tiết và gap ngoài nằm trong bảng CSV.
- Tăng tỷ lệ train không cho cải thiện đơn điệu trong các kết quả này. Các tập validation khác nhau về số mẫu và lồng nhau nên không dùng một hàng tốt nhất để khẳng định tỷ lệ chia tối ưu phổ quát. Chưa chọn mô hình/ngưỡng cuối hoặc đánh giá test.

## Giới hạn Gaussian NB và hướng kiểm chứng

One-hot 0/1 và các biến nhị phân không Gaussian; các mức cùng category phụ thuộc nhau. Biến tổng có tương quan với biến thành phần. PCA tạo đầu vào liên tục và không tương quan toàn train nhưng không bảo đảm Gaussian/độc lập có điều kiện trong mỗi lớp. Smoothing không sửa các giả định này, và điểm xác suất có thể quá tự tin, chưa được hiệu chuẩn.

Hướng kiểm chứng tùy chọn chưa triển khai: NB hỗn hợp Gaussian cho số liên tục, Bernoulli cho cột nhị phân ở bản chưa PCA; cộng log-likelihood và prior đúng một lần. Không đưa PC vào Bernoulli NB. Giữ Gaussian NB là mô hình nhóm 1 theo phân công.

## File và chạy lại

- `src/classification.py`: factory `make_classifier`, `FoldBalancedGaussianNB`, `grouped_training_cv`, `tune_classifier` và `evaluate_classifier`. Trọng số NB tính trong `fit`, không tạo một vector theo toàn train rồi dùng cho mọi fold.
- `src/evaluation.py`: `positive_scores(estimator, X)` và `classification_metrics(y_true, y_pred, scores)` dùng chung được cho Logistic. Thiếu một lớp ở dữ liệu đánh giá thì AUC/AP báo NaN; Precision/Recall/F1 không xác định dùng zero_division=0. Hàm hồi quy chưa được triển khai trong file này.
- `results/tables/classification/`: `classification_metrics.csv`, `classification_cv_selection.csv`, `classification_cv_folds.csv`, `classification_train_validation_gaps.csv`, `classification_audit.json` và bảng đầy đủ mọi ứng viên/fold của từng cấu hình.
- `results/figures/classification/nb_svm_*`: so sánh default/balanced/tuned, confusion matrix tuned, ROC/PR và nhãn dự đoán so với thực tế.
- `data/processed/classification/<tỷ_lệ>_<biểu_diễn>_<mô_hình>/`: ba pipeline joblib và dự đoán train/validation kèm metadata/thực tế/điểm. Các file dữ liệu/model được Git bỏ qua. Pipeline nhận X sạch không có nhãn, tự transform khi dự đoán.
- Dùng Jupyter mở notebook 04, Restart Kernel and Run All. Notebook chạy từ gốc hoặc `notebooks/`; cần splits từ notebook 01. Run All chạy lại toàn bộ grid, có thể mất vài phút. Không mở `src/classification.py` để mong chạy toàn bộ phân tích: chạy file chỉ in hướng dẫn notebook.

## Kiểm tra và phần chờ Tuananh

Bộ 29 kiểm tra tự động đạt: 24 kiểm tra trước và 5 kiểm tra mới về PR-AUC/AP, trọng số NB theo từng fit/clone, phân hoạch CV không giao nhóm, imputer/PCA train-only và margin SVM. Chạy bằng `python -B -m unittest discover -s tests -v`.

Notebook đã chạy toàn bộ 6 cell code không lỗi; xuất đủ 72 dòng độ đo, 12 bảng grid, 11 hình và 36 pipeline cuối. Đã đối chiếu pipeline lưu/đọc với dự đoán validation, kiểm tra hình confusion matrix/ROC–PR/dự đoán–thực tế và xác nhận mọi file splits giữ nguyên byte/mtime. Các đầu ra không khẳng định SVM đã hết overfit.

Đã có giao diện đánh giá phân loại cho Tuananh dùng; hai thành viên chưa xác nhận đọc duyệt/chốt giao diện chung. Tuananh bổ sung Logistic vào phần B cùng điều kiện; ChinhBui ghép phần C so sánh ba nhóm. Xác suất Logistic ngoài fold là đầu vào phần hồi quy, không thay bằng margin SVM hay xác suất NB trong lần bàn giao này.
