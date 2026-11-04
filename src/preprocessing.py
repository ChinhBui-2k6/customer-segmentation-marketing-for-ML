"""Tiền xử lý và kỹ thuật đặc trưng (Feature Engineering) cho dữ liệu Marketing Campaign.

Module cung cấp các hàm và pipeline phục vụ:
1. Đọc dữ liệu thô (hỗ trợ file dạng tab-delimited).
2. Làm sạch dữ liệu: loại bỏ cột hằng số, định danh, xử lý outlier bất hợp lý.
3. Kỹ thuật tạo đặc trưng (Feature Engineering): Age, Customer_Tenure, Total_Spent, Total_Purchases, v.v.
4. Xây dựng scikit-learn Pipeline / ColumnTransformer:
   - Điền khuyết (Imputation)
   - Mã hóa biến phân loại (One-Hot Encoding)
   - Chuẩn hóa thang đo (StandardScaler / RobustScaler)
5. Lưu và nạp dữ liệu đã xử lý.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def load_raw(filename: str = "marketing_campaign.csv", sep: str = "\t") -> pd.DataFrame:
    """Nạp file dữ liệu thô từ thư mục data/raw.

    Args:
        filename: Tên file dữ liệu (mặc định: 'marketing_campaign.csv').
        sep: Ký tự phân cách (mặc định: '\\t').

    Returns:
        pd.DataFrame chứa dữ liệu gốc.
    """
    path = config.RAW_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    df = pd.read_csv(path, sep=sep)
    return df


def clean_raw_data(
    df: pd.DataFrame,
    drop_uninformative: bool = True,
    filter_outliers: bool = True,
) -> pd.DataFrame:
    """Làm sạch sơ bộ dữ liệu ban đầu.

    - Loại bỏ các cột không mang thông tin phân lớp/hồi quy: 'ID', 'Z_CostContact', 'Z_Revenue'.
    - Lọc bỏ các dòng ngoại lai bất hợp lý (Age > 100 hoặc Year_Birth < 1920, Income quá dị biệt).

    Args:
        df: DataFrame gốc.
        drop_uninformative: Loại bỏ các cột vô giá trị dự báo.
        filter_outliers: Lọc các bản ghi outlier rõ rệt.

    Returns:
        DataFrame đã được làm sạch cơ bản.
    """
    data = df.copy()

    # 1. Loại bỏ bản ghi hoàn toàn trùng lặp
    data = data.drop_duplicates().reset_index(drop=True)

    # 2. Loại bỏ cột định danh và cột hằng số
    if drop_uninformative:
        cols_to_drop = [col for col in ["ID", "Z_CostContact", "Z_Revenue"] if col in data.columns]
        data = data.drop(columns=cols_to_drop)

    # 3. Lọc bỏ các bản ghi dị thường về năm sinh và thu nhập
    if filter_outliers:
        if "Year_Birth" in data.columns:
            # Loại bỏ người sinh trước 1920 (tuổi > 100 vào thời điểm khảo sát)
            data = data[data["Year_Birth"] >= 1920]
        if "Income" in data.columns:
            # Lọc bỏ giá trị Income ngoại lai phi thực tế (> 200.000)
            data = data[(data["Income"].isna()) | (data["Income"] < 200000)]

    return data.reset_index(drop=True)


def engineer_features(
    df: pd.DataFrame,
    reference_year: int = 2014,
    drop_redundant: bool = True,
) -> pd.DataFrame:
    """Tạo thêm các đặc trưng kinh doanh hữu ích từ các biến có sẵn.

    Các đặc trưng mới:
    - Age: Tuổi của khách hàng (dựa trên reference_year).
    - Customer_Tenure_Days: Số ngày gắn bó tính đến thời điểm khảo sát.
    - Total_Spent: Tổng số tiền chi tiêu cho tất cả danh mục hàng hóa.
    - Total_Purchases: Tổng số lần mua sắm qua các kênh (Web, Catalog, Store, Deals).
    - Total_Accepted_Cmp: Tổng số lần chấp thuận các chiến dịch trước (Cmp1 -> Cmp5).
    - Children: Tổng số trẻ em trong nhà (Kidhome + Teenhome).
    - Is_Parent: Có con trong gia đình hay không (0 hoặc 1).
    - Family_Size: Kích thước hộ gia đình ước tính.

    Args:
        df: DataFrame đầu vào.
        reference_year: Năm mốc để tính tuổi và thời gian (dữ liệu khảo sát ~2012-2014).
        drop_redundant: Loại bỏ các cột gốc đã được suy biến (như Year_Birth, Dt_Customer).

    Returns:
        DataFrame kèm các đặc trưng mới.
    """
    data = df.copy()

    # Tuổi
    if "Year_Birth" in data.columns:
        data["Age"] = reference_year - data["Year_Birth"]

    # Thời gian khách hàng gắn bó (Customer Tenure tính bằng ngày)
    if "Dt_Customer" in data.columns:
        dt_series = pd.to_datetime(data["Dt_Customer"], format="%d-%m-%Y")
        max_date = dt_series.max()
        data["Customer_Tenure_Days"] = (max_date - dt_series).dt.days

    # Chuẩn hóa Marital_Status về các nhóm lớn
    if "Marital_Status" in data.columns:
        status_map = {
            "Married": "Partner",
            "Together": "Partner",
            "Single": "Single",
            "Divorced": "Single",
            "Widow": "Single",
            "Alone": "Single",
            "Absurd": "Single",
            "YOLO": "Single",
        }
        data["Marital_Group"] = data["Marital_Status"].map(status_map).fillna("Single")

    # Chuẩn hóa Education
    if "Education" in data.columns:
        edu_map = {
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate",
        }
        data["Education_Level"] = data["Education"].map(edu_map).fillna("Graduate")

    # Tổng chi tiêu
    spending_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    present_spent = [c for c in spending_cols if c in data.columns]
    if present_spent:
        data["Total_Spent"] = data[present_spent].sum(axis=1)

    # Tổng số lần mua
    purchase_cols = [
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]
    present_purchases = [c for c in purchase_cols if c in data.columns]
    if present_purchases:
        data["Total_Purchases"] = data[present_purchases].sum(axis=1)

    # Tổng chiến dịch đã phản hồi trước đó
    campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
    present_campaigns = [c for c in campaign_cols if c in data.columns]
    if present_campaigns:
        data["Total_Accepted_Cmp"] = data[present_campaigns].sum(axis=1)

    # Con cái và quy mô gia đình
    if "Kidhome" in data.columns and "Teenhome" in data.columns:
        data["Children"] = data["Kidhome"] + data["Teenhome"]
        data["Is_Parent"] = (data["Children"] > 0).astype(int)

        if "Marital_Group" in data.columns:
            data["Family_Size"] = (
                data["Marital_Group"].apply(lambda x: 2 if x == "Partner" else 1) + data["Children"]
            )

    # Loại bỏ các cột gốc đã được thay thế
    if drop_redundant:
        redundant_cols = [
            col for col in ["Year_Birth", "Dt_Customer", "Marital_Status", "Education"] if col in data.columns
        ]
        data = data.drop(columns=redundant_cols)

    return data


def split_feature_types(
    df: pd.DataFrame,
    target_col: Optional[str] = "Response",
) -> Tuple[List[str], List[str]]:
    """Tách danh sách cột thành 2 nhóm: numeric (số) và categorical (phân loại).

    Args:
        df: DataFrame cần kiểm tra.
        target_col: Tên cột mục tiêu (được bỏ qua không xếp vào features).

    Returns:
        (numeric_features, categorical_features)
    """
    cols = [c for c in df.columns if c != target_col]
    numeric_features = (
        df[cols].select_dtypes(include=["number", "int64", "float64"]).columns.tolist()
    )
    categorical_features = (
        df[cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    )
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: Optional[object] = None,
) -> ColumnTransformer:
    """Tạo ColumnTransformer hoàn chỉnh cho học máy.

    Pipeline cho biến số:
    1. SimpleImputer: Điền khuyết bằng median (phù hợp với các thuộc tính lệch như Income).
    2. Scaler: Chuẩn hóa dữ liệu (mặc định là StandardScaler).

    Pipeline cho biến phân loại:
    1. SimpleImputer: Điền khuyết bằng giá trị xuất hiện nhiều nhất.
    2. OneHotEncoder: Mã hóa One-hot (loại bỏ cột đầu tiên drop='first' hoặc giữ nguyên).

    Args:
        numeric_cols: Danh sách tên cột dạng số.
        categorical_cols: Danh sách tên cột dạng phân loại.
        scaler: Đối tượng scaler tùy chọn (mặc định: StandardScaler).

    Returns:
        ColumnTransformer sẵn sàng đưa vào pipeline.
    """
    if scaler is None:
        scaler = StandardScaler()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def save_processed(df: pd.DataFrame, filename: str = "cleaned_customers.csv") -> Path:
    """Lưu DataFrame đã xử lý vào thư mục data/processed.

    Args:
        df: DataFrame cần lưu.
        filename: Tên file kết quả.

    Returns:
        Đường dẫn file đã lưu.
    """
    config.ensure_dirs()
    path = config.PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    return path
