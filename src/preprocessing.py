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
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

REFERENCE_DATE = "2014-06-30"
MIN_BIRTH_YEAR = 1920
MAX_INCOME = 200000
SPENDING_COLUMNS = [
    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds",
]
PURCHASE_COLUMNS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CAMPAIGN_COLUMNS = [f"AcceptedCmp{i}" for i in range(1, 6)]
RAW_COLUMNS = [
    "ID", "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome",
    "Teenhome", "Dt_Customer", "Recency", *SPENDING_COLUMNS, "NumDealsPurchases",
    *PURCHASE_COLUMNS, "NumWebVisitsMonth", *CAMPAIGN_COLUMNS, "Complain",
    "Z_CostContact", "Z_Revenue", "Response",
]


def validate_raw_data(df: pd.DataFrame) -> None:
    """Bao loi ro rang khi sai schema/dinh dang; khong tu suy doan nhan."""
    missing = sorted(set(RAW_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Thieu cot {missing}. File marketing_campaign.csv phai doc voi sep='\\t'.")
    if df.empty:
        raise ValueError("Du lieu rong.")
    for col in set(RAW_COLUMNS) - {"Education", "Marital_Status", "Dt_Customer"}:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if (numeric.isna() & df[col].notna()).any():
            raise ValueError(f"Gia tri khong phai so trong cot {col}.")
        if col != "Income" and numeric.isna().any():
            raise ValueError(f"Cot {col} khong duoc thieu gia tri.")
    for col in ["Response", "Complain", *CAMPAIGN_COLUMNS]:
        if not pd.to_numeric(df[col]).isin([0, 1]).all():
            raise ValueError(f"Cot {col} chi duoc nhan 0/1.")
    dates = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y", errors="coerce")
    if dates.isna().any() or (dates > pd.Timestamp(REFERENCE_DATE)).any():
        raise ValueError("Dt_Customer sai dinh dang dd-mm-yyyy hoac vuot moc 2014-06-30.")
    # ID trung voi thong tin khac nhau can duoc doi chieu, khong tu chon mot dong.
    if df.drop_duplicates()["ID"].duplicated().any():
        raise ValueError("ID trung nhung thong tin khac nhau; can doi chieu du lieu goc.")


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
    validate_raw_data(df)
    return df


def clean_raw_data(
    df: pd.DataFrame,
    drop_uninformative: bool = True,
    filter_outliers: bool = True,
) -> pd.DataFrame:
    """Làm sạch sơ bộ dữ liệu ban đầu.

    - Loại bỏ các cột không mang thông tin phân lớp/hồi quy: 'ID', 'Z_CostContact', 'Z_Revenue'.
    - Lọc Year_Birth ngoài [1920, 2014], Income > 200000 theo quy tắc khảo sát.
    - Income âm được coi là thiếu; không điền median trước khi chia tập.
    - Giữ khách hàng khác ID dù có cùng hồ sơ; bàn giao nhóm trùng riêng.

    Args:
        df: DataFrame gốc.
        drop_uninformative: Loại bỏ các cột vô giá trị dự báo.
        filter_outliers: Lọc các bản ghi outlier rõ rệt.

    Returns:
        DataFrame đã được làm sạch cơ bản.
    """
    data = df.copy()
    numeric_cols = (set(RAW_COLUMNS) & set(data.columns)) - {"Education", "Marital_Status", "Dt_Customer"}
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="raise")
    if "Income" in data.columns:
        data.loc[data["Income"] < 0, "Income"] = float("nan")

    # 1. Loại bỏ bản ghi hoàn toàn trùng lặp
    data = data.drop_duplicates().reset_index(drop=True)

    # 2. Loại bỏ cột định danh và cột hằng số
    if drop_uninformative:
        cols_to_drop = [col for col in ["ID", "Z_CostContact", "Z_Revenue"] if col in data.columns]
        data = data.drop(columns=cols_to_drop)

    # 3. Lọc bỏ các bản ghi dị thường về năm sinh và thu nhập
    if filter_outliers:
        if "Year_Birth" in data.columns:
            # Quy tắc miền giá trị; năm 1920 tương ứng 94 tuổi tại mốc 2014.
            data = data[data["Year_Birth"].between(MIN_BIRTH_YEAR, 2014)]
        if "Income" in data.columns:
            # Lọc bỏ giá trị Income ngoại lai phi thực tế (> 200.000)
            data = data[(data["Income"].isna()) | (data["Income"] <= MAX_INCOME)]

    return data.reset_index(drop=True)


def engineer_features(
    df: pd.DataFrame,
    reference_year: int = 2014,
    drop_redundant: bool = True,
    reference_date: str = REFERENCE_DATE,
) -> pd.DataFrame:
    """Tạo thêm các đặc trưng kinh doanh hữu ích từ các biến có sẵn.

    Các đặc trưng mới:
    - Age: Tuổi của khách hàng (dựa trên reference_year).
    - Customer_Tenure_Days: Số ngày gắn bó tính đến thời điểm khảo sát.
    - Total_Spent: Tổng số tiền chi tiêu cho tất cả danh mục hàng hóa.
    - Total_Purchases: Tổng mua qua Web, Catalog, Store; Deals không phải kênh riêng.
    - Total_Accepted_Cmp: Tổng số lần chấp thuận các chiến dịch trước (Cmp1 -> Cmp5).
    - Children: Tổng số trẻ em trong nhà (Kidhome + Teenhome).
    - Is_Parent: Có con trong gia đình hay không (0 hoặc 1).
    - Family_Size: Kích thước hộ gia đình ước tính.

    Args:
        df: DataFrame đầu vào.
        reference_year: Năm mốc để tính tuổi và thời gian (dữ liệu khảo sát ~2012-2014).
        drop_redundant: Loại bỏ các cột gốc đã được suy biến (như Year_Birth, Dt_Customer).
        reference_date: Mốc cố định cho tenure, không lấy max ngày của từng batch.

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
        cutoff = pd.Timestamp(reference_date)
        if dt_series.isna().any() or (dt_series > cutoff).any():
            raise ValueError("Dt_Customer thieu hoac sau reference_date.")
        data["Customer_Tenure_Days"] = (cutoff - dt_series).dt.days

    # Chuẩn hóa Marital_Status về các nhóm lớn
    if "Marital_Status" in data.columns:
        status_map = {
            "Married": "Partner",
            "Together": "Partner",
            "Single": "Single",
            "Divorced": "Single",
            "Widow": "Single",
            "Alone": "Single",
            "Absurd": "Unknown",
            "YOLO": "Unknown",
        }
        data["Marital_Group"] = data["Marital_Status"].map(status_map).fillna("Unknown")

    # Chuẩn hóa Education
    if "Education" in data.columns:
        # Giữ 5 nhãn gốc: không suy đoán tương đương bằng cấp của "2n Cycle".
        data["Education_Level"] = data["Education"].fillna("Unknown")

    # Tổng chi tiêu
    spending_cols = SPENDING_COLUMNS
    present_spent = [c for c in spending_cols if c in data.columns]
    if present_spent:
        data["Total_Spent"] = data[present_spent].sum(axis=1, min_count=len(present_spent))

    # Tổng số lần mua
    purchase_cols = PURCHASE_COLUMNS
    present_purchases = [c for c in purchase_cols if c in data.columns]
    if present_purchases:
        data["Total_Purchases"] = data[present_purchases].sum(axis=1, min_count=len(present_purchases))

    # Tổng chiến dịch đã phản hồi trước đó
    campaign_cols = CAMPAIGN_COLUMNS
    present_campaigns = [c for c in campaign_cols if c in data.columns]
    if present_campaigns:
        data["Total_Accepted_Cmp"] = data[present_campaigns].sum(axis=1, min_count=len(present_campaigns))

    # Con cái và quy mô gia đình
    if "Kidhome" in data.columns and "Teenhome" in data.columns:
        data["Children"] = data["Kidhome"] + data["Teenhome"]
        data["Is_Parent"] = (data["Children"] > 0).astype(int)

        if "Marital_Group" in data.columns:
            data["Family_Size"] = (
                data["Marital_Group"].map({"Partner": 2, "Single": 1}) + data["Children"]
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
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", scaler),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
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


def create_handoff(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tao du lieu chua impute/encode/scale va metadata tach rieng khoi features.

    duplicate_group chi dua vao X (khong Response). Cac dong cung nhom phai
    o cung mot phia khi chia train/validation de tranh ho so giong nhau bi ro ri.
    """
    validate_raw_data(raw)
    retained = clean_raw_data(raw, drop_uninformative=False)
    featured = engineer_features(retained).drop(columns=["ID", "Z_CostContact", "Z_Revenue"])
    row_lookup = raw.reset_index(drop=True).reset_index().drop_duplicates("ID").set_index("ID")["index"]
    hashes = pd.util.hash_pandas_object(featured.drop(columns="Response"), index=False)
    metadata = pd.DataFrame({
        "processed_row": range(len(featured)),
        "source_row": retained["ID"].map(row_lookup).to_numpy(),
        "customer_id": retained["ID"].to_numpy(),
        "duplicate_group": hashes.map(lambda value: f"{value:016x}").to_numpy(),
    })
    return featured, metadata


def export_handoff(raw: pd.DataFrame) -> dict:
    """Xuat CSV local va thong ke tong hop; khong fit bat ky transformer nao."""
    featured, metadata = create_handoff(raw)
    config.ensure_dirs()
    save_processed(featured, "cleaned_marketing_campaign.csv")
    save_processed(metadata, "customer_row_metadata.csv")
    num_cols, cat_cols = split_feature_types(featured)
    schema = pd.DataFrame({
        "column": featured.columns,
        "dtype": featured.dtypes.astype(str).values,
        "role": ["target" if c == "Response" else "feature" for c in featured],
        "kind": ["categorical" if c in cat_cols else "numeric" for c in featured],
        "missing": featured.isna().sum().values,
        "unique": featured.nunique().values,
    })
    schema.to_csv(config.TABLES_DIR / "preprocessing_feature_schema.csv", index=False)
    for name, frame in [("raw", raw), ("cleaned", featured)]:
        profile = pd.DataFrame({
            "dtype": frame.dtypes.astype(str),
            "non_null": frame.notna().sum(), "missing": frame.isna().sum(),
            "missing_pct": frame.isna().mean().mul(100), "unique": frame.nunique(),
        }).rename_axis("column")
        profile.to_csv(config.TABLES_DIR / f"preprocessing_{name}_profile.csv")
        frame.describe().T.rename_axis("column").to_csv(
            config.TABLES_DIR / f"preprocessing_{name}_statistics.csv")
    category_counts = []
    for stage, frame, columns in [
        ("raw", raw, ["Education", "Marital_Status", "Response"]),
        ("cleaned", featured, cat_cols + ["Response"]),
    ]:
        for column in columns:
            for value, count in frame[column].value_counts(dropna=False).items():
                category_counts.append({"stage": stage, "column": column, "value": value,
                                        "count": int(count), "percent": count / len(frame) * 100})
    pd.DataFrame(category_counts).to_csv(config.TABLES_DIR / "preprocessing_category_counts.csv", index=False)
    unique_raw = raw.drop_duplicates()
    invalid_birth = ~unique_raw["Year_Birth"].between(MIN_BIRTH_YEAR, 2014)
    invalid_income = unique_raw["Income"] > MAX_INCOME
    summary = {
        "schema_version": 1,
        "reference_date": REFERENCE_DATE,
        "raw_rows": len(raw), "raw_columns": len(raw.columns),
        "exact_duplicates_removed": int(raw.duplicated().sum()),
        "birth_outliers": int(invalid_birth.sum()),
        "income_outliers": int(invalid_income.sum()),
        "outlier_rows_removed": int((invalid_birth | invalid_income).sum()),
        "cleaned_rows": len(featured), "features": len(num_cols) + len(cat_cols),
        "numeric_features": len(num_cols), "categorical_features": len(cat_cols),
        "missing_after_cleaning": {c: int(n) for c, n in featured.isna().sum().items() if n},
        "raw_response_counts": {str(k): int(v) for k, v in raw.Response.value_counts().items()},
        "cleaned_response_counts": {str(k): int(v) for k, v in featured.Response.value_counts().items()},
        "duplicate_profiles_excluding_id_raw": int(raw.drop(columns="ID").duplicated().sum()),
        "duplicate_feature_rows_cleaned": int(metadata.duplicate_group.duplicated().sum()),
        "duplicate_groups_with_multiple_rows": int((metadata.duplicate_group.value_counts() > 1).sum()),
        "categorical_values": {c: sorted(featured[c].dropna().unique().tolist()) for c in cat_cols},
        "imputed": False, "encoded": False, "scaled": False,
    }
    (config.TABLES_DIR / "preprocessing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(export_handoff(load_raw()), ensure_ascii=False, indent=2))
