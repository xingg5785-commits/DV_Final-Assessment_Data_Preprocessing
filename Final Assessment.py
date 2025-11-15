import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

xls = pd.ExcelFile(r"D:\OneDrive\桌面\Food Environment Atlas Data.xls")
print(xls.sheet_names)

sheets = [
    "ACCESS","STORES","RESTAURANTS","ASSISTANCE","INSECURITY",
    "PRICES_TAXES","LOCAL","HEALTH","SOCIOECONOMIC"
]

# Columns to keep original
geo_columns = ["FIPS", "State", "County"]

# Dictionary to store cleaned DataFrames
cleaned = {}

for s in sheets:
    print("\n======================")
    print(f"DESCRIBE FOR SHEET: {s}")
    print("======================\n")

    df = pd.read_excel(xls, sheet_name=s)

    # Explore Data Analysis
    print("Print the number of rows: {}".format(df.shape[0]))
    print("Print the number of columns: {}".format(df.shape[1]))
    print()

    print(f"Display basic statistics of {s}:")
    print(df.describe(include="all"))
    print()

    print(f"Listing the basic information of {s}:")
    print(df.head(5))
    print()

    print(f"Find out the data type for {s}:")
    print(df.dtypes)
    print(df.head(5))
    print()

    print(f"Find out the missing values for {s}:")
    print(df.isnull().sum().sort_values(ascending=False))
    print()

    # ==== Remove the missing values ====
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
		c for c in df.select_dtypes(include=["object"]).columns
		if c not in geo_columns  # Protect geographic fields
	]

    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Standardize FIPS for later merge (create FIPS_str, keep original FIPS)
    if "FIPS" in df.columns:
        df["FIPS_str"] = df["FIPS"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)

    # Impute numeric columns (only if present)
    if len(numeric_features) > 0:
        df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])
    else:
        print(f"Note: no numeric columns to impute for {s}.")

    # Impute categorical columns only if there are non-ID categorical cols to impute
    if len(categorical_features) > 0:
        print(f"Categorical columns to impute for sheet {s}: {categorical_features_to_impute}")
        df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
    else:
        print(f"No categorical columns to impute for sheet {s} (only ID categorical columns or none).")

    print(f"\nCheck the missing values after imputation:")
    print(df.isnull().sum().sort_values(ascending=False))

    # ==== Encoding categorical data ====
    # Select Non-Numerical Columns
    label_encoder = LabelEncoder()
    for col in categorical_features:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

    print("\nLabel Encoder Data:")
    print(df.head())

    # ==== Feature scaling ====
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    print("\nScaled Data:")
    print(df.head())

    cleaned[s] = df.copy()
    print(f"Stored cleaned sheet for: {s}")

out_path = r"D:\OneDrive\桌面\Cleaned_Food_Environment_Data.xlsx"
with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    for sheet_name, df_clean in cleaned.items():
        safe_name = sheet_name[:31]
        df_clean.to_excel(writer, sheet_name=safe_name, index=False)

print("Saved cleaned workbook to:", out_path)
