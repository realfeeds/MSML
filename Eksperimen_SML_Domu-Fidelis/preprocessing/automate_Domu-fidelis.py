import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump

def preprocess_data(path):
    df = pd.read_csv(path)

    # Missing values
    cont_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] # kolom yg nol dianggap missing
    df[cont_cols] = df[cont_cols].replace(0,np.nan)
    df.isnull().sum()

    for col in cont_cols:
        df[col].fillna(df[col].median(), inplace=True)
    df.isnull().sum()
    print("Missing values handled")

    # Duplicate values
    df.drop_duplicates(inplace=True)
    print(f"Shape after dropping duplicates: {df.shape}")
    print("Duplicates handled")

    # Split
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )
    print("Data split")

    # Scale
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data scaled")

    df_train_preprocessed = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
    )

    df_test_preprocessed = pd.DataFrame(
        X_test_scaled,
        columns=X.columns
    )

    # Export dataset
    df_train_preprocessed["Outcome"] = y_train.reset_index(drop=True)
    df_test_preprocessed["Outcome"] = y_test.reset_index(drop=True)

    df_train_preprocessed.to_csv(
        "df_train_preprocessing.csv",
        index=False
    )

    df_test_preprocessed.to_csv(
        "df_test_preprocessing.csv",
        index=False
    )

    print("Preprocessed train and test datasets saved successfully")

    # Dump scaler
    dump(scaler, "scaler.pkl")
