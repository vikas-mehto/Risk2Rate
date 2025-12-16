import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except Exception:
    xgb_available = False

# Optional: SHAP
try:
    import shap
    shap_available = True
except Exception:
    shap_available = False


# ---------------- PATHS ----------------
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "insurance.csv")
MODEL_PATH = os.path.join(ROOT, "model", "insurance_model.pkl")
EXPLAINER_PATH = os.path.join(ROOT, "model", "shap_explainer.pkl")

os.makedirs(os.path.join(ROOT, "model"), exist_ok=True)


# ---------------- LOAD DATA ----------------
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        raise FileNotFoundError("insurance.csv not found")
    return df


# ---------------- TRAIN ----------------
def preprocess_and_train():
    df = load_data()
    df = df.dropna()

    # ✅ FIXED: EXPLICIT TARGET COLUMN
    TARGET = "annual_premium"

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. Available columns: {df.columns.tolist()}"
        )

    print(f"[INFO] Using target column: {TARGET}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Feature types
    numeric_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_feats = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    # Preprocessing
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_feats),
            ("cat", categorical_transformer, categorical_feats),
        ]
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("rf", model)
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"[SUCCESS] Model saved to {MODEL_PATH}")

    # ---------------- SHAP ----------------
    if shap_available:
        try:
            explainer = shap.TreeExplainer(pipeline.named_steps["rf"])
            with open(EXPLAINER_PATH, "wb") as f:
                pickle.dump(
                    {
                        "explainer": explainer,
                        "feature_names": numeric_feats
                    },
                    f
                )
            print("[SUCCESS] SHAP explainer saved")
        except Exception as e:
            print("[WARN] SHAP explainer not created:", e)


if __name__ == "__main__":
    preprocess_and_train()
