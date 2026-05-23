import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(
    ROOT,
    "insurance.csv"
)

MODEL_PATH = os.path.join(
    ROOT,
    "model",
    "insurance_model.pkl"
)

os.makedirs(
    os.path.join(ROOT, "model"),
    exist_ok=True
)

# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    raise FileNotFoundError(
        "insurance.csv not found"
    )

# ============================================================
# TRAIN MODEL
# ============================================================

def preprocess_and_train():

    df = load_data()

    df = df.dropna()

    TARGET = "annual_premium"

    X = df.drop(columns=[TARGET])

    y = df[TARGET]

    # ============================================================
    # FEATURE TYPES
    # ============================================================

    numeric_feats = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_feats = X.select_dtypes(
        include=["object", "bool"]
    ).columns.tolist()

    # ============================================================
    # PREPROCESSOR
    # ============================================================

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "num",

                Pipeline([
                    ("scaler", StandardScaler())
                ]),

                numeric_feats
            ),

            (
                "cat",

                Pipeline([
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]),

                categorical_feats
            )

        ]
    )

    # ============================================================
    # MODEL
    # ============================================================

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # ============================================================
    # PIPELINE
    # ============================================================

    pipeline = Pipeline([

        ("pre", preprocessor),

        ("rf", model)

    ])

    # ============================================================
    # TRAIN TEST SPLIT
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ============================================================
    # TRAIN
    # ============================================================

    pipeline.fit(X_train, y_train)

    # ============================================================
    # SAVE MODEL
    # ============================================================

    joblib.dump(
        pipeline,
        MODEL_PATH,
        compress=3
    )

    print(
        f"[SUCCESS] Model saved at {MODEL_PATH}"
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    preprocess_and_train()
