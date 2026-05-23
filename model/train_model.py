import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ---------------- PATHS ----------------
ROOT = os.path.dirname(os.path.dirname(__file__))
<<<<<<< HEAD
=======
DATA_PATH = os.path.join(ROOT, "insurance.csv")
MODEL_PATH = os.path.join(ROOT, "model", "insurance_model.pkl")
>>>>>>> f01b809ff3737afbf70c4a60185b793acfb67141

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

# ---------------- LOAD DATA ----------------
def load_data():
<<<<<<< HEAD
=======
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        raise FileNotFoundError("insurance.csv not found")
>>>>>>> f01b809ff3737afbf70c4a60185b793acfb67141

    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    raise FileNotFoundError(
        "insurance.csv not found"
    )

# ---------------- TRAIN ----------------
def preprocess_and_train():

    df = load_data()

    df = df.dropna()

    TARGET = "annual_premium"

    X = df.drop(columns=[TARGET])

    y = df[TARGET]

<<<<<<< HEAD
    numeric_feats = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_feats = X.select_dtypes(
        include=["object"]
=======
    numeric_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_feats = X.select_dtypes(
        include=["object", "bool"]
>>>>>>> f01b809ff3737afbf70c4a60185b793acfb67141
    ).columns.tolist()

    # ---------------- PREPROCESSOR ----------------
    preprocessor = ColumnTransformer(
        transformers=[
<<<<<<< HEAD

            (
                "num",

                Pipeline([
                    ("scaler", StandardScaler())
                ]),

=======
            (
                "num",
                Pipeline([
                    ("scaler", StandardScaler())
                ]),
>>>>>>> f01b809ff3737afbf70c4a60185b793acfb67141
                numeric_feats
            ),

            (
                "cat",
<<<<<<< HEAD

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

    # ---------------- MODEL ----------------
=======
                Pipeline([
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_feats
            )
        ]
    )

    # ---------------- SMALLER MODEL ----------------
>>>>>>> f01b809ff3737afbf70c4a60185b793acfb67141
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
<<<<<<< HEAD

        ("pre", preprocessor),

        ("rf", model)

=======
        ("pre", preprocessor),
        ("rf", model)
>>>>>>> f01b809ff3737afbf70c4a60185b793acfb67141
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline.fit(X_train, y_train)

<<<<<<< HEAD
    # ---------------- SAVE MODEL ----------------
=======
    # ---------------- SAVE SMALL MODEL ----------------
>>>>>>> f01b809ff3737afbf70c4a60185b793acfb67141
    joblib.dump(
        pipeline,
        MODEL_PATH,
        compress=3
    )
<<<<<<< HEAD
=======

    print(f"[SUCCESS] Model saved at {MODEL_PATH}")
>>>>>>> f01b809ff3737afbf70c4a60185b793acfb67141

    print(
        f"[SUCCESS] Model saved at {MODEL_PATH}"
    )

if __name__ == "__main__":

    preprocess_and_train()
