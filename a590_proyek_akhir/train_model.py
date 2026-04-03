from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_URL = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/students_performance/data.csv"
ARTIFACT_PATH = Path(__file__).resolve().parent / "model" / "student_performance_pipeline.joblib"
FEATURE_COLUMNS = [
    "Marital_status",
    "Application_mode",
    "Course",
    "Daytime_evening_attendance",
    "Previous_qualification_grade",
    "Admission_grade",
    "Debtor",
    "Tuition_fees_up_to_date",
    "Gender",
    "Scholarship_holder",
    "Age_at_enrollment",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade",
]
CATEGORICAL_COLUMNS = [
    "Marital_status",
    "Application_mode",
    "Course",
    "Daytime_evening_attendance",
    "Debtor",
    "Tuition_fees_up_to_date",
    "Gender",
    "Scholarship_holder",
]
TARGET_STATUSES = ["Dropout", "Graduate"]


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only final labels so the classifier learns a valid binary target."""
    train_df = df[df["Status"].isin(TARGET_STATUSES)].copy()
    train_df["is_dropout"] = (train_df["Status"] == "Dropout").astype(int)
    return train_df


def extract_feature_importance(pipeline: Pipeline, top_n: int = 10) -> list[dict]:
    """Return top absolute logistic coefficients with their sign."""
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()
    coefficients = model.coef_.ravel()

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)

    top_features = importance_df.head(top_n)
    return [
        {
            "feature": row["feature"],
            "coefficient": float(row["coefficient"]),
            "abs_coefficient": float(row["abs_coefficient"]),
        }
        for _, row in top_features.iterrows()
    ]


def main() -> None:
    df = pd.read_csv(DATA_URL, sep=";")
    train_df = prepare_training_data(df)

    x = train_df[FEATURE_COLUMNS]
    y = train_df["is_dropout"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            )
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    feature_importance = extract_feature_importance(pipeline)
    class_distribution = train_df["Status"].value_counts().to_dict()

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "training_rows": int(len(train_df)),
            "class_distribution": class_distribution,
            "target_statuses": TARGET_STATUSES,
            "sklearn_version": sklearn.__version__,
        },
        ARTIFACT_PATH,
    )

    print("Model artifact saved:", ARTIFACT_PATH)
    print("Training statuses:", TARGET_STATUSES)
    print("Training rows:", len(train_df))
    print("Metrics:", metrics)
    print("Top feature importance:", feature_importance)


if __name__ == "__main__":
    main()
