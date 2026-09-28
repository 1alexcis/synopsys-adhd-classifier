"""Train and evaluate an ADHD classifier from precomputed EEG features."""

from __future__ import annotations

import argparse
from pathlib import Path

TARGET_COLUMN = "group"
NON_FEATURE_COLUMNS = {TARGET_COLUMN, "file_id", "band", "channel_id"}
NOTEBOOK_FEATURES = [
    "std_dev",
    "rms",
    "skewness",
    "kurtosis",
    "spectral_entropy",
    "band_power",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "shannons_entropy",
]


def load_features(path: Path) -> pd.DataFrame:
    """Load the precomputed feature table expected by the classifier."""
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"features CSV not found: {path}")
    return pd.read_csv(path)


def feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return model feature columns after excluding labels and metadata."""
    if TARGET_COLUMN not in features_df.columns:
        raise ValueError(f"features CSV must include a {TARGET_COLUMN!r} column")

    columns = [column for column in features_df.columns if column not in NON_FEATURE_COLUMNS]
    if not columns:
        raise ValueError("features CSV does not contain any model feature columns")
    return columns


def numeric_features(features_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Validate and return numeric model features."""
    features = features_df[columns].apply(pd.to_numeric, errors="raise")
    missing_counts = features.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        summary = ", ".join(f"{column}={count}" for column, count in missing_counts.items())
        raise ValueError(f"feature columns contain missing values: {summary}")
    return features


def run_anova(features_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Run one-way ANOVA for each feature against the target group."""
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    rows = []
    for feature in columns:
        model = ols(f'Q("{feature}") ~ C({TARGET_COLUMN})', data=features_df).fit()
        table = sm.stats.anova_lm(model, typ=2)
        group_row = f"C({TARGET_COLUMN})"
        rows.append(
            {
                "feature": feature,
                "f_statistic": table.loc[group_row, "F"],
                "p_value": table.loc[group_row, "PR(>F)"],
            }
        )
    return pd.DataFrame(rows).sort_values("p_value")


def run_pca(features: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Run PCA and return PC1 loadings plus cumulative explained variance."""
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaled_features = StandardScaler().fit_transform(features)
    pca = PCA()
    pca.fit(scaled_features)

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])],
        index=features.columns,
    )
    important_features_pc1 = loadings["PC1"].abs().sort_values(ascending=False)
    cumulative_explained_variance = pd.Series(
        pca.explained_variance_ratio_.cumsum(),
        index=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )
    return important_features_pc1, cumulative_explained_variance


def split_data(
    features: pd.DataFrame,
    labels: pd.Series,
    test_size: float,
    validation_size: float,
    random_state: int,
):
    """Split data into train, validation, and test sets."""
    from sklearn.model_selection import train_test_split

    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1")

    x_intermediate, x_test, y_intermediate, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    validation_fraction = validation_size / (1 - test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_intermediate,
        y_intermediate,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=y_intermediate,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_classifier(kernel: str, c_value: float, gamma: str) -> object:
    """Build the scaled SVM classifier used by this workflow."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return make_pipeline(StandardScaler(), SVC(kernel=kernel, C=c_value, gamma=gamma))


def evaluate_holdout(classifier, x_train, x_validation, x_test, y_train, y_validation, y_test):
    """Train on the training split and report validation/test accuracy."""
    from sklearn.metrics import accuracy_score

    classifier.fit(x_train, y_train)
    validation_predictions = classifier.predict(x_validation)
    test_predictions = classifier.predict(x_test)
    return {
        "validation_accuracy": accuracy_score(y_validation, validation_predictions),
        "test_accuracy": accuracy_score(y_test, test_predictions),
    }


def evaluate_cross_validation(classifier, features, labels, folds: int, random_state: int):
    """Run shuffled K-fold cross-validation."""
    from sklearn.model_selection import KFold, cross_val_score

    kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    return cross_val_score(classifier, features, labels, cv=kfold, scoring="accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("data/features_df.csv"),
        help="Path to the precomputed features CSV.",
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kernel", default="rbf", help="SVC kernel.")
    parser.add_argument("--c-value", type=float, default=1.0, help="SVC regularization value.")
    parser.add_argument("--gamma", default="scale", choices=["scale", "auto"], help="SVC gamma mode.")
    parser.add_argument("--skip-anova", action="store_true")
    parser.add_argument("--skip-pca", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_df = load_features(args.features_csv)
    columns = feature_columns(features_df)
    features = numeric_features(features_df, columns)
    labels = features_df[TARGET_COLUMN]

    print(f"Loaded {len(features_df)} rows and {len(columns)} model features.")

    available_notebook_features = [column for column in NOTEBOOK_FEATURES if column in columns]
    if available_notebook_features and not args.skip_anova:
        print("\nANOVA results:")
        print(run_anova(features_df, available_notebook_features).to_string(index=False))

    if not args.skip_pca:
        important_features_pc1, cumulative_explained_variance = run_pca(features)
        print("\nImportant features for PC1:")
        print(important_features_pc1.head().to_string())
        print("\nCumulative explained variance:")
        print(cumulative_explained_variance.to_string())

    split = split_data(
        features,
        labels,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    x_train, x_validation, x_test, y_train, y_validation, y_test = split
    print(
        "\nSplit sizes: "
        f"train={len(x_train)}, validation={len(x_validation)}, test={len(x_test)}"
    )

    classifier = build_classifier(args.kernel, args.c_value, args.gamma)
    holdout_scores = evaluate_holdout(
        classifier,
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
    )
    print(
        "\nHoldout accuracy: "
        f"validation={holdout_scores['validation_accuracy']:.3f}, "
        f"test={holdout_scores['test_accuracy']:.3f}"
    )

    cv_classifier = build_classifier(args.kernel, args.c_value, args.gamma)
    cv_scores = evaluate_cross_validation(
        cv_classifier,
        features,
        labels,
        folds=args.cv_folds,
        random_state=args.random_state,
    )
    print(
        "\nCross-validation accuracy: "
        f"mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}, "
        f"scores={[round(score, 3) for score in cv_scores]}"
    )


if __name__ == "__main__":
    main()
