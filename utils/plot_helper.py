from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_distributions_by_condition(df, columns):
    n = len(columns)
    _, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 4 * n))

    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        counts = df.groupby(["condition", col]).size().unstack(fill_value=0)
        counts.plot(kind="bar", ax=ax, stacked=False, legend=True)
        ax.set_title(f"Distribución de {col} por (Nuevo/Usado)")
        ax.set_ylabel("Frecuencia")
        ax.set_xlabel("condition")
        ax.legend(title=col)

        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(str(int(height)), xy=(bar.get_x() + bar.get_width() / 2, height), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_percentile_distributions_by_condition(df, numeric_columns, quantiles=10):
    n = len(numeric_columns)
    _, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 4 * n))

    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_columns):
        clean_df = df[[col, "condition"]].dropna()

        try:
            clean_df["quantile_bin"] = pd.qcut(clean_df[col], q=quantiles, duplicates="drop")

            counts = clean_df.groupby(["quantile_bin", "condition"], observed=False).size().unstack(fill_value=0)

            counts.plot(kind="bar", ax=ax, stacked=False)
            ax.set_title(f"Distribución de {col} por percentiles y condición")
            ax.set_ylabel("Frecuencia")
            ax.set_xlabel(f"{col} (percentiles)")
            ax.legend(title="condition")
            ax.tick_params(axis="x", rotation=45)

        except ValueError:
            ax.set_title(f"{col} no tiene suficientes valores únicos para {quantiles} percentiles")
            ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_model_metrics(y, y_pred, X, model):
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.savefig("./docs/Matrix.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    importances = model.feature_importances_
    features = pd.Series(importances, index=X.columns)
    features = features[features > 0].sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=features.values, y=features.index)
    plt.title("Importancia de Variables en XGBoost")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_validation(X, y, model):
    n_splits = 100
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies = []
    precisions = []

    X_np = X.values
    y_np = y.values

    for _, test_idx in skf.split(X_np, y_np):
        X_test = X_np[test_idx]
        y_test = y_np[test_idx]
        y_pred_test = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, average="binary")
        accuracies.append(acc)
        precisions.append(prec)

    plt.figure(figsize=(8, 6))
    plt.scatter(accuracies, precisions, alpha=0.7)
    plt.xlabel("Accuracy")
    plt.ylabel("Precision")
    plt.title("Scatter plot of Accuracy vs Precision (100 random blocks)")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig("./docs/Validation.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
