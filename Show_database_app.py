# encoding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="PLD Analysis App", page_icon="🧊", layout="wide")

st.title("Data Overview and Model Analysis App")

st.header("Upload your data")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is None:
    uploaded_file = "data/processed/model_data_subset.csv"
    st.warning("No file uploaded, using default data instead", icon="⚠️")

df = pd.read_csv(uploaded_file, sep=";")
df = df.dropna()
st.write("Data preview:", df.head())

flat_columns = [
    f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in df.columns
]
label_column = st.selectbox("Select the label column", df.columns[1])
if "_" in label_column:
    label_column = tuple(label_column.split("_"))

if st.button("Plot data distribution"):
    st.header("Dimensionality Reduction and Visualization")

    def create_plots(df, label_column):
        # Select numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_columns].drop(label_column, axis=1, errors="ignore")

        # Flatten X for PCA and t-SNE
        X_flat = pd.DataFrame(
            X.values, columns=[f"{col[0]}_{col[1]}" for col in X.columns]
        )

        # Convert label column to categorical
        labels = df[label_column].values
        # label_categories = pd.Categorical(labels)

        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2)

        pca_result = pca.fit_transform(X_flat)
        tsne_result = tsne.fit_transform(X_flat)

        color_map = ["red" if label == "active" else "blue" for label in labels]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("PCA", "TSNE"))

        # create scatter plot for PCA with color encoding
        fig.add_trace(
            go.Scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                mode="markers",
                marker=dict(color=color_map, opacity=0.5, size=2),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # create scatter plot for TSNE with color encoding
        fig.add_trace(
            go.Scatter(
                x=tsne_result[:, 0],
                y=tsne_result[:, 1],
                mode="markers",
                marker=dict(color=color_map, opacity=0.5, size=2),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(title_text="PCA and TSNE Visualization")
        fig.update_xaxes(title_text="PCA1", row=1, col=1)
        fig.update_yaxes(title_text="PCA2", row=1, col=1)
        fig.update_xaxes(title_text="TSNE1", row=1, col=2)
        fig.update_yaxes(title_text="TSNE2", row=1, col=2)

        st.plotly_chart(fig)

    create_plots(df, label_column)


st.header("Model Prediction")


def safe_feature_importance(model, X):
    importances = model.feature_importances_
    feature_names = X.columns.tolist()

    # Use the minimum length to avoid index mismatch
    min_length = min(len(importances), len(feature_names))
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names[:min_length], "importance": importances[:min_length]}
    )

    return feature_importance_df.sort_values("importance", ascending=False)


def plot_feature_importance(model, X):
    feature_importance_df = safe_feature_importance(model, X)
    top_10 = feature_importance_df.head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_10["feature"], top_10["importance"])

    ax.set_title("Top 10 Most Influential Columns")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    st.pyplot(fig)

    return feature_importance_df


def create_boxplots(df, label_column, top_features):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        "Top 9 Most Significant Features Comparison per Groups",
        fontsize=16,
        fontweight="bold",
    )
    axes = axes.flatten()

    for i, feature in enumerate(top_features[:9]):
        if feature in df.columns:
            sns.boxplot(x=df[label_column], y=df[feature], ax=axes[i])
            axes[i].set_title(f"{feature}")
            axes[i].set_xlabel("")

            # Calculate p-value
            groups = [group for _, group in df.groupby(label_column)[feature]]
            f_value, p_value = stats.f_oneway(*groups)
            axes[i].text(
                0.05,
                0.95,
                f"p-value: {p_value:.4f}",
                transform=axes[i].transAxes,
                verticalalignment="top",
            )
        else:
            axes[i].text(
                0.5, 0.5, f"Feature '{feature}' not found", ha="center", va="center"
            )
            axes[i].axis("off")

    plt.tight_layout()
    st.pyplot(fig)


model_file = st.file_uploader("Upload your model (PKL file)", type="pkl")
if model_file is not None:
    model = pickle.load(model_file)
    st.write("Model loaded successfully")

    # Ensure X only contains numeric columns
    X = df.drop(label_column, axis=1).select_dtypes(include=[np.number])

    # Plot feature importance
    feature_importance_df = plot_feature_importance(model, X)

    # Display warning about mismatch
    if len(model.feature_importances_) != len(X.columns):
        st.warning(
            f"Mismatch between feature importances ({len(model.feature_importances_)}) "
            f"and DataFrame columns ({len(X.columns)}). "
            "Some features may have been dropped during model training."
        )

    # Create boxplots
    top_features = feature_importance_df["feature"].tolist()
    create_boxplots(df, label_column, top_features)
