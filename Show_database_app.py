import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from scipy import stats






uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    df = df.dropna()
    st.write("Data preview:", df.head())
    

if 'df' in locals():
    flat_columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in df.columns]
    label_column = st.selectbox("Select the label column", df.columns)
    if '_' in label_column:
        label_column = tuple(label_column.split('_'))
    


def create_plots(df, label_column):
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_columns].drop(label_column, axis=1, errors='ignore')
    
    # Flatten X for PCA and t-SNE
    X_flat = pd.DataFrame(X.values, columns=[f"{col[0]}_{col[1]}" for col in X.columns])
    
    # Convert label column to categorical
    labels = df[label_column].values
    label_categories = pd.Categorical(labels)
    
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)
    
    pca_result = pca.fit_transform(X_flat)
    tsne_result = tsne.fit_transform(X_flat)
    
    # Create a color map
    unique_labels = np.unique(labels)
    color_map = dict(zip(unique_labels, sns.color_palette("husl", len(unique_labels))))
    colors = [color_map[label] for label in labels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=colors)
    ax1.set_title("PCA")
    
    scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors)
    ax2.set_title("t-SNE")
    
    # Create a custom legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=10, label=label)
                       for label, color in color_map.items()]
    fig.legend(handles=legend_elements, title=str(label_column), loc='center right')
    
    plt.tight_layout()
    st.pyplot(fig)


if 'df' in locals() and 'label_column' in locals():
    create_plots(df, label_column)

    

model_file = st.file_uploader("Upload your model (PKL file)", type="pkl")
if model_file is not None:
    model = pickle.load(model_file)
    st.write("Model loaded successfully")
    
def safe_feature_importance(model, X):
    importances = model.feature_importances_
    feature_names = X.columns.tolist()

    # Use the minimum length to avoid index mismatch
    min_length = min(len(importances), len(feature_names))
    feature_importance_df = pd.DataFrame({
        'feature': feature_names[:min_length],
        'importance': importances[:min_length]
    })

    return feature_importance_df.sort_values('importance', ascending=False)

def plot_feature_importance(model, X):
    feature_importance_df = safe_feature_importance(model, X)
    top_10 = feature_importance_df.head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_10['feature'], top_10['importance'])

    ax.set_title("Top 10 Most Influential Columns")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    return feature_importance_df

def create_boxplots(df, label_column, top_features):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Top 9 Most Significant Features Comparison per Groups', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features[:9]):
        if feature in df.columns:
            sns.boxplot(x=df[label_column], y=df[feature], ax=axes[i])
            axes[i].set_title(f"{feature}")
            axes[i].set_xlabel('')
            
            # Calculate p-value
            groups = [group for _, group in df.groupby(label_column)[feature]]
            f_value, p_value = stats.f_oneway(*groups)
            axes[i].text(0.05, 0.95, f'p-value: {p_value:.4f}', transform=axes[i].transAxes, 
                         verticalalignment='top')
        else:
            axes[i].text(0.5, 0.5, f"Feature '{feature}' not found", 
                         ha='center', va='center')
            axes[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

if 'model' in locals() and 'df' in locals():
    # Ensure X only contains numeric columns
    X = df.drop(label_column, axis=1).select_dtypes(include=[np.number])
    
    # Plot feature importance
    feature_importance_df = plot_feature_importance(model, X)
    
    # Display warning about mismatch
    if len(model.feature_importances_) != len(X.columns):
        st.warning(f"Mismatch between feature importances ({len(model.feature_importances_)}) "
                   f"and DataFrame columns ({len(X.columns)}). "
                   "Some features may have been dropped during model training.")
    
    # Create boxplots
    top_features = feature_importance_df['feature'].tolist()
    create_boxplots(df, label_column, top_features)