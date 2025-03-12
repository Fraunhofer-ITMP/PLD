<h1 align="center">
  Predicting cellular phospholipidosis on different cell lines using repurposing libraries and machine learning
  <br/>

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 <!-- [![DOI:10.1093/bioinformatics/btac716](http://img.shields.io/badge/DOI-110.1093/bioinformatics/btac716-B31B1B.svg)](https://doi.org/10.1093/bioinformatics/btac716) -->
</h1>


# TOC

* [About](#about)
* [Organization](#data-organization)
* [Dataset building](#how-did-we-build-the-dataset)
* [Model training](#how-to-build-own-model-in-knime)
* [Citation](#citation)

# About

This is the source and data repository for the using ML models to predicting the cellular phospholipidosis activity of compounds. A subsequent publication titled "Predicting cellular phospholipidosis on different cell lines using repurposing libraries and machine learning" is under preparation.

For our model training workflow, we have leveraged both KNIME and Python frameworks to allow both communities to reuse our work. Below we describe in detail the Python framework only. For the KNIME framework, please take a look [here](https://hub.knime.com/s/m6rnKt_4iYtDI1yt).

# Data organization

```
.
├── data
│   ├── processed
│   │   ├── complete_data.tsv
│   │   └── model_data_subset.csv
│   └── raw
│       └── Phospholipidosis_KI_ITMP.xlsx
├── figures
│   ├── Confusion_Matrix_XGBoost_SMOTE_chemphys.JPG
│   ├── PLD_tSNE.html
│   ├── confusion_matrix_PLD_XGBoost.png
│   ├── feature_importance.png
│   ├── figure_1.png
│   ├── figure_2.png
│   ├── supplementary_figure_1.png
│   ├── top10_features_boxplots.svg
│   └── top10_features_boxplots_with_significance.svg
├── models
│   ├── final_model_PLD_XGBoost.pkl
│   └── final_model_PLD_XGBoost_params.json
├── notebooks
│   ├── 0_data_eda.ipynb
│   ├── 1_data_processing.ipynb
│   ├── 2_model_training.ipynb
│   └── 3_feature_importance.ipynb
├── LICENSE
├── Phospholipidosis_v4_AND.knwf
├── README.md
├── Show_database_app.py
└── requirements.txt
```

# How did we build the dataset?

The dataset was built on the KNIME workflow. So more details can be found either in our manuscript or the KNIME workflow.

# How to build own model in Python?

We use the conda environment to build and run our codes. Please follow the following steps to build the conda environment with all the necessary python packages
```bash
git clone https://github.com/Fraunhofer-ITMP/PLD.git
conda create --name=pld python=3.10
conda activate pld
conda cd PLD
pip install -r requirements.txt
```
To use the Jupyter notebooks, you need to ensure that the conda environment is available for use. To do so, following the following lines in the terminal.
```bash
pip install ipykernel
python -m ipykernel install --user --name=pld
```
After this, *"pld"* should be displayed as a kernel in your VSCode environment. Alternatively, you could spin the jupyter notebook from the conda environment itself using the following command: `jupyter notebook`

# How to build own model in KNIME?

You can do so by downloading the [workflow](https://hub.knime.com/s/m6rnKt_4iYtDI1yt) locally and running it.

# Overview of the model

[Show_database_app.py](Show_database_app.py) is a Streamlit app which allows user to "see" the training set that has been used and eventually the XOR dataset which is not part of the training set. Moreover, it shows the top10 most important features of any model saved as pickle file and provides a set of boxplot to visualize how much these features really are different in the labelling group ('Active' - 'Inactive')


# Citation

If you use our work, please cite us as follows:
> Maria K. *et al.*, Interdisciplinary Study on Drug-Induced-Phospholipidosis of Repurposing Libraries through Machine Learning and Experimental Evaluation in Different Cell Lines. *In press*