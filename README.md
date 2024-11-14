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

# About

This is the source and data repository for the using ML models to predicting the cellular phospholipidosis activity of compounds. A subsequent publication titled "Predicting cellular phospholipidosis on different cell lines using repurposing libraries and machine learning" is under preparation.

For our model training workflow, we have leveraged both KNIME and Python frameworks to allow both communities to reuse our work. Below we describe in detail the Python framework only. For the KNIME framework, please take a look [here](). # TODO: Add link to KNIME space

# Data organization

TODO: Add tree here

# How to use our model?


# How did we build the dataset?

The dataset was built on the KNIME workflow. So more details can be found either in our manuscript or the KNIME workflow.

# How to build own model in Python?

We use the conda environment to build and run our codes. Please follow the following steps to build the conda environment with all the necessary python packages
```bash
git clone https://github.com/Fraunhofer-ITMP/PLD.git
conda create --name=pld python=3.9
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


Sample the modelling effort on Phospholipidosis together with Karolinska data.

It contains data input KNIME workflow and a Python notebook which implements XGBoost classification model reported in the publications (ref)

Show_database_app.py is a Streamlit app which allows user to "see" the training set that has been used and eventually the XOR dataset which is not part of the training set. Moreover, it shows the top10 most important features of any model saved as pickle file and provides a set of boxplot to visualize how much these features really are different in the labelling group ('Active' - 'Inactive')

