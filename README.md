# PLD
Sample the modelling effort on Phospholipidosis together with Karolinska data.

It contains data input KNIME workflow and a Python notebook which implements XGBoost classification model reported in the publications (ref)

Show_database_app.py is a Streamlit app which allows user to "see" the training set that has been used and eventually the XOR dataset which is not part of the training set. Moreover, it shows the top10 most important features of any model saved as pickle file and provides a set of boxplot to visualize how much these features really are different in the labelling group ('Active' - 'Inactive')

