# 202312-5-Fraudulent-Transaction-Detection

The contents of teh repository are distributed as follows:
- DAG-Files: Directory containing python files pertaiing to each task in the airflow workflow designed for this project
  - data.py: for fetching and merging data files to form the final dataset we used for model training.
  - KNN.py, MLP(1).py, NaiveNayes.py, RF.py, SVC.py, logReg.py, xGB.py: python files containing train and test scripts to be run in the airflow pipeline.
  - fraud.py: airflow script that calls all of the above during airflow execution and collates the performance metrics obtained from each model task in the airflow.
 
- dashboards
  - hist.html: histogram dashboard for visualizing continuous data.
  - pies_identity.html: pie chart dashboard for visualizing categorical data from the identity part of the dataset.
  - pies_transaction.html: pie chart dashboard for visualizing categorical data from the trasaction part of the dataset.
 
- BDA_models.ipynb: python notebook containing experimental script with created before usage in the airflow script which was run on GCP. 

- eda_and_models.ipynb: we conducted exploratory data analysis (EDA) to gain valuable insights into the dataset and better understand the problem at hand, referencing and adapting techniques from a Kaggle notebook  https://www.kaggle.com/code/artgor/eda-and-models#Data-Exploration that provided a comprehensive overview. This aided us in obtaining a superficial understanding of the data and framing our approach.
