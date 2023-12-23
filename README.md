# 202312-5-Fraudulent-Transaction-Detection

The contents of teh repository are distributed as follows:
- DAG-Files: Directory containing python files pertaiing to each task in the airflow workflow designed for this project
  - data.py: for fetching and merging data files to form the final dataset we used for model training.
  - KNN.py, MLP(1).py, NaiveNayes.py, RF.py, SVC.py, logReg.py, xGB.py: python files containing train and test scripts to be run in the airflow pipeline.
  - : for collating the performance metrics obtained from each model task in the airflow.
  - fraud.py: airflow script that calls all of the above during airflow execution.
 
- Dashboards 


Included in this repository is the Jupyter notebook eda_and_models.ipynb, where we conducted exploratory data analysis (EDA) to gain valuable insights into the dataset and better understand the problem at hand. We referenced and adapted techniques from a Kaggle notebook  https://www.kaggle.com/code/artgor/eda-and-models#Data-Exploration that provided a comprehensive overview, aiding us in obtaining a superficial understanding of the data and framing our approach.
