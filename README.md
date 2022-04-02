# Home Credit Default Risk
This is a group project for ISSS610 Applied Machine Learning.
## Notebook Instructions 
The Programs folder contains 8 notebooks, which comprise of 4 phases of the projects:
### 0 Data processing: 
- Filename: `0.DataIngestion.ipynb`
- Input file: `application_train.csv` and other tables on Kaggle website
- Output file: `data_v1.csv`
- Functions:
    - Merged 7 tables
    
### 1 EDA
- Filename: `1.EDA.ipynb`
- Input file: `data_v1.csv`
- Output file: None
- Functions:
    - Plotted histograms, Boxplots, Correlation plots for the merged data

### 2 Data Preprocessing
- Filename: `2.DataProcessing.ipynb`
- Input file: `data_v1.csv`
- Output file: `test_data.csv`, `train_data.csv`
- Functions:
    - Handled missing values
    - Standarized and normalized numerical features
    - Conducted one-hot encoding for catergorical features
    - Stratified train test splitting

### 2 Data Preprocessing
- Filename: `2.DataProcessing.ipynb`
- Input file: `data_v1.csv`
- Output file: `test_data.csv`, `train_data.csv`
- Functions:
    - Handled missing values
    - Standarized and normalized numerical features
    - Conducted one-hot encoding for catergorical features
    - Stratified train test splitting
### 3. Modeling
- Filename:
    - `3.Modeling_DeepFM.ipynb`
    - `3.Modeling_LGBM.ipynb`
    - `3.Modeling_RandomForest.ipynb`
    - `3.Modling_LogisticRegression.ipynb`
- Input file: `test_data.csv`, `train_data.csv`
- Output file: None
- Function:
    - Train and evaluate the models


[Data Description](https://www.kaggle.com/c/home-credit-default-risk/overview)
![image](https://user-images.githubusercontent.com/44923423/150918954-1c6df444-bb94-4b2e-b7cb-1180540578a7.png)


## Results

| |ML models|           |     |  |Deep Learning|
|-------|---------|------------------|------------|--------|-------------|
|       |ExtraTree|LogisticRegression|RandomForest|LightGBM|DeepFM       |
|recall |0.7907   |0.3116            |0.7907      |0.8296  |0.8326       |
|roc_auc|0.9836   |0.85              |0.9836      |0.9448  |0.8492       |

