# Home Credit Default Risk
This is a group project for ISSS610 Applied Machine Learning.
## Notebook Instructions 
The Programs folder contains 8 notebooks, which comprise of 4 phases of the projects:
### 0 Data processing: 
- Filename: [0.DataIngestion.ipynb](https://github.com/WideSu/ISSS610_AML_Group_Project/blob/main/0.DataIngestion.ipynb)
- Input files: `application_train.csv` and other tables on Kaggle website
- Output file: `data_v1.csv`
- Functions:
    - Merged 7 tables
    
### 1 EDA
- Filename: [1.EDA.ipynb](https://github.com/WideSu/ISSS610_AML_Group_Project/blob/main/1.EDA.ipynb)
- Input file: `data_v1.csv`
- Output file: None
- Functions:
    - Plotted histograms, Boxplots, Correlation plots for the merged data

### 2 Data Preprocessing
- Filename: [2.DataProcessing.ipynb](https://github.com/WideSu/ISSS610_AML_Group_Project/blob/main/2.DataProcessing.ipynb)
- Input file: `data_v1.csv`
- Output files: [test_data.csv](https://github.com/WideSu/ISSS610_AML_Group_Project/blob/main/test_data.csv), [train_data.csv](https://github.com/WideSu/ISSS610_AML_Group_Project/blob/main/train_data.csv)
- Functions:
    - Handled missing values
    - Standarized and normalized numerical features
    - Conducted one-hot encoding for catergorical features
    - Stratified train test splitting

### 3. Modeling
- Filename:
    - [3.Modeling_DeepFM.ipynb](https://github.com/WideSu/ISSS610_AML_Group_Project/blob/main/3.Modeling_DeepFM.ipynb)
    - [3.Modeling_LGBM.ipynb](https://github.com/WideSu/ISSS610_AML_Group_Project/blob/main/3.Modeling_LGBM.ipynb)
    - [3.Modeling_RandomForest.ipynb](https://github.com/WideSu/ISSS610_AML_Group_Project/blob/main/3.Modeling_RandomForest.ipynb)
    - [3.Modling_LogisticRegression.ipynb](https://github.com/WideSu/ISSS610_AML_Group_Project/blob/main/3.Modling_LogisticRegression.ipynb)
- Input files: [test_data.csv](https://github.com/WideSu/ISSS610_AML_Group_Project/blob/main/test_data.csv), [train_data.csv](https://github.com/WideSu/ISSS610_AML_Group_Project/blob/main/train_data.csv)
- Output file: None
- Function:
    - Train and evaluate the models

## Data
The whole dataset contains 8 tables as shown in Fig 1, application train and test data are the main tables that contain the target, the information about the loan and information about the loan applicant at the application time. The other 6 tables contain information about the credit history of the applicants. In practice, the occurrence of default is far less than normal repayment. We found that among 300 thousand training data, where label ‘1’ indicates default, only 8% of it has positive labels. We also found that some tables, such as the bureau balance table, have too many rows, because one previous credit record may have many associated rows in those tables, and one applicant may have multiple credit records. Hence, we need to do some pre-processing to aggregate the rows in these tables.
<img width="358" alt="image" src="https://user-images.githubusercontent.com/44923423/161380920-99cc7c11-6c7e-4fe6-b654-b448f5df799d.png">

## Evaluation Metrics
Dealing with highly imbalanced data (1:10) as shown in Fig 2, we need to choose our evaluation metrics carefully. We select recall rate and AUC score as our evaluation metrics for our models. We chose to prioritise recall over precision mainly based on our domain knowledge: The problem with a low recall is that the company would incur a lot of loss associated with bad debts and that agents will spend much time and effort trying to get the payment from the applicant. While the problem of low precision is that the company will lose some customers and revenues. We believe that misclassifying a default application will lead to greater loss to the company. We also chose AUC score as we want to evaluate the overall ability of classifying positive instances from negative instances of the model. We can change the threshold to make recall score higher, but the AUC score will remain the same and by referring to it, we make sure our model isn’t going too far on the way of sacrificing precision for recall.
![image](https://user-images.githubusercontent.com/44923423/161380947-382a00a4-b116-4f8c-b033-92ac1fc0922d.png)

## Results
| |ML models|           |     |  |Deep Learning|
|-------|---------|------------------|------------|--------|-------------|
|       |ExtraTree|LogisticRegression|RandomForest|LightGBM|DeepFM       |
|recall |0.7907   |0.3116            |0.7907      |0.8296  |0.8326       |
|roc_auc|0.9836   |0.85              |0.9836      |0.9448  |0.8492       |

## Conclusion
Credit risk-related research is vital for guiding new researchers and practitioners who want to improve their credit risk management practices. For this reason, algorithms integrating various criteria and models have been developed to predict risk-based credit scores.
In our project, we used three machine learning models, namely Logistic Regression, Random Forest, LightGBM, and one deep learning model named DeepFM to do the binary classification task for credit risk assessment and achieved 0.82’s recall rate on the test data, which is good considering the highly unbalanced data and high dimension challenges we faced. And the features we selected can be used as new determinants of creditworthiness. From the resulting feature importance of our models, we found that among the most important features, the more the applicant earned, the longer the applicant have registered, the more document the applicant provides, the lesser likely will applicant default the repayment.
However, there’re some limitation for our work. Firstly, we didn’t utilise the ability of model fully. Now we conducted the same feature engineering for all the models, such as standardization and normalization for numerical data and one-hot encoding for categorical data. But, some models like DeepFM, LightGBM can handle categorical data. Handling categorical data using the natural support of those model may lead to better performance. 

There’re other topics in credit risk assessment which we can explore in the future. According to the findings of the most studied topics in credit risk-related research are credit risk score. Moreover, we discovered that the most common objective of the papers related to credit scoring was the suggestion of new techniques, like utilising models in other area in this field. Further, the regulations renewed to overcome the risk management difficulties in the digital era have increased the relevant research output in the field. 
![image](https://user-images.githubusercontent.com/44923423/161380966-cebb7b68-26c6-42a2-9e0c-fc83bd415906.png)
