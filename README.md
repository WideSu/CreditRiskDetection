# ISSS610_AML_Group_Project
The group project for ISSS610 Applied Machine Learning. After discussion, Xuezhe's idea win the most votes👏👏👏, so we decide to do the Home Credit Group project.
## Home Credit Group(2019)-classification/regression
### Problem statement:
Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

### What's Home Credit Group?

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

[Data Description](https://www.kaggle.com/c/home-credit-default-risk/overview)
![image](https://user-images.githubusercontent.com/44923423/150918954-1c6df444-bb94-4b2e-b7cb-1180540578a7.png)

### Task assignment
 - Data Visualization: Yu Di, Rao Ningzhen, Ren Xuezhe
 - Data ingestion and feature engineering: Ding Yanmu, Huang Anni
 - Models: 5 ppl, each one will choose a model
### Data overview
Comes from a Kaggle competition: Home Credit Default Risk
- Rows: 307512 rows
- Columns: 346 columns (much more than 16 numerical columns)
6 csv files
- Tables:
 - application_{train|test}.csv: This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
 - bureau.csv: All client’s previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
 - bureau_balance.csv: Monthly balances of previous credits in Credit Bureau.
 - POS_CASH_balance.csv: Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
![image](https://user-images.githubusercontent.com/44923423/156871738-3a4c5799-75c9-4c2a-9f72-42250a858665.png)

### Timeline:
- Feb 15(week 6): Due for [project proposal](https://docs.google.com/presentation/d/1UhU1AEJEKgL3x9Jgf06br34fP8WF7JeT4YM4FYSVr9s/edit?usp=sharing)
- March 5(week 8): Finish our models
