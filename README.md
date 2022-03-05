# ISSS610_AML_Group_Project
The group project for ISSS610 Applied Machine Learning. After discussion, Xuezhe's idea win the most votes👏👏👏, so we decide to do the Home Credit Group project.
# Feature selection
- Delete the features whose missing value is above 30%
- Delete the features whose variance is below 0.01
```python
['NAME_CONTRACT_TYPE_x', 'NAME_TYPE_SUITE_x', 'NAME_INCOME_TYPE',
       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
       'WEEKDAY_APPR_PROCESS_START_x', 'ORGANIZATION_TYPE',
       'NAME_CONTRACT_STATUS', 'CODE_GENDER', 'FLAG_OWN_CAR',
       'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
       'FLAG_PHONE', 'FLAG_EMAIL', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_5',
       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8', 'REGION_RATING_CLIENT',
       'AMT_REQ_CREDIT_BUREAU_WEEK']
 ```
# Normalize
# Standarize
# Select features
- Select by models
- Select by KBest
