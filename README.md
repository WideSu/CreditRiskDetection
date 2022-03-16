# ISSS610_AML_Group_Project
I and Yanmu are in charge of this part
# Feature selection
In general, there're 62 features, and (26 numerical features, and 36 categorical features)
- Deleted features without target value
- Deleted features without description
- Deleted the features whose missing value is above 10%
- Delete the numerical features whose variance is below $10^(-8)$

```python
cat_col = ['FLAG_DOCUMENT_14',
 'FLAG_DOCUMENT_2',
 'FLAG_PHONE',
 'FLAG_DOCUMENT_9',
 'FLAG_DOCUMENT_11',
 'FLAG_DOCUMENT_20',
 'FLAG_DOCUMENT_19',
 'WEEKDAY_APPR_PROCESS_START',
 'FLAG_DOCUMENT_16',
 'FLAG_DOCUMENT_12',
 'FLAG_MOBIL',
 'FLAG_DOCUMENT_7',
 'FLAG_DOCUMENT_10',
 'NAME_INCOME_TYPE',
 'NAME_CONTRACT_TYPE',
 'FLAG_DOCUMENT_17',
 'NAME_HOUSING_TYPE',
 'ORGANIZATION_TYPE',
 'FLAG_DOCUMENT_3',
 'FLAG_DOCUMENT_21',
 'FLAG_WORK_PHONE',
 'FLAG_OWN_CAR',
 'FLAG_CONT_MOBILE',
 'FLAG_DOCUMENT_8',
 'FLAG_EMP_PHONE',
 'FLAG_DOCUMENT_4',
 'FLAG_DOCUMENT_18',
 'NAME_TYPE_SUITE',
 'FLAG_DOCUMENT_15',
 'NAME_EDUCATION_TYPE',
 'FLAG_DOCUMENT_6',
 'FLAG_OWN_REALTY',
 'FLAG_DOCUMENT_5',
 'NAME_FAMILY_STATUS',
 'FLAG_DOCUMENT_13',
 'FLAG_EMAIL']
num_col = ['AMT_INCOME_TOTAL',
 'AMT_CREDIT',
 'AMT_GOODS_PRICE',
 'REGION_POPULATION_RELATIVE',
 'DAYS_BIRTH',
 'DAYS_REGISTRATION',
 'DAYS_ID_PUBLISH',
 'CNT_FAM_MEMBERS',
 'REGION_RATING_CLIENT',
 'REGION_RATING_CLIENT_W_CITY',
 'HOUR_APPR_PROCESS_START',
 'EXT_SOURCE_2',
 'OBS_30_CNT_SOCIAL_CIRCLE',
 'DEF_30_CNT_SOCIAL_CIRCLE',
 'OBS_60_CNT_SOCIAL_CIRCLE',
 'DEF_60_CNT_SOCIAL_CIRCLE',
 'DAYS_LAST_PHONE_CHANGE']
 ```
# Data Description
## Numerical data

|FIELD1|Column                     |Desc                                                                                                              |
|------|---------------------------|------------------------------------------------------------------------------------------------------------------|
|0     |REGION_RATING_CLIENT       |Our rating of the region where client lives (1,2,3)                                                               |
|1     |AMT_INCOME_TOTAL           |Income of the client                                                                                              |
|2     |DEF_60_CNT_SOCIAL_CIRCLE   |How many observation of client's social surroundings defaulted on 60 (days past due) DPD                          |
|3     |DEF_30_CNT_SOCIAL_CIRCLE   |How many observation of client's social surroundings defaulted on 30 DPD (days past due)                          |
|4     |OBS_30_CNT_SOCIAL_CIRCLE   |How many observation of client's social surroundings with observable 30 DPD (days past due) default               |
|5     |EXT_SOURCE_2               |Normalized score from external data source                                                                        |
|6     |OBS_60_CNT_SOCIAL_CIRCLE   |How many observation of client's social surroundings with observable 60 DPD (days past due) default               |
|7     |AMT_CREDIT                 |Credit amount of the loan                                                                                         |
|8     |DAYS_REGISTRATION          |How many days before the application did client change his registration                                           |
|9     |REG_CITY_NOT_LIVE_CITY     |Flag if client's permanent address does not match contact address (1=different, 0=same, at city level)            |
|10    |DAYS_ID_PUBLISH            |How many days before the application did client change the identity document with which he applied for the loan   |
|11    |REG_CITY_NOT_WORK_CITY     |Flag if client's permanent address does not match work address (1=different, 0=same, at city level)               |
|12    |REG_REGION_NOT_WORK_REGION |Flag if client's permanent address does not match work address (1=different, 0=same, at region level)             |
|13    |SK_ID_CURR                 |ID of loan in our sample                                                                                          |
|14    |LIVE_REGION_NOT_WORK_REGION|Flag if client's contact address does not match work address (1=different, 0=same, at region level)               |
|15    |CODE_GENDER                |Gender of the client                                                                                              |
|16    |REG_REGION_NOT_LIVE_REGION |Flag if client's permanent address does not match contact address (1=different, 0=same, at region level)          |
|17    |CNT_CHILDREN               |Number of children the client has                                                                                 |
|18    |AMT_GOODS_PRICE            |For consumer loans it is the price of the goods for which the loan is given                                       |
|19    |CNT_FAM_MEMBERS            |How many family members does client have                                                                          |
|20    |REGION_RATING_CLIENT_W_CITY|Our rating of the region where client lives with taking city into account (1,2,3)                                 |
|21    |REGION_POPULATION_RELATIVE |Normalized population of region where client lives (higher number means the client lives in more populated region)|
|22    |DAYS_LAST_PHONE_CHANGE     |How many days before application did client change phone                                                          |
|23    |LIVE_CITY_NOT_WORK_CITY    |Flag if client's contact address does not match work address (1=different, 0=same, at city level)                 |
|24    |HOUR_APPR_PROCESS_START    |Approximately at what hour did the client apply for the loan                                                      |
|25    |DAYS_BIRTH                 |Client's age in days at the time of application                                                                   |


## Catergorical data

|FIELD1|Column                    |Desc                                                                           |
|------|--------------------------|-------------------------------------------------------------------------------|
|0     |FLAG_DOCUMENT_14          |Did client provide document 14                                                 |
|1     |FLAG_DOCUMENT_2           |Did client provide document 2                                                  |
|2     |FLAG_PHONE                |Did client provide home phone (1=YES, 0=NO)                                    |
|3     |FLAG_DOCUMENT_9           |Did client provide document 9                                                  |
|4     |FLAG_DOCUMENT_11          |Did client provide document 11                                                 |
|5     |FLAG_DOCUMENT_20          |Did client provide document 20                                                 |
|6     |FLAG_DOCUMENT_19          |Did client provide document 19                                                 |
|7     |WEEKDAY_APPR_PROCESS_START|On which day of the week did the client apply for the loan                     |
|8     |FLAG_DOCUMENT_16          |Did client provide document 16                                                 |
|9     |FLAG_DOCUMENT_12          |Did client provide document 12                                                 |
|10    |FLAG_MOBIL                |Did client provide mobile phone (1=YES, 0=NO)                                  |
|11    |FLAG_DOCUMENT_7           |Did client provide document 7                                                  |
|12    |FLAG_DOCUMENT_10          |Did client provide document 10                                                 |
|13    |NAME_INCOME_TYPE          |Clients income type (businessman, working, maternity leave)                    |
|14    |NAME_CONTRACT_TYPE        |Identification if loan is cash or revolving                                    |
|15    |FLAG_DOCUMENT_17          |Did client provide document 17                                                 |
|16    |NAME_HOUSING_TYPE         |What is the housing situation of the client (renting, living with parents, ...)|
|17    |ORGANIZATION_TYPE         |Type of organization where client works                                        |
|18    |FLAG_DOCUMENT_3           |Did client provide document 3                                                  |
|19    |FLAG_DOCUMENT_21          |Did client provide document 21                                                 |
|20    |FLAG_WORK_PHONE           |Did client provide home phone (1=YES, 0=NO)                                    |
|21    |FLAG_OWN_CAR              |Flag if the client owns a car                                                  |
|22    |FLAG_CONT_MOBILE          |Was mobile phone reachable (1=YES, 0=NO)                                       |
|23    |FLAG_DOCUMENT_8           |Did client provide document 8                                                  |
|24    |FLAG_EMP_PHONE            |Did client provide work phone (1=YES, 0=NO)                                    |
|25    |FLAG_DOCUMENT_4           |Did client provide document 4                                                  |
|26    |FLAG_DOCUMENT_18          |Did client provide document 18                                                 |
|27    |NAME_TYPE_SUITE           |Who was accompanying client when he was applying for the loan                  |
|28    |FLAG_DOCUMENT_15          |Did client provide document 15                                                 |
|29    |NAME_EDUCATION_TYPE       |Level of highest education the client achieved                                 |
|30    |FLAG_DOCUMENT_6           |Did client provide document 6                                                  |
|31    |FLAG_OWN_REALTY           |Flag if client owns a house or flat                                            |
|32    |FLAG_DOCUMENT_5           |Did client provide document 5                                                  |
|33    |NAME_FAMILY_STATUS        |Family status of the client                                                    |
|34    |FLAG_DOCUMENT_13          |Did client provide document 13                                                 |
|35    |FLAG_EMAIL                |Did client provide email (1=YES, 0=NO)                                         |

# Normalize
# Standarize
# One-hot encoder

'''
cat_col = ['FLAG_DOCUMENT_14_0.0',
 'FLAG_DOCUMENT_14_1.0',
 'FLAG_DOCUMENT_2_0.0',
 'FLAG_PHONE_0.0',
 'FLAG_PHONE_1.0',
 'FLAG_DOCUMENT_9_0.0',
 'FLAG_DOCUMENT_9_1.0',
 'FLAG_DOCUMENT_11_0.0',
 'FLAG_DOCUMENT_11_1.0',
 'FLAG_DOCUMENT_20_0.0',
 'FLAG_DOCUMENT_20_1.0',
 'FLAG_DOCUMENT_19_0.0',
 'FLAG_DOCUMENT_19_1.0',
 'WEEKDAY_APPR_PROCESS_START_FRIDAY',
 'WEEKDAY_APPR_PROCESS_START_MONDAY',
 'WEEKDAY_APPR_PROCESS_START_SATURDAY',
 'WEEKDAY_APPR_PROCESS_START_SUNDAY',
 'WEEKDAY_APPR_PROCESS_START_THURSDAY',
 'WEEKDAY_APPR_PROCESS_START_TUESDAY',
 'WEEKDAY_APPR_PROCESS_START_WEDNESDAY',
 'FLAG_DOCUMENT_16_0.0',
 'FLAG_DOCUMENT_16_1.0',
 'FLAG_DOCUMENT_12_0.0',
 'FLAG_MOBIL_1.0',
 'FLAG_DOCUMENT_7_0.0',
 'FLAG_DOCUMENT_10_0.0',
 'NAME_INCOME_TYPE_Commercial associate',
 'NAME_INCOME_TYPE_Pensioner',
 'NAME_INCOME_TYPE_State servant',
 'NAME_INCOME_TYPE_Working',
 'NAME_CONTRACT_TYPE_Cash loans',
 'NAME_CONTRACT_TYPE_Revolving loans',
 'FLAG_DOCUMENT_17_0.0',
 'FLAG_DOCUMENT_17_1.0',
 'NAME_HOUSING_TYPE_Co-op apartment',
 'NAME_HOUSING_TYPE_House / apartment',
 'NAME_HOUSING_TYPE_Municipal apartment',
 'NAME_HOUSING_TYPE_Office apartment',
 'NAME_HOUSING_TYPE_Rented apartment',
 'NAME_HOUSING_TYPE_With parents',
 'ORGANIZATION_TYPE_Advertising',
 'ORGANIZATION_TYPE_Agriculture',
 'ORGANIZATION_TYPE_Bank',
 'ORGANIZATION_TYPE_Business Entity Type 1',
 'ORGANIZATION_TYPE_Business Entity Type 2',
 'ORGANIZATION_TYPE_Business Entity Type 3',
 'ORGANIZATION_TYPE_Cleaning',
 'ORGANIZATION_TYPE_Construction',
 'ORGANIZATION_TYPE_Culture',
 'ORGANIZATION_TYPE_Electricity',
 'ORGANIZATION_TYPE_Emergency',
 'ORGANIZATION_TYPE_Government',
 'ORGANIZATION_TYPE_Hotel',
 'ORGANIZATION_TYPE_Housing',
 'ORGANIZATION_TYPE_Industry: type 1',
 'ORGANIZATION_TYPE_Industry: type 10',
 'ORGANIZATION_TYPE_Industry: type 11',
 'ORGANIZATION_TYPE_Industry: type 12',
 'ORGANIZATION_TYPE_Industry: type 13',
 'ORGANIZATION_TYPE_Industry: type 2',
 'ORGANIZATION_TYPE_Industry: type 3',
 'ORGANIZATION_TYPE_Industry: type 4',
 'ORGANIZATION_TYPE_Industry: type 5',
 'ORGANIZATION_TYPE_Industry: type 7',
 'ORGANIZATION_TYPE_Industry: type 8',
 'ORGANIZATION_TYPE_Industry: type 9',
 'ORGANIZATION_TYPE_Insurance',
 'ORGANIZATION_TYPE_Kindergarten',
 'ORGANIZATION_TYPE_Legal Services',
 'ORGANIZATION_TYPE_Medicine',
 'ORGANIZATION_TYPE_Military',
 'ORGANIZATION_TYPE_Mobile',
 'ORGANIZATION_TYPE_Other',
 'ORGANIZATION_TYPE_Police',
 'ORGANIZATION_TYPE_Postal',
 'ORGANIZATION_TYPE_Realtor',
 'ORGANIZATION_TYPE_Religion',
 'ORGANIZATION_TYPE_Restaurant',
 'ORGANIZATION_TYPE_School',
 'ORGANIZATION_TYPE_Security',
 'ORGANIZATION_TYPE_Security Ministries',
 'ORGANIZATION_TYPE_Self-employed',
 'ORGANIZATION_TYPE_Services',
 'ORGANIZATION_TYPE_Telecom',
 'ORGANIZATION_TYPE_Trade: type 1',
 'ORGANIZATION_TYPE_Trade: type 2',
 'ORGANIZATION_TYPE_Trade: type 3',
 'ORGANIZATION_TYPE_Trade: type 4',
 'ORGANIZATION_TYPE_Trade: type 5',
 'ORGANIZATION_TYPE_Trade: type 6',
 'ORGANIZATION_TYPE_Trade: type 7',
 'ORGANIZATION_TYPE_Transport: type 1',
 'ORGANIZATION_TYPE_Transport: type 2',
 'ORGANIZATION_TYPE_Transport: type 3',
 'ORGANIZATION_TYPE_Transport: type 4',
 'ORGANIZATION_TYPE_University',
 'ORGANIZATION_TYPE_XNA',
 'FLAG_DOCUMENT_3_0.0',
 'FLAG_DOCUMENT_3_1.0',
 'FLAG_DOCUMENT_21_0.0',
 'FLAG_DOCUMENT_21_1.0',
 'FLAG_WORK_PHONE_0.0',
 'FLAG_WORK_PHONE_1.0',
 'FLAG_OWN_CAR_0',
 'FLAG_OWN_CAR_1',
 'FLAG_CONT_MOBILE_0.0',
 'FLAG_CONT_MOBILE_1.0',
 'FLAG_DOCUMENT_8_0.0',
 'FLAG_DOCUMENT_8_1.0',
 'FLAG_EMP_PHONE_0.0',
 'FLAG_EMP_PHONE_1.0',
 'FLAG_DOCUMENT_4_0.0',
 'FLAG_DOCUMENT_18_0.0',
 'FLAG_DOCUMENT_18_1.0',
 'NAME_TYPE_SUITE_Children',
 'NAME_TYPE_SUITE_Family',
 'NAME_TYPE_SUITE_Group of people',
 'NAME_TYPE_SUITE_Other_A',
 'NAME_TYPE_SUITE_Other_B',
 'NAME_TYPE_SUITE_Spouse, partner',
 'NAME_TYPE_SUITE_Unaccompanied',
 'FLAG_DOCUMENT_15_0.0',
 'FLAG_DOCUMENT_15_1.0',
 'NAME_EDUCATION_TYPE_Academic degree',
 'NAME_EDUCATION_TYPE_Higher education',
 'NAME_EDUCATION_TYPE_Incomplete higher',
 'NAME_EDUCATION_TYPE_Lower secondary',
 'NAME_EDUCATION_TYPE_Secondary / secondary special',
 'FLAG_DOCUMENT_6_0.0',
 'FLAG_DOCUMENT_6_1.0',
 'FLAG_OWN_REALTY_0',
 'FLAG_OWN_REALTY_1',
 'FLAG_DOCUMENT_5_0.0',
 'FLAG_DOCUMENT_5_1.0',
 'NAME_FAMILY_STATUS_Civil marriage',
 'NAME_FAMILY_STATUS_Married',
 'NAME_FAMILY_STATUS_Separated',
 'NAME_FAMILY_STATUS_Single / not married',
 'NAME_FAMILY_STATUS_Widow',
 'FLAG_DOCUMENT_13_0.0',
 'FLAG_DOCUMENT_13_1.0',
 'FLAG_EMAIL_0.0',
 'FLAG_EMAIL_1.0']
'''

# Select features
- Select by models
 -All kinds of classifiers

- Select by KBest(Haven't done)
# Try some ML Models to see the result
- Light GBM(TO-DO)
- Gradient Boosting Machine
- Random Forest

'''
RandomForest classifer achieved mean: 0.7898 recall (std: +-2.31%)
RandomForest classifer achieved mean: 0.9831 roc_auc (std: +-0.17%)
'''

-Logistic Regression

'''
LogisticRegression classifer achieved mean: 0.07814 recall (std: +-2.50%)
LogisticRegression classifer achieved mean: 0.7853 roc_auc (std: +-0.78%)
'''

-ExtraTree

'''
ExtraTree classifer achieved mean: 0.7898 recall (std: +-2.31%)<br>
ExtraTree classifer achieved mean: 0.9845 roc_auc (std: +-0.39%)
'''
## The results for tree-based classifiers are good, why?
1.	Random forest is bagging method, which has low variance high bias. So, it’s robust to out-liners in data, which works well in our scenario. In predictive maintenance, the parameter logs for machines with alarms have more out-liners than normal since not all alarms can be predicted. Some machines don’t work well for environment change or manual interruptions.
2.	Random forest selects different features in each sub decision tree, so it ovoid selecting the same features all the time, which lead to robustness to out-liners  in features. Not all the changes in features are significantly different before alarms. For example, ‘actual_position_r_traveler’ is randomly scattered for alarm and alarm data.

