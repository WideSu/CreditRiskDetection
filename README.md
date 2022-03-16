# ISSS610_AML_Group_Project
I and Yanmu are in charge of this part
# Feature selection
In general, there're 79 features, and (17 numerical features, and 62 categorical features)
- Deleted features without target value
- Deleted features without description
- Deleted the features whose missing value is above 10%
- Delete the numerical features whose variance is below 10^(-8)

```python
cat_col = ['NAME_CONTRACT_TYPE','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',\
 'OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE',\
 'EMERGENCYSTATE_MODE','CREDIT_ACTIVE','CREDIT_CURRENCY','CREDIT_TYPE','FLAG_LAST_APPL_PER_CONTRACT','NAME_CASH_LOAN_PURPOSE',\
 'NAME_CONTRACT_STATUS','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE','CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY',\
 'NAME_PORTFOLIO','NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION',\
 'NAME_CONTRACT_STATUS','NAME_CONTRACT_STATUS','FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE',\
 'FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6',\
 'FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',\
 'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20',\
 'FLAG_DOCUMENT_21','NFLAG_LAST_APPL_IN_DAY','NFLAG_INSURED_ON_APPROVAL']
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

|    | Column                          | Desc                                                                                                               |
| -- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 0  | AMT\_INCOME\_TOTAL              | Income of the client                                                                                               |
| 1  | AMT\_CREDIT                     | Credit amount of the loan                                                                                          |
| 2  | AMT\_GOODS\_PRICE               | For consumer loans it is the price of the goods for which the loan is given                                        |
| 3  | REGION\_POPULATION\_RELATIVE    | Normalized population of region where client lives (higher number means the client lives in more populated region) |
| 4  | DAYS\_BIRTH                     | Client's age in days at the time of application                                                                    |
| 5  | DAYS\_REGISTRATION              | How many days before the application did client change his registration                                            |
| 6  | DAYS\_ID\_PUBLISH               | How many days before the application did client change the identity document with which he applied for the loan    |
| 7  | CNT\_FAM\_MEMBERS               | How many family members does client have                                                                           |
| 8  | REGION\_RATING\_CLIENT          | Our rating of the region where client lives (1,2,3)                                                                |
| 9  | REGION\_RATING\_CLIENT\_W\_CITY | Our rating of the region where client lives with taking city into account (1,2,3)                                  |
| 10 | HOUR\_APPR\_PROCESS\_START      | Approximately at what hour did the client apply for the loan                                                       |
| 11 | EXT\_SOURCE\_2                  | Normalized score from external data source                                                                         |
| 12 | OBS\_30\_CNT\_SOCIAL\_CIRCLE    | How many observation of client's social surroundings with observable 30 DPD (days past due) default                |
| 13 | DEF\_30\_CNT\_SOCIAL\_CIRCLE    | How many observation of client's social surroundings defaulted on 30 DPD (days past due)                           |
| 14 | OBS\_60\_CNT\_SOCIAL\_CIRCLE    | How many observation of client's social surroundings with observable 60 DPD (days past due) default                |
| 15 | DEF\_60\_CNT\_SOCIAL\_CIRCLE    | How many observation of client's social surroundings defaulted on 60 (days past due) DPD                           |
| 16 | DAYS\_LAST\_PHONE\_CHANGE       | How many days before application did client change phone                                                           |

## Catergorical data

|    | Column                          | Desc                                                                                                                                                                                                                                                                                    |
| -- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0  | NAME\_CONTRACT\_TYPE            | Identification if loan is cash or revolving                                                                                                                                                                                                                                             |
| 1  | NAME\_TYPE\_SUITE               | Who was accompanying client when he was applying for the loan                                                                                                                                                                                                                           |
| 2  | NAME\_INCOME\_TYPE              | Clients income type (businessman, working, maternity leave)                                                                                                                                                                                                                             |
| 3  | NAME\_EDUCATION\_TYPE           | Level of highest education the client achieved                                                                                                                                                                                                                                          |
| 4  | NAME\_FAMILY\_STATUS            | Family status of the client                                                                                                                                                                                                                                                             |
| 5  | NAME\_HOUSING\_TYPE             | What is the housing situation of the client (renting, living with parents, ...)                                                                                                                                                                                                         |
| 6  | OCCUPATION\_TYPE                | What kind of occupation does the client have                                                                                                                                                                                                                                            |
| 7  | WEEKDAY\_APPR\_PROCESS\_START   | On which day of the week did the client apply for the loan                                                                                                                                                                                                                              |
| 8  | ORGANIZATION\_TYPE              | Type of organization where client works                                                                                                                                                                                                                                                 |
| 9  | FONDKAPREMONT\_MODE             | Normalized information about building where the client lives, What is average (\_AVG suffix), modus (\_MODE suffix), median (\_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| 10 | HOUSETYPE\_MODE                 | Normalized information about building where the client lives, What is average (\_AVG suffix), modus (\_MODE suffix), median (\_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| 11 | WALLSMATERIAL\_MODE             | Normalized information about building where the client lives, What is average (\_AVG suffix), modus (\_MODE suffix), median (\_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| 12 | EMERGENCYSTATE\_MODE            | Normalized information about building where the client lives, What is average (\_AVG suffix), modus (\_MODE suffix), median (\_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| 13 | CREDIT\_ACTIVE                  | Status of the Credit Bureau (CB) reported credits                                                                                                                                                                                                                                       |
| 14 | CREDIT\_CURRENCY                | Recoded currency of the Credit Bureau credit                                                                                                                                                                                                                                            |
| 15 | CREDIT\_TYPE                    | Type of Credit Bureau credit (Car, cash,...)                                                                                                                                                                                                                                            |
| 16 | FLAG\_LAST\_APPL\_PER\_CONTRACT | Flag if it was last application for the previous contract. Sometimes by mistake of client or our clerk there could be more applications for one single contract                                                                                                                         |
| 17 | NAME\_CASH\_LOAN\_PURPOSE       | Purpose of the cash loan                                                                                                                                                                                                                                                                |
| 18 | NAME\_CONTRACT\_STATUS          | Contract status during the month                                                                                                                                                                                                                                                        |
| 19 | NAME\_CONTRACT\_STATUS          | Contract status during the month                                                                                                                                                                                                                                                        |
| 20 | NAME\_PAYMENT\_TYPE             | Payment method that client chose to pay for the previous application                                                                                                                                                                                                                    |
| 21 | CODE\_REJECT\_REASON            | Why was the previous application rejected                                                                                                                                                                                                                                               |
| 22 | NAME\_CLIENT\_TYPE              | Was the client old or new client when applying for the previous application                                                                                                                                                                                                             |
| 23 | NAME\_GOODS\_CATEGORY           | What kind of goods did the client apply for in the previous application                                                                                                                                                                                                                 |
| 24 | NAME\_PORTFOLIO                 | Was the previous application for CASH, POS, CAR                                                                                                                                                                                                                                         |
| 25 | NAME\_PRODUCT\_TYPE             | Was the previous application x-sell o walk-in                                                                                                                                                                                                                                           |
| 26 | CHANNEL\_TYPE                   | Through which channel we acquired the client on the previous application                                                                                                                                                                                                                |
| 27 | NAME\_SELLER\_INDUSTRY          | The industry of the seller                                                                                                                                                                                                                                                              |
| 28 | NAME\_YIELD\_GROUP              | Grouped interest rate into small medium and high of the previous application                                                                                                                                                                                                            |
| 29 | PRODUCT\_COMBINATION            | Detailed product combination of the previous application                                                                                                                                                                                                                                |
| 30 | NAME\_CONTRACT\_STATUS          | Contract status during the month                                                                                                                                                                                                                                                        |
| 31 | NAME\_CONTRACT\_STATUS          | Contract status during the month                                                                                                                                                                                                                                                        |
| 32 | FLAG\_OWN\_CAR                  | Flag if the client owns a car                                                                                                                                                                                                                                                           |
| 33 | FLAG\_OWN\_REALTY               | Flag if client owns a house or flat                                                                                                                                                                                                                                                     |
| 34 | FLAG\_MOBIL                     | Did client provide mobile phone (1=YES, 0=NO)                                                                                                                                                                                                                                           |
| 35 | FLAG\_EMP\_PHONE                | Did client provide work phone (1=YES, 0=NO)                                                                                                                                                                                                                                             |
| 36 | FLAG\_WORK\_PHONE               | Did client provide home phone (1=YES, 0=NO)                                                                                                                                                                                                                                             |
| 37 | FLAG\_CONT\_MOBILE              | Was mobile phone reachable (1=YES, 0=NO)                                                                                                                                                                                                                                                |
| 38 | FLAG\_PHONE                     | Did client provide home phone (1=YES, 0=NO)                                                                                                                                                                                                                                             |
| 39 | FLAG\_EMAIL                     | Did client provide email (1=YES, 0=NO)                                                                                                                                                                                                                                                  |
| 40 | FLAG\_DOCUMENT\_2               | Did client provide document 2                                                                                                                                                                                                                                                           |
| 41 | FLAG\_DOCUMENT\_3               | Did client provide document 3                                                                                                                                                                                                                                                           |
| 42 | FLAG\_DOCUMENT\_4               | Did client provide document 4                                                                                                                                                                                                                                                           |
| 43 | FLAG\_DOCUMENT\_5               | Did client provide document 5                                                                                                                                                                                                                                                           |
| 44 | FLAG\_DOCUMENT\_6               | Did client provide document 6                                                                                                                                                                                                                                                           |
| 45 | FLAG\_DOCUMENT\_7               | Did client provide document 7                                                                                                                                                                                                                                                           |
| 46 | FLAG\_DOCUMENT\_8               | Did client provide document 8                                                                                                                                                                                                                                                           |
| 47 | FLAG\_DOCUMENT\_9               | Did client provide document 9                                                                                                                                                                                                                                                           |
| 48 | FLAG\_DOCUMENT\_10              | Did client provide document 10                                                                                                                                                                                                                                                          |
| 49 | FLAG\_DOCUMENT\_11              | Did client provide document 11                                                                                                                                                                                                                                                          |
| 50 | FLAG\_DOCUMENT\_12              | Did client provide document 12                                                                                                                                                                                                                                                          |
| 51 | FLAG\_DOCUMENT\_13              | Did client provide document 13                                                                                                                                                                                                                                                          |
| 52 | FLAG\_DOCUMENT\_14              | Did client provide document 14                                                                                                                                                                                                                                                          |
| 53 | FLAG\_DOCUMENT\_15              | Did client provide document 15                                                                                                                                                                                                                                                          |
| 54 | FLAG\_DOCUMENT\_16              | Did client provide document 16                                                                                                                                                                                                                                                          |
| 55 | FLAG\_DOCUMENT\_17              | Did client provide document 17                                                                                                                                                                                                                                                          |
| 56 | FLAG\_DOCUMENT\_18              | Did client provide document 18                                                                                                                                                                                                                                                          |
| 57 | FLAG\_DOCUMENT\_19              | Did client provide document 19                                                                                                                                                                                                                                                          |
| 58 | FLAG\_DOCUMENT\_20              | Did client provide document 20                                                                                                                                                                                                                                                          |
| 59 | FLAG\_DOCUMENT\_21              | Did client provide document 21                                                                                                                                                                                                                                                          |
| 60 | NFLAG\_LAST\_APPL\_IN\_DAY      | Flag if the application was the last application per day of the client. Sometimes clients apply for more applications a day. Rarely it could also be error in our system that one application is in the database twice                                                                  |
| 61 | NFLAG\_INSURED\_ON\_APPROVAL    | Did the client requested insurance during the previous application                                                                                                                                                                                                                      |
# Normalize
# Standarize
# Select features
- Select by models
- Select by KBest
# Try some ML Models to see the result
- Logistic regression
- Light GBM
- Gradient Boosting Machine
