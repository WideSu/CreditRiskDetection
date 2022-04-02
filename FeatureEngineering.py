# ! pip install category_encoders
# ! pip install lightgbm
import logging, sys
logging.disable(sys.maxsize)
# Basic
import numpy as np
import pandas as pd
from scipy import stats
import numpy as np
import warnings
from collections import defaultdict
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# Plot Learning Rate
from sklearn.model_selection import learning_curve
# Processing
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, Normalizer, MinMaxScaler, StandardScaler, \
    QuantileTransformer, PowerTransformer, normalize,  OneHotEncoder
from sklearn.impute import SimpleImputer # Handle missing values
# Feature engineering
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold,SelectFromModel, RFE, SelectKBest, f_classif, mutual_info_classif, SelectFpr


#Feature ranking with recursive feature elimination([RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html))
from sklearn.feature_selection import RFE
# Models
# import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier as xgbc

# Evaluation
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, recall_score
from sklearn.model_selection import KFold, StratifiedKFold
warnings.simplefilter(action='ignore', category=FutureWarning)

# Feature selected
feature_dict = {'RandomForest': ['REGION_RATING_CLIENT',
  'DEF_30_CNT_SOCIAL_CIRCLE',
  'REG_CITY_NOT_WORK_CITY',
  'CODE_GENDER',
  'DAYS_BIRTH',
  'SK_ID_CURR',
  'CNT_FAM_MEMBERS',
  'AMT_INCOME_TOTAL',
  'OBS_60_CNT_SOCIAL_CIRCLE',
  'AMT_CREDIT',
  'HOUR_APPR_PROCESS_START',
  'OBS_30_CNT_SOCIAL_CIRCLE',
  'AMT_GOODS_PRICE',
  'DAYS_ID_PUBLISH',
  'EXT_SOURCE_2',
  'REGION_POPULATION_RELATIVE',
  'CNT_CHILDREN',
  'LIVE_CITY_NOT_WORK_CITY',
  'DAYS_LAST_PHONE_CHANGE',
  'DAYS_REGISTRATION'],
 'ExtraTree': ['DEF_30_CNT_SOCIAL_CIRCLE',
  'REG_CITY_NOT_WORK_CITY',
  'CODE_GENDER',
  'DAYS_BIRTH',
  'SK_ID_CURR',
  'CNT_FAM_MEMBERS',
  'AMT_INCOME_TOTAL',
  'OBS_60_CNT_SOCIAL_CIRCLE',
  'AMT_CREDIT',
  'HOUR_APPR_PROCESS_START',
  'OBS_30_CNT_SOCIAL_CIRCLE',
  'AMT_GOODS_PRICE',
  'DAYS_ID_PUBLISH',
  'EXT_SOURCE_2',
  'REGION_POPULATION_RELATIVE',
  'CNT_CHILDREN',
  'DAYS_LAST_PHONE_CHANGE',
  'DAYS_REGISTRATION',
  'WEEKDAY_APPR_PROCESS_START_FRIDAY_1_1',
  'ORGANIZATION_TYPE_Other_0_0'],
 'LogisticRegression': ['DEF_30_CNT_SOCIAL_CIRCLE',
  'CODE_GENDER',
  'SK_ID_CURR',
  'EXT_SOURCE_2',
  'CNT_CHILDREN',
  'DAYS_REGISTRATION',
  'FLAG_DOCUMENT_11_0.0_0_0',
  'FLAG_DOCUMENT_11_0.0_0_1',
  'FLAG_DOCUMENT_11_0.0_1_0',
  'FLAG_DOCUMENT_11_0.0_1_1',
  'FLAG_DOCUMENT_11_1.0_0_0',
  'FLAG_DOCUMENT_11_1.0_0_1',
  'FLAG_DOCUMENT_11_1.0_1_0',
  'FLAG_DOCUMENT_11_1.0_1_1',
  'ORGANIZATION_TYPE_Advertising_0_0',
  'ORGANIZATION_TYPE_Advertising_0_1',
  'ORGANIZATION_TYPE_Advertising_1_0',
  'ORGANIZATION_TYPE_Advertising_1_1',
  'ORGANIZATION_TYPE_Hotel_0_0',
  'ORGANIZATION_TYPE_Hotel_1_1']}

# After joining, there're many duplicated columns
def remove_duplicated_cols(data, feature_list):
    before_removal_df = data.copy(deep=True)
    duplicated_col_dict = defaultdict(list)
    for col in data.columns:
        if col not in feature_list:
            if col[-1].isalpha() and col[-2] == '_':
                duplicated_col_dict[col[:-2]].append(col)
    for col_name, col_list in duplicated_col_dict.items():
        data = data.drop(columns=col_list[1:])
        data = data.rename(columns={col_list[0]: col_name})
    print('Removed {} duplicated columns'.format(len(before_removal_df.columns) - len(data.columns)))
    print('Keeped {} columns for each orginal column'.format(len(duplicated_col_dict)))
    return data
def show_description(features, desc_df):
    desc_list = []
    for col in features:
        if len(desc_df[desc_df.Row == col].Description.values) > 0:
            # print(data_desc[data_desc.Row == col].Description.values[0])
            desc = desc_df[desc_df.Row == col].Description.values[0]
        else:
            desc = None
        desc_list.append(desc)
    res = pd.DataFrame([features,desc_list],index=['Column','Desc']).T
    return res

# Select those columns by datatype
def select_col_by_dtype(data, datatype):
    df = data.copy(deep=True)
    columns = df.select_dtypes(datatype).columns
    print('There are {} columns whose datatype is'.format(len(columns)),datatype)
    df = df.loc[:,columns]
    return df

# Remove low-variance features whose variance lower than 1*10^(-8) - similar to constant
# This feature selection algorithm looks only at the features (X), not the desired outputs (y), and can thus be used for unsupervised learning.
def remove_lowvariance(data, variance_threshhold):
    df = data.copy(deep=True) 
    constant_filter = VarianceThreshold(threshold=variance_threshhold)
    constant_filter.fit(df)
    without_constant_features = df.columns[constant_filter.get_support()]
    constant_features = list(set(df.columns) - set(without_constant_features))
    print(len(constant_features),'constant features are removed, namely',constant_features)
    print('There are {} columns after removing constants'.format(len(without_constant_features)))
    return df.loc[:,without_constant_features]

def filter_columns_by_missing_ratio(data, missing_ratio_threshold):
    missing_ratio_series = (data.isnull().sum()/len(data)).sort_values(ascending=False)
    highly_missing_cols = missing_ratio_series[missing_ratio_series > missing_ratio_threshold].index
    return highly_missing_cols

def plot_col_num_for_dif_missing_ratio(data):
    missing_ratio_list = [i*0.1 for i in range(1,10)]
    col_num_list = []
    for missing_ratio in missing_ratio_list:
        highly_missing_cols = filter_columns_by_missing_ratio(data, missing_ratio)
        col_num = len(data.columns)-len(highly_missing_cols)
        col_num_list.append(col_num)
    missing_ratio_df = pd.DataFrame([missing_ratio_list,col_num_list],index=['MissingRatio','ColNumb']).T
    ax = missing_ratio_df.plot(x='MissingRatio',y='ColNumb',style='o--')
    return missing_ratio_df

# Standardization, or mean removal and variance scaling
def my_standarization(standarize_type, x_vars, data):
    '''
    Standarize dataframe

    Input:
        standarize_type: string
            For linear:
            "RobustScaler": Robust for outlinears
            "MinMaxScaler": Transform features by scaling each feature to a given range. Default is [0,1]
            "StandardScaler": mean=0, std=1
            For non-linear:
            "QuantileTransformer": Change into uniform distribution
            "PowerTransformer": Change into gausian distribution
        x_vars: string: the column name list for the feartures that you want to normalize
        y: string: the column name of y
        data: DataFrame
    Output: Dataframe
        The standarized dataframe
    '''
    x_vars = list(set(x_vars) & set(data.columns))
    standarized_data = data.copy(deep=True)
    print('Data shape', data.shape)
    for x_var in x_vars:
        X = data.loc[:,x_var].values.reshape(-1, 1)
        # print(x_var)
        # print('X shape',X.shape)
        if standarize_type == "RobustScaler":
            scaler = RobustScaler()
        if standarize_type == "MinMaxScaler":
            scaler = MinMaxScaler()
        if standarize_type == "StandardScaler":
            scaler = StandardScaler()
        if standarize_type == "QuantileTransformer":
            scaler = QuantileTransformer()
        if standarize_type == "PowerTransformer":
            scaler = QuantileTransformer()
        standarized_data[x_var] = scaler.fit_transform(X)
    print("Finished Standarization")
    #     print("-"*20)
    #     print(standarized_data.describe())
    return standarized_data


def my_normalization(norm_type, x_vars, data):
    '''
    Normalize dataframe

    Input:
        norm_type: string
            "l1": divide the mean
            "l2": divide the std
            "max": divide the max
        x_vars: string: the column name list for the feartures that you want to normalize
        y_var: string: the column name of y
        data: DataFrame
    Output: Dataframe
        The normalized dataframe
    '''
    x_vars = list(set(x_vars) & set(data.columns))
    normalized_data = data.copy(deep=True)
    X = data.loc[:,x_vars]
    X_normalized = Normalizer(norm=norm_type).fit_transform(X)
    normalized_data.loc[:,x_vars] = X_normalized
    print("Finished Normalization")
    #     print("-"*20)
    #     print(normalized_data.describe())
    return normalized_data
def my_read_csv(data_dir):
    data = pd.read_csv(data_dir)
    if 'Unnamed: 0' in data.columns:
        data.drop(columns='Unnamed: 0',inplace=True)
    if 'index' in data.columns:
        data.drop(columns='index',inplace=True)
    return data

def select_features(data,x_vars,y_var,feature_num):
    '''
    feature_num = 20
    feature_dict = select_features(data, x_vars, y_var, feature_num)   
    '''
    X, y = data.loc[:,x_vars],data.loc[:,y_var]
    # X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2)
    random_state = 0
    res_dict = {}
    cl_model_dict = {}
    cl_model_list = ["RandomForest","ExtraTree",'LogisticRegression'] # "LightGBM"
    # select_k_best_param_list = ["f_classif","mutual_info_classif", "SelectFpr"]
    for cl_model_name in cl_model_list:
        print("Choose by ",cl_model_name)
        if cl_model_name == "RandomForest":
            classifier = RandomForestClassifier(random_state=random_state)
        if cl_model_name == "ExtraTree":
            classifier = ExtraTreesClassifier(n_estimators=100,
                             random_state=random_state, n_jobs=-1)
        if cl_model_name == "LogisticRegression":
            classifier = LogisticRegression(random_state=random_state)
        # if cl_model_name == 'LightGBM':
        '''
        write your code here
        '''
        #     classifier
        classifier.fit(X, y)
        cl_model_dict.update({cl_model_name:classifier})
        sfm = SelectFromModel(classifier, max_features=feature_num)
        sfm.fit(X, y)
        feature_list=list(X.columns[sfm.get_support()])
        res_dict.update({cl_model_name:feature_list})
    return res_dict
def make_classification(model_name,x_vars,y_var,train_data, test_data):
    '''
    Example usage:
        make_classification(model_name='RandomForest', x_vars = feature_dict['RandomForest'], y_var='TARGET', train_data = train_data, test_data = test_data)
    '''
    random_state = 0
    data = pd.concat([train_data,test_data])
    x_vars = list(set(data.columns) & set(x_vars))
    # print('x_vars',len(x_vars))

    X_train, X_test, y_train, y_test = train_data.loc[:,x_vars], test_data.loc[:,x_vars], train_data.loc[:,y_var],test_data.loc[:,y_var]
    if model_name == 'RandomForest':
        classifier = RandomForestClassifier(random_state=random_state)
    if model_name == 'LogisticRegression':
        classifier = LogisticRegression(random_state=random_state)
    if model_name == 'ExtraTree':
        classifier = ExtraTreesClassifier(n_estimators=100,
                             random_state=random_state, n_jobs=-1)
    '''
    write your code here
    '''
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict_proba(X_test)[:,1]
    y_pred_lable = classifier.predict(X_test)
    roc_score = roc_auc_score(y_test, y_pred)
    r_score = recall_score(y_test, y_pred_lable)
    print('{} classifer achieved recall {:.4}'.format(model_name,r_score))
    print('{} classifer achieved roc_auc {:.4}'.format(model_name,roc_score))

   
    
