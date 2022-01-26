# ISSS610_AML_Group_Project
The group project for ISSS610 Applied Machine Learning
# 1. Anni HUANG
## 1.1 Online News Popularity(2015)-classification/regression
### Problem Statement:

Media and newspaper companies want their passages to go viral. But they're so many factors that can contribute to it. Like the date and time when it posts the keywords, the positive and negative words they used, the category of the news, and so on. We propose the study to analyze the factors which lead to the popularity of news. And that will help the editors get more attention from the public.
<br>
### Data Description
This dataset summarizes a heterogeneous set of features about articles published by Mashable in a period of two years. The goal is to predict the number of shares in social networks (popularity).<br>
[Data description](https://archive-beta.ics.uci.edu/ml/datasets/online+news+popularity)
### Proposed Models
Four machine learning models
- (a) Linear Regression
- (b) Logistic Regression
- (c) Ridge Regression
- (d) Lasso Regression

One deep learning model
- (e) CNN

### Evaluation Matrix
Four performance metrics 
- Root mean squared error, 
- Coefficient of Variance, 
- Mean Absolute Error,
- Median Absolute Error

will be used on our models


```python
import pandas as pd
df11 = pd.read_csv("OnlineNewsPopularity.csv")
df11.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>timedelta</th>
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>...</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>http://mashable.com/2013/01/07/amazon-instant-...</td>
      <td>731.0</td>
      <td>12.0</td>
      <td>219.0</td>
      <td>0.663594</td>
      <td>1.0</td>
      <td>0.815385</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.100000</td>
      <td>0.7</td>
      <td>-0.350000</td>
      <td>-0.600</td>
      <td>-0.200000</td>
      <td>0.500000</td>
      <td>-0.187500</td>
      <td>0.000000</td>
      <td>0.187500</td>
      <td>593</td>
    </tr>
    <tr>
      <th>1</th>
      <td>http://mashable.com/2013/01/07/ap-samsung-spon...</td>
      <td>731.0</td>
      <td>9.0</td>
      <td>255.0</td>
      <td>0.604743</td>
      <td>1.0</td>
      <td>0.791946</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.033333</td>
      <td>0.7</td>
      <td>-0.118750</td>
      <td>-0.125</td>
      <td>-0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>711</td>
    </tr>
    <tr>
      <th>2</th>
      <td>http://mashable.com/2013/01/07/apple-40-billio...</td>
      <td>731.0</td>
      <td>9.0</td>
      <td>211.0</td>
      <td>0.575130</td>
      <td>1.0</td>
      <td>0.663866</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.100000</td>
      <td>1.0</td>
      <td>-0.466667</td>
      <td>-0.800</td>
      <td>-0.133333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>1500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>http://mashable.com/2013/01/07/astronaut-notre...</td>
      <td>731.0</td>
      <td>9.0</td>
      <td>531.0</td>
      <td>0.503788</td>
      <td>1.0</td>
      <td>0.665635</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.136364</td>
      <td>0.8</td>
      <td>-0.369697</td>
      <td>-0.600</td>
      <td>-0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>1200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>http://mashable.com/2013/01/07/att-u-verse-apps/</td>
      <td>731.0</td>
      <td>13.0</td>
      <td>1072.0</td>
      <td>0.415646</td>
      <td>1.0</td>
      <td>0.540890</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>0.033333</td>
      <td>1.0</td>
      <td>-0.220192</td>
      <td>-0.500</td>
      <td>-0.050000</td>
      <td>0.454545</td>
      <td>0.136364</td>
      <td>0.045455</td>
      <td>0.136364</td>
      <td>505</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>




```python
df11.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 39644 entries, 0 to 39643
    Data columns (total 61 columns):
     #   Column                          Non-Null Count  Dtype  
    ---  ------                          --------------  -----  
     0   url                             39644 non-null  object 
     1    timedelta                      39644 non-null  float64
     2    n_tokens_title                 39644 non-null  float64
     3    n_tokens_content               39644 non-null  float64
     4    n_unique_tokens                39644 non-null  float64
     5    n_non_stop_words               39644 non-null  float64
     6    n_non_stop_unique_tokens       39644 non-null  float64
     7    num_hrefs                      39644 non-null  float64
     8    num_self_hrefs                 39644 non-null  float64
     9    num_imgs                       39644 non-null  float64
     10   num_videos                     39644 non-null  float64
     11   average_token_length           39644 non-null  float64
     12   num_keywords                   39644 non-null  float64
     13   data_channel_is_lifestyle      39644 non-null  float64
     14   data_channel_is_entertainment  39644 non-null  float64
     15   data_channel_is_bus            39644 non-null  float64
     16   data_channel_is_socmed         39644 non-null  float64
     17   data_channel_is_tech           39644 non-null  float64
     18   data_channel_is_world          39644 non-null  float64
     19   kw_min_min                     39644 non-null  float64
     20   kw_max_min                     39644 non-null  float64
     21   kw_avg_min                     39644 non-null  float64
     22   kw_min_max                     39644 non-null  float64
     23   kw_max_max                     39644 non-null  float64
     24   kw_avg_max                     39644 non-null  float64
     25   kw_min_avg                     39644 non-null  float64
     26   kw_max_avg                     39644 non-null  float64
     27   kw_avg_avg                     39644 non-null  float64
     28   self_reference_min_shares      39644 non-null  float64
     29   self_reference_max_shares      39644 non-null  float64
     30   self_reference_avg_sharess     39644 non-null  float64
     31   weekday_is_monday              39644 non-null  float64
     32   weekday_is_tuesday             39644 non-null  float64
     33   weekday_is_wednesday           39644 non-null  float64
     34   weekday_is_thursday            39644 non-null  float64
     35   weekday_is_friday              39644 non-null  float64
     36   weekday_is_saturday            39644 non-null  float64
     37   weekday_is_sunday              39644 non-null  float64
     38   is_weekend                     39644 non-null  float64
     39   LDA_00                         39644 non-null  float64
     40   LDA_01                         39644 non-null  float64
     41   LDA_02                         39644 non-null  float64
     42   LDA_03                         39644 non-null  float64
     43   LDA_04                         39644 non-null  float64
     44   global_subjectivity            39644 non-null  float64
     45   global_sentiment_polarity      39644 non-null  float64
     46   global_rate_positive_words     39644 non-null  float64
     47   global_rate_negative_words     39644 non-null  float64
     48   rate_positive_words            39644 non-null  float64
     49   rate_negative_words            39644 non-null  float64
     50   avg_positive_polarity          39644 non-null  float64
     51   min_positive_polarity          39644 non-null  float64
     52   max_positive_polarity          39644 non-null  float64
     53   avg_negative_polarity          39644 non-null  float64
     54   min_negative_polarity          39644 non-null  float64
     55   max_negative_polarity          39644 non-null  float64
     56   title_subjectivity             39644 non-null  float64
     57   title_sentiment_polarity       39644 non-null  float64
     58   abs_title_subjectivity         39644 non-null  float64
     59   abs_title_sentiment_polarity   39644 non-null  float64
     60   shares                         39644 non-null  int64  
    dtypes: float64(59), int64(1), object(1)
    memory usage: 18.5+ MB


## 1.2 Trip duration(2020)-regression
### Problem Statement
Trip duration is the most fundamental measure in all modes of transportation. Hence, it is crucial to predict the trip-time precisely for the advancement of Intelligent Transport Systems (ITS) and traveller information systems. In order to predict the trip duration, data mining techniques are employed to predict the trip duration of rental bikes in Seoul Bike sharing system. The prediction is carried out with the combination of Seoul Bike data and weather data.
### Data Description
The Data used include trip duration, trip distance, pickup-dropoff latitude and longitude, temperature, precipitation, wind speed, humidity, solar radiation, snowfall, ground temperature and 1-hour average dust concentration. Feature engineering is done to extract additional features from the data. 
### Proposed Models
Four statistical models are used to predict the trip duration. 
- (a) Linear regression, 
- (b) Gradient boosting machines, 
- (c) k nearest neighbor and 
- (d) Random Forest(RF). 

One Deep Learning model is used to predict the trip duration.
- (e) CNN

### Evaluation Metrix
Four performance metrics 
- Root mean squared error, 
- Coefficient of Variance, 
- Mean Absolute Error,
- Median Absolute Error

is used to determine the efficiency of the models. In comparison with the other models, the optimum model RF can explain the variance of 93% in the testing set and 98% (R2) in the training set. The outcome proves that RF is effective to be employed for the prediction of trip duration.
<br>
[Dataset description](https://plu.mx/plum/a/?mendeley_data_id=gtfh9z865f&theme=plum-bigben-theme)


```python
df12 = pd.read_csv("For_modeling.csv")
```


```python
df12.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Duration</th>
      <th>Distance</th>
      <th>PLong</th>
      <th>PLatd</th>
      <th>DLong</th>
      <th>DLatd</th>
      <th>Haversine</th>
      <th>Pmonth</th>
      <th>Pday</th>
      <th>...</th>
      <th>Dmin</th>
      <th>DDweek</th>
      <th>Temp</th>
      <th>Precip</th>
      <th>Wind</th>
      <th>Humid</th>
      <th>Solar</th>
      <th>Snow</th>
      <th>GroundTemp</th>
      <th>Dust</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>50</td>
      <td>37.544666</td>
      <td>126.888359</td>
      <td>37.544666</td>
      <td>126.888359</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>-3.2</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-2.2</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>24</td>
      <td>7670</td>
      <td>37.506199</td>
      <td>127.003944</td>
      <td>37.551250</td>
      <td>127.035103</td>
      <td>5.713529</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>25</td>
      <td>0</td>
      <td>-3.2</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-2.2</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8</td>
      <td>1390</td>
      <td>37.544590</td>
      <td>127.057083</td>
      <td>37.537014</td>
      <td>127.061096</td>
      <td>0.913702</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>9</td>
      <td>0</td>
      <td>-3.2</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-2.2</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8</td>
      <td>1820</td>
      <td>37.571102</td>
      <td>127.023560</td>
      <td>37.561447</td>
      <td>127.034920</td>
      <td>1.468027</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>10</td>
      <td>0</td>
      <td>-3.2</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-2.2</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>850</td>
      <td>37.573242</td>
      <td>127.015907</td>
      <td>37.565849</td>
      <td>127.016403</td>
      <td>0.823227</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>-3.2</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-2.2</td>
      <td>25.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
df12.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9601139 entries, 0 to 9601138
    Data columns (total 26 columns):
     #   Column      Dtype  
    ---  ------      -----  
     0   Unnamed: 0  int64  
     1   Duration    int64  
     2   Distance    int64  
     3   PLong       float64
     4   PLatd       float64
     5   DLong       float64
     6   DLatd       float64
     7   Haversine   float64
     8   Pmonth      int64  
     9   Pday        int64  
     10  Phour       int64  
     11  Pmin        int64  
     12  PDweek      int64  
     13  Dmonth      int64  
     14  Dday        int64  
     15  Dhour       int64  
     16  Dmin        int64  
     17  DDweek      int64  
     18  Temp        float64
     19  Precip      float64
     20  Wind        float64
     21  Humid       float64
     22  Solar       float64
     23  Snow        float64
     24  GroundTemp  float64
     25  Dust        float64
    dtypes: float64(13), int64(13)
    memory usage: 1.9 GB


# 2 Rao Ningzhen
## 2.1 Backorder Prediction(2021) - regression/classification
### Backorder Defination:
A backorder generally indicates that customer demand for a product or service exceeds a company’s capacity to supply it. Product backorder may be the result of strong sales performance (e.g. the product is in such high demand that production cannot keep up with sales). However, backorders can upset consumers, lead to canceled orders and decreased customer loyalty. Companies want to avoid backorders, but also avoid overstocking every product (leading to higher inventory costs). Machine learning can identify patterns related to backorders before customers order. Production can then adjust to minimize delays while customer service can provide accurate dates to keep customers informed and happy. The predictive analytics approach enables the maximum product to get in the hands of customers at the lowest cost to the organization.
A backorder is an order (or part of an order) waiting to be filled, usually because the merchant in question does not have that item currently stocked in the warehouse.
### Problem Statement:
In supply chain system, Material backorder is a common problem, impacting an inventory system service level and effectiveness. Identifying parts with the highest chances of shortage prior its occurrence can present a high opportunity to improve an overall company’s performance. In this project, we will train classifiers to predict future backordered products and generate predictions for a test set.<br>
[Data Description](https://www.kaggle.com/chandanareddy12/back-order-prediction)
### Data Fields
- sku - sku code
- nationalinv - Current inventory level of component
- leadtime - Transit time
- in transit qty - Quantity in transit
- forecastxmonth - Forecast sales for the net 3, 6, 9 months
- salesxmonth - Sales quantity for the prior 1, 3, 6, 9 months
- minbank - Minimum recommended amount in stock • potentialissue - Indictor variable noting potential issue with item
- pieces past due - Parts overdue from source
- perfxmonthsavg - Source performance in the last 6 and 12 months 
- localboqty - Amount of stock orders overdue 
- X17-X22 - General Risk Flags 
- wentonbackorder - Product went on backorder
### Proposed Models
Four statistical models are used to predict the trip duration. 
- (a) Linear regression, 
- (b) Gradient boosting machines, 
- (c) k nearest neighbor and 
- (d) Random Forest(RF). 
One Deep Learning model is used to predict the trip duration.
- (e) CNN
### Evaluation Metrix
Four performance metrics 
- Root mean squared error, 
- Coefficient of Variance, 
- Mean Absolute Error,
- Median Absolute Error

is used to determine the efficiency of the models.



```python
df21 = pd.read_csv("Back_order_train.csv")
df21.head()
```

    /opt/anaconda3/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3145: DtypeWarning: Columns (0) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sku</th>
      <th>national_inv</th>
      <th>lead_time</th>
      <th>in_transit_qty</th>
      <th>forecast_3_month</th>
      <th>forecast_6_month</th>
      <th>forecast_9_month</th>
      <th>sales_1_month</th>
      <th>sales_3_month</th>
      <th>sales_6_month</th>
      <th>...</th>
      <th>pieces_past_due</th>
      <th>perf_6_month_avg</th>
      <th>perf_12_month_avg</th>
      <th>local_bo_qty</th>
      <th>deck_risk</th>
      <th>oe_constraint</th>
      <th>ppap_risk</th>
      <th>stop_auto_buy</th>
      <th>rev_stop</th>
      <th>went_on_backorder</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1026827</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>-99.00</td>
      <td>-99.00</td>
      <td>0.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1043384</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1043696</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>-99.00</td>
      <td>-99.00</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1043852</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.10</td>
      <td>0.13</td>
      <td>0.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1044048</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>-99.00</td>
      <td>-99.00</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
df21.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1687861 entries, 0 to 1687860
    Data columns (total 23 columns):
     #   Column             Non-Null Count    Dtype  
    ---  ------             --------------    -----  
     0   sku                1687861 non-null  object 
     1   national_inv       1687860 non-null  float64
     2   lead_time          1586967 non-null  float64
     3   in_transit_qty     1687860 non-null  float64
     4   forecast_3_month   1687860 non-null  float64
     5   forecast_6_month   1687860 non-null  float64
     6   forecast_9_month   1687860 non-null  float64
     7   sales_1_month      1687860 non-null  float64
     8   sales_3_month      1687860 non-null  float64
     9   sales_6_month      1687860 non-null  float64
     10  sales_9_month      1687860 non-null  float64
     11  min_bank           1687860 non-null  float64
     12  potential_issue    1687860 non-null  object 
     13  pieces_past_due    1687860 non-null  float64
     14  perf_6_month_avg   1687860 non-null  float64
     15  perf_12_month_avg  1687860 non-null  float64
     16  local_bo_qty       1687860 non-null  float64
     17  deck_risk          1687860 non-null  object 
     18  oe_constraint      1687860 non-null  object 
     19  ppap_risk          1687860 non-null  object 
     20  stop_auto_buy      1687860 non-null  object 
     21  rev_stop           1687860 non-null  object 
     22  went_on_backorder  1687860 non-null  object 
    dtypes: float64(15), object(8)
    memory usage: 296.2+ MB


# 3 Ren Xuezhe
## 3.1 Home Credit Group(2019)-classification/regression
### Problem statement:
Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

### Home Credit Group

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

[Data Description](https://www.kaggle.com/c/home-credit-default-risk/overview)
![image](https://user-images.githubusercontent.com/44923423/150918954-1c6df444-bb94-4b2e-b7cb-1180540578a7.png)



# 4 Ding Yanmu
Image Classification

# 5 Yu Di
### 5.1 Yelp Challenge(2021)-Regression
 - Task: regression
 - Dataset Link: https://www.yelp.com/dataset
 - Kaggle Challenge: https://www.kaggle.com/yelp-dataset/yelp-dataset?select=yelp_academic_dataset_business.json
 - Dataset Name : Yelp
 - Models: lightGBM,DRF,NeuMF,BPRMF, deepconn
 
### Context
This dataset is a subset of Yelp's businesses, reviews, and user data. It was originally put together for the Yelp Dataset Challenge which is a chance for students to conduct research or analysis on Yelp's data and share their discoveries. In the most recent dataset you'll find information about businesses across 8 metropolitan areas in the USA and Canada.

### Content
This dataset contains five JSON files and the user agreement.
More information about those files can be found here.
