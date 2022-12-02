from flask import Flask, request, render_template, session
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

app= Flask(__name__)

df_qs= pd.read_csv('data_qs2.csv') # quetionere data

df_ad= pd.read_csv('data_ad.csv') # Kaggle data

qs_columns = ['Timestamp', 'Age', 'Male', 'Country',
       'Area_Income', 'Daily_Internet_Usage',
       'click_new',
       'click_per_day',
       'Daily_Time_on_Site',
       'cyber_security',
       'Clicked',
       'prefered_headline',
       'others',
       'specific_product']

df_qs.columns = qs_columns

df_ad.columns = ['Daily_Time_on_Site', 'Age', 'Area_Income',
       'Daily_Internet_Usage', 'Ad_Topic', 'City', 'Male', 'Country',
       'Timestamp', 'Clicked']


df_qs.drop(['others', 'specific_product'],axis = 1, inplace = True)

df_qs.dropna(inplace = True )

def drop_trash(df,col,trash):
  for i in trash:
    df2 = df[df[col].apply(lambda x: i not in x)]
  return df2

df_qs = drop_trash(df_qs,'Country', ['Option 21'])
df_qs = drop_trash(df_qs,'Male', [';'])
df_qs = drop_trash(df_qs,'Age', [';'])
df_qs = drop_trash(df_qs,'Area_Income', [';'])
df_qs = drop_trash(df_qs,'Daily_Internet_Usage', [';'])
df_qs = drop_trash(df_qs,'Clicked', [';'])
df_qs = drop_trash(df_qs,'Daily_Time_on_Site', [';'])
df_qs = drop_trash(df_qs,'click_new', [';'])


df_qs['Age'] = (df_qs['Age']
                .map({'35-44': 40, '25-34': 30, '45-54': 50, 
                      '16-24': 20, '55 and above': 60}))


df_qs['Clicked'] = df_qs['Clicked'] .map({'Yes': 1, 'No': 0})


df_qs['Male'] = df_qs['Male'] .map({'Male': 1, 'Female': 0})

df_qs['Daily_Time_on_Site'] = (df_qs['Daily_Time_on_Site']
                .map({'10 - 30 minutes': 20, '0 - 10 minutes':5, '11 - 30 minutes':20, 
                      '31 - 60 minutes': 46, '61- 120 minutes': 90,
                      '121 - 180 minutes': 150, '121- 180 minutes': 150,
                      '31  -60 minutes': 46}))


df_qs['Daily_Internet_Usage'] = (df_qs['Daily_Internet_Usage']
                .map({'181 minutes and above': 240, '121- 180 minutes':150, '180 minutes and above':240, 
                      '31- 120 minutes': 76, '10 - 30 minutes': 20,
                      '31 - 60 minutes': 46, '61- 120 minutes': 90,
                      '0 - 9 minutes': 5}))



df_qs['Area_Income'] = (df_qs['Area_Income']
                .map({'Less than 100,000': 4500, 'Above 1,000, 000':90000, '101,000 - 300,000':12030, 
                      '501,000- 1000,000': 45030, '301,000 - 500,000': 24030}))


df_ad.drop(['Ad_Topic', 'City', 'Country', 'Timestamp'], axis = 1, inplace = True )

df_qs.drop(['click_per_day', 'click_new', 'Country', 'Timestamp', 'cyber_security', 'prefered_headline'], axis = 1, inplace = True )


df_qs.dropna(inplace = True )

# Balancing the questioneer dataset

from sklearn.utils import resample

#create two different dataframe of majority and minority class 
df_majority = df_qs[(df_qs['Clicked']==0)] 
df_minority = df_qs[(df_qs['Clicked']==1)] 

# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 256, # to match majority class
                                 random_state=42)  # reproducible results
# Combine majority class with upsampled minority class
df_unsampled = pd.concat([df_minority_upsampled, df_majority])

df_qs = df_unsampled

df_merged = pd.concat([df_qs, df_ad], ignore_index=True)

# Classification Modeling

# Split dataset
X, y = df_merged.iloc[:, :-1], df_merged.iloc[:, -1]

# Create train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model=RandomForestClassifier(n_estimators=300)




# First we need to know which columns are binary, nominal and numerical
def get_columns_by_category():
    categorical_mask = X.select_dtypes(
        include=['object']).apply(pd.Series.nunique) == 2
    numerical_mask = X.select_dtypes(
        include=['int64', 'float64']).apply(pd.Series.nunique) > 5

    binary_columns = X[categorical_mask.index[categorical_mask]].columns
    nominal_columns = X[categorical_mask.index[~categorical_mask]].columns
    numerical_columns = X[numerical_mask.index[numerical_mask]].columns

    return binary_columns, nominal_columns, numerical_columns

binary_columns, nominal_columns, numerical_columns = get_columns_by_category()

# Now we can create a column transformer pipeline

transformers = [('binary', OrdinalEncoder(), binary_columns),
                ('nominal', OneHotEncoder(), nominal_columns),
                ('numerical', StandardScaler(), numerical_columns)]

transformer_pipeline = ColumnTransformer(transformers, remainder='passthrough')

pipe = Pipeline([('transformer', transformer_pipeline), ('Random Forest Classifier', model)])
pipe.fit(X_train, y_train)
  


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/form')
def main1():
    return render_template('form.html')

@app.route('/predict', methods= ['POST'])
def index():
    age= int(request.form['age'])
    gender= request.form['gender']
    income= float(request.form['income'])
    usage= float(request.form['usage'])
    time= float(request.form['time'])

    arr = pd.DataFrame((np.array([[age,gender,income,usage,time]])
        ), columns=X_train.columns)    
    pred= pipe.predict(arr).tolist()[0]

    return render_template('after.html', data=pred)
    


if __name__ == '__main__':
    app.run(debug= True, use_reloader=False)
