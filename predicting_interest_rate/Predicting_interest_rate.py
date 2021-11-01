#!/usr/bin/env python
# coding: utf-8

# # Prediction of interest rate from lending club data

# For full code please visit https://github.com/pranaymallipudi/stoutcases/tree/main/predicting_interest_rate

# In[1]:


import os
import pandas as pd
import numpy as np
#from pandas_profiling import ProfileReport
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px # for interactive plotting
import plotly.graph_objects as go # for interactive plotting
from scipy.stats import norm
#plt.style.use('default')
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


# ### Read the raw data

# - The dataset has 10,000 observations and 55 variables

# In[2]:


raw = pd.read_csv('loans_full_schema.csv')


# In[3]:


raw.head()


# ### Below variables seems to generate after the event (i.e finalizing interest rate) and might not be available when running the model in production systems. So it's bettter to drop them because in reality they won't be available to model

# - installment
# - loan_status
# - initial_listing_status
# - disbursement_method
# - balance
# - paid_total
# - paid_principal
# - paid_interest
# - paid_late_fees

# In[4]:


cols_to_drop = ['installment','loan_status','initial_listing_status','disbursement_method',                'balance','paid_total','paid_principal','paid_interest','paid_late_fees']
raw.drop(cols_to_drop,axis=1,inplace=True)


# ### Column issue month just has the data for 3 months. So we can't use this to detect seasonality etc. So we can drop this variable as well

# In[5]:


raw.issue_month.value_counts()


# In[6]:


raw.drop('issue_month',axis=1,inplace=True)


# ### Let's see which state  is mostly represented in our data

# - As the number of states are more, we can use interactive plotly graph with a slider for ease

# In[7]:


fig = px.bar(x = raw.state.value_counts().index,
             y = raw.state.value_counts().values,
        labels= {'x': 'State (Slide the slider to move around)','y':'Number of customers by state'},
        width = 800, height= 500)

fig.update_xaxes(rangeslider_visible = True)
#fig.update_traces(marker_color='red')
fig.update_xaxes(type='category')
fig.show()


# - Based on above graph, California has the highest representation followed by Texas, New york, Florida and Illinois

# ### Let's see how employee length column is distributed

# In[8]:


sns.barplot(x=raw.emp_length.value_counts(normalize=True).index,
            y= raw.emp_length.value_counts(normalize=True).values*100,
           color = 'blue')
plt.ylabel('% of customers')
plt.xlabel('employee_length')
plt.show()


# - Since almost 35% of values are greater than or equal to 10, it's better to bucket this variable as binary type with cutoff of 10

# ### Let's see how different categories in home ownership column are distributed

# In[9]:


raw['emp_length1'] = np.where(raw['emp_length']>=10,1,0)
raw.drop('emp_length',axis=1,inplace=True)


# In[10]:


sns.barplot(x=raw.homeownership.value_counts(normalize=True).index,
            y= raw.homeownership.value_counts(normalize=True).values*100,
           color = 'blue')
plt.ylabel('% of customers')
plt.show()


# - Percentage of people who owned houses are very less and this makes sense as people with own houses are generally financially stable

# ### Now let's explore another important variable annual income

# In[11]:


sns.distplot(raw['annual_income'])
plt.show()


# - Distribution of annual income is heavily rightskewed which is expected as:
#     - People with less income mostly seek loans
#     - Finacial data is usually rightskewed
# - Let's look at density plot (violin plot) and box plots to identify any outliers in the data

# ### Violin plot and box plot for annual income indicating outilers

# In[12]:


fig, axes = plt.subplots(1, 2)
fig.suptitle('Box and density plots for annual income')
sns.violinplot(ax = axes[0],data=raw,y='annual_income')
sns.boxplot(ax = axes[1],data=raw,y='annual_income')
plt.ylabel('')
plt.show()


# - Above graph indicates outliers
# - Let's calculate Inter quartile range(IQR) to see the upper limit of reasonable values. i.e Q3+1.5IQR

# ### Calculate IQR for annual income

# In[13]:


iqr = raw.annual_income.describe()['75%']-raw.annual_income.describe()['25%']
upper_limit = raw.annual_income.describe()['75%'] + 1.5*iqr
print(f'upper limit is {upper_limit}')


# - Let's look at observations above upper limit to check for any disparity like data error/fake values
# - Filter the data in which annual income is unverified as those might indicate some disparity
# - verified_income column indicates whether income is verified or not
# 

# ### Check all observations above upper limit that are not verified to see any disparity

# - Upon examination, the data looks fine as people with more income has job titles like ceo, president, etc. which are in general high paying jobs. So we can trust this data even though it's unverified

# In[14]:


raw[(raw['annual_income']>170000.0)&   (raw['verified_income']=='Not Verified')].sort_values(by='annual_income',ascending=False).head()


# - In the whole dataset we have couple of extreme values which are above 1.5mil. Even though these values are verified it's better to remove them as few machine learning models can be very sensitive to outliers. This helps in generalizing the model better 

# In[15]:


raw[(raw['annual_income']>1.5*10**6)].sort_values(by='annual_income',ascending=False).head()


# In[16]:


#print(raw.shape)
raw = raw[(raw['annual_income']<=1.5*10**6)]
#print(raw.shape)


# ### Most of the values  in below columns are zeroes. So it's better to drop them

# 
# - plot1: Delinquencies in last 2 years
# - plot2: Number of collections in the last 12 months
# - plot3: Number of current accounts which are delinquent
# - plot4: Number of current accounts that are 120 days past due
# - plot5: Number of current accounts that are 30 days past due
# - plot6: Tax liens
# 

# In[17]:


count = 0
fig, axes = plt.subplots(1, 6,figsize=(12,5))
cols = ['delinq_2y','num_collections_last_12m','current_accounts_delinq',          'num_accounts_120d_past_due','num_accounts_30d_past_due','tax_liens']
for i in cols:
    sns.pointplot(ax = axes[count],x= ['zero','non-zero'],              y = [raw[i].isin([0]).mean()*100,100-raw[i].isin([0]).mean()*100])
    count=count+1


# In[18]:


raw.drop(cols,axis=1,inplace=True)


# ### Till now we have eliminated/corrected variables that we are very sure of by doing univariate analysis. Now it's time to look deep into data

# ### Before this let's also finish basic cleaning of dataset

# - Based on application type, whether it is single or joint, merge columns like annual income, verification status, and debt to income ratio into a single column
# - If a customer applies through joint application replace individual columns like annual income with joint annual income and so on
# - After above step drop columns related to joint application including the indicator column which indicates if the application is joint or single

# In[19]:


single_columns = ['annual_income','verified_income','debt_to_income']
joint_columns = ['annual_income_joint','verification_income_joint','debt_to_income_joint']
for i,j in zip(single_columns,joint_columns):
    raw[i] = np.where(raw['application_type']=='individual',
                     raw[i],raw[j])


# In[20]:


joint_columns.append('application_type')


# In[21]:


raw.drop(columns = joint_columns,axis=1,inplace=True)


# ### Now let's do missing value treatment

# - Below table represents percent of missing values in descending order. Columns in which values are not missing are not shown
# 

# In[22]:


raw.isnull().mean().sort_values(ascending=False)[raw.isnull().mean().sort_values(ascending=False).values>0]*100


# - Let's check number of times 0 appears in months_since_90d_late, months_since_last_delinq, months_since_last_credit_inquiry

# #### Treatment for months_since_90d_late and months_since_last_delinq

# In[23]:


for i in ['months_since_90d_late','months_since_last_delinq','months_since_last_credit_inquiry']:
    print(f'number of times 0 appears in {i} is {raw[i].isin([0]).sum()}')


# - As 0 never occured in months_since_90d_late and months_since_last_delinq columns we can safely assume 0's are represented by NA's
# - Hence replace NA's in below columns with 0's

# In[24]:


for i in ['months_since_90d_late','months_since_last_delinq']:
    raw[i].fillna(0,inplace = True)


# #### Treatment for months_since_last_credit_inquiry and verified_income

# - Months since last credit enquiry is a categorical column
# - Since it contains 0's we cannot conclude NA's as 0
# - Dropping 12% of values i.e 1200 observations would cause loss of data
# - Hence we can impute these values with mode value
# - Similary for income verification status

# In[25]:


for i in ['months_since_last_credit_inquiry','verified_income']:
    raw[i].fillna(raw[i].mode()[0],inplace = True)


# #### Treatment for emp_title

# - Employee title represents the employment type of a customer
# - almost 800 values are missing
# - Since there is no logical way to impute this, we can fill missing values with new category called 'Other'
# 

# In[26]:


raw['emp_title'].fillna('other',inplace = True)


# #### Missing value treatment complete

# In[27]:


if len(raw.isnull().mean().sort_values(ascending=False)[raw.isnull().mean().sort_values(ascending=False).values>0]*100)==0:
    print('Now there are no missing values in the dataset')
else:
    print('Now there are still missing values. Take care of them')


# ### Time to clean data types of variables

# - Check datatypes of variables. We can see categorical variables are coded as objects. 
# - Let's convert them to categorical type using astype function

# In[28]:


print(f'The data types before cleaning are {raw.dtypes.unique()}')


# In[29]:


categoricals = raw.select_dtypes(include=[object]).columns

for category in categoricals:
    try:
        raw[category] = raw[category].astype('category')
    except:
        pass


# - Now we can see all of the objects are converted to categories

# In[30]:


dtypes_new = pd.DataFrame(raw.dtypes,columns = ['types'])['types'].value_counts()[:3].index


# In[31]:


print(f'The data types after cleaning are {dtypes_new}')


# ### Let's start the deep dive into data

# - Visualize Heat map of correlations of all variables with the target variable
# - Term seems decently correlated
# - Usually more the number of months to pay, more the interest rate
# - Heatmap validates above assumption
# - Nothing seems out of place, we can move ahead with other analysis

# #### Heatmaps and feature engineering from insights

# In[32]:


cor_matrix = raw.corr()
plt.figure(figsize = (2,10))
sns.heatmap(cor_matrix[['interest_rate']],
            annot = True,
            cmap = 'Reds')
plt.show()


# - I am interested in seeing the distribution of debt_to_income as there is slight correlation

# In[33]:


from scipy.stats import norm
sns.distplot(raw['debt_to_income'], fit=norm)
plt.show()


# In[34]:


# cor_matrix = raw.corr()
# plt.figure(figsize = (50,30))
# sns.heatmap(cor_matrix,
#             annot = True,
#             cmap = 'Reds')
# plt.show()


# - Now let's visualize all the correlations using a heatmap
# - Due to large number of variables, it is tough to generate insights out of a static plot
# - Hence we used plotly to create interactive heatmap which upon hovering gives details about correlation values and the corresponding values
# - Number of open credit lines and number of satisfactory accounts are highly correlated. So it's actually prudent to create a new variable called number of satisfactory accounts per credit lines which will give us a better idea about customer

# In[35]:


fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x = cor_matrix.columns, 
        y = cor_matrix.index, 
        z = cor_matrix.values, 
        colorscale = 'reds',
        hovertemplate = 'correlation:%{z} <br> var1: %{y}  <br> var1: %{x}',
        name = '')
)

fig.update_layout(title_text = 'Heat map',
                height = 800,
                  plot_bgcolor='rgba(0,0,0,0)')
fig.show()


# In[36]:


raw['satis_pr_cred_lines'] = raw['num_satisfactory_accounts']/raw['open_credit_lines']
raw['satis_pr_cred_lines'].fillna(0,inplace=True)


# #### Scatter plots with dependent variables to validate hypothesis

# - Grade gives an estimation about customers credit history. Let's see how grade interacts with interest rate
# - Subgrade is a more granular version of Grade
# - As expected as quality of grade/subgrade decreases, interest rate increases and seems like by far the most important variable

# In[37]:


fig,axes = plt.subplots(1,2,figsize = (20,5))
#fig.figsize(10,5)
subg = raw.groupby('grade').mean()
subg1 = raw.groupby('sub_grade').mean()
#plt.figure(figsize = (10,10))
sns.scatterplot(ax= axes[0],x=subg.index, y=subg['interest_rate'])
sns.scatterplot(ax= axes[1],x=subg1.index, y=subg1['interest_rate'])
#plt.title("Subgrade vs Interest Rate ScatterPlot")
fig.suptitle('Subgrade vs Interest Rate ScatterPlot')
plt.show()


# ### Now let's prepare the data for modeling

# #### Train test split

# - Let's split the data into train and test at ratio of 80:20 using scikitlearn
#     - Use random state for reproducible results

# In[38]:


X = raw.drop('interest_rate',axis=1)
y = raw['interest_rate']


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# #### Normalizing

# - Now let's normalize our data as it makes model more effective and helps in faster convergence. Let's use sklearns standardscalar function to achieve the same
#     - Normalization is applied on numerical variables only
#     - Get the numerical column types using pandas select dtypes function
#     - We should fit the standardscalar object on train data and just transform test data without fitting it

# In[40]:


numericals = list(X_train.select_dtypes(include = 'float64').columns) + list(X_train.select_dtypes(include = 'int64').columns)


# In[41]:


sc = StandardScaler()
X_train[numericals] = sc.fit_transform(X_train[numericals])
X_test[numericals] = sc.transform(X_test[numericals])


# #### Shrinking categorical variables

# - After normalizing, let's shrink the categorical variables
#     - Few categorical variables like job title and state has lots of categories. It wouldn't make sense to consider many categories
#     - One way to shrink is to consider top 15 occured categories and replace least occured categories with "other" using a custom shrink function
#     - For example we can see how employee titles that occur less are shrinked to "other"

# In[42]:


categoricals = list(X_train.select_dtypes(include = 'category').columns)


# In[43]:


def shrink_categoricals(X_train, X_test, categoricals, top=15):

    for category in categoricals:
        if category not in X_train.columns:
            continue
        tops = X_train[category].value_counts().index[:top]
        def helper(x):
            if x in tops:
                return x
            else:
                return "other"
        X_train[category] = X_train[category].apply(helper)
        X_test[category] = X_test[category].apply(helper)
    return


# In[44]:


shrink_categoricals(X_train, X_test, categoricals)


# In[45]:


X_train['emp_title'].value_counts()


# #### Label encoding categorical variables

# - Now let's encode categorical data
#     - We can achieve this using sklearn's LabelEncoder
#     - Use fit method on train data and transform method on test data

# In[46]:


for category in categoricals:
        if category not in X_train.columns:
            continue
        le = LabelEncoder()
        X_train[category] = le.fit_transform(X_train[category])
        X_test[category] = le.transform(X_test[category])


# ### Now we have the data prepared. It's time to build intelligent models which can predict interest rate

# ### For first model let's start with simple and heuristic linear regression
# 

# - We have many variables and as already seen from heatmap few of the predictor variables are highly correlated
# - This can cause multicollinearity and might increase the variance of model leading to overfitting
# - To avoid this, let's use Ridge regression (l2 norm) which helps in regularization of model by adding α/2 times square of l2 norm of weight vectors
# - We can use 10 fold cross validation and train the model

# In[47]:


# instantiate linear regression object
ridge = RidgeCV(cv=10)

# fit or train the linear regression model on the training set and store parameters
ridge.fit(X_train, y_train)

# show the alpha parameter used in final ridgeCV model
print(f'The final α used is {ridge.alpha_}')

# show the coefficients of each variable
# ridge.coef_


# #### Now let's plot variable importances from ridge regression, which is basically the coefficients of parameters

# - Open credit lines seems inversely proportional to interest rate. More credit lines indicates that the customer has more loan accepted.This represents good behavior and might result in lower interest rate.
# - Number of satisfactory accounts and grade seems to have strong positive relation with interest rate which makes sense

# In[48]:


var_imp_ridge = dict()
for i,j in zip(X_train.columns,ridge.coef_):
    var_imp_ridge[i] = j 


# In[49]:


plt.figure(figsize = (10,5))
var_imp_ridge = {k:v for k,v in sorted(var_imp_ridge.items(), key =lambda item: item[1])}
plt.bar([x for x in var_imp_ridge.keys()], [y for y in var_imp_ridge.values()])
plt.xticks(rotation=90)
plt.show()



# #### Let's calculate RMSE on both train and test data

# - RMSE values on training and test data are close to each other and values are considerably small. We can thus conclude our model is doing fairly well.

# In[50]:


# use trained RidgeCV regression model to predict interest rates of training and test data
train_pred = ridge.predict(X_train)

ridge_test_pred = ridge.predict(X_test)

# print RMSE of training predictions
print('RMSE on training data: ', np.sqrt(mean_squared_error(y_train, train_pred)))
print('RMSE on testing data: ', np.sqrt(mean_squared_error(y_test, ridge_test_pred)))
print('R2 on training data: ', r2_score(y_train, train_pred))
print('R2 on test data: ', r2_score(y_test, ridge_test_pred))


# In[51]:


#r2_score(y_train, train_pred)


# In[52]:


#r2_score(y_test, ridge_test_pred)


# #### Learning curves

# - As an additional measure let's plot learning curves to understand how the model is performing as we add more observations. This gives us an idea about whether the model is underfitting or overfitting
# - As we can see graph, the model error converges for both test and train and the error seems to be less

# In[53]:


def plot_learning_curves(model):
    X_train1, X_val, y_train1, y_val = X_train, X_test, y_train, y_test
    train_errors, val_errors = [], []
    for m in range(1, len(X_train1)):
        model.fit(X_train1[:m], y_train1[:m])
        y_train_predict = model.predict(X_train1[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train1[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend()


# In[54]:


ridge1 = RidgeCV()
plot_learning_curves(ridge1)
plt.title('Model Learning curve')
plt.xlabel('Training set size')
plt.ylabel('RMSE')
plt.show()


# ### Now let's build a usually more effective but computationally expensive ensemble model, Random forest
# 

# - Random forests is an ensemble of decision trees and based on bagging/bootstrap aggregation
# - Decision trees when unconstrained can overfit the data
# - The process of constraining a model is called regularization
# - In random forests regarulization can be done using hyperparameter tuning
# - To select a best set of hyperparamters we can form a grid of best parameter combinations we can think of use and GridsearchCV with a cross validation of 3 to come up with best set of parameters
# - In our case, best parameters are:
#     - Estimators=80 - Number of trees
#     - max_depth = 15 - Each tree will have a maximum of 15 levels of splits
#     - Every other parameter is set to default value

# In[55]:


# %%time

# #rf = RandomForestRegressor(n_estimators=120, max_depth=25, bootstrap=True, max_features=3)
# param_grid = [{'n_estimators':[60, 70, 80, 100, 120], 'max_depth':[15, 20, 25, None], 'bootstrap':[True, False], 'max_features':[None, 2, 3]}]
# forest = RandomForestRegressor()
# grid_search = GridSearchCV(forest, param_grid, cv=3, scoring="r2")
# grid_search.fit(X_train, y_train)
# final = grid_search.best_params_
# #print(final)
# #grid_search.best_estimator_

#since we already ran gridsearchcv let's use the best params from it
grid_search = RandomForestRegressor(n_estimators=80, max_depth=15, bootstrap=True, max_features=None)
grid_search.fit(X_train, y_train)


# In[56]:


#grid_search.best_estimator_


# #### Fitting and predicting on both test and train
# - Let's fit the model on train data and evaluate on both train and test data

# In[57]:


#rf.fit(X_train, y_train)
y_pred_train_rf = grid_search.predict(X_train)
y_pred_test_rf = grid_search.predict(X_test)


# #### RMSE values
# - As we can see, RMSE value on training set is around 0.33 compared to 1.17 in linear regression and RMSE value on test set is around 0.63 compared to 1.18 in linear regression
# - This is a good lift compared to linear regression model

# In[58]:


print('RMSE on training data: ', np.sqrt(mean_squared_error(y_train, y_pred_train_rf)))
print('RMSE on test data: ', np.sqrt(mean_squared_error(y_test, y_pred_test_rf)))
print('R2 on training data: ', r2_score(y_train, y_pred_train_rf))
print('R2 on test data: ', r2_score(y_test, y_pred_test_rf))


# #### Feature importances

# - Grade and subgrade seems to have very high values which also makes sense from our earlier graphs

# In[59]:


imp_dict = {}
for i,j in list(zip(X_test, grid_search.feature_importances_)):
    imp_dict[i] = j
imp_dict = {k:v for k,v in sorted(imp_dict.items(), key =lambda item: item[1])}

plt.figure(figsize = (10,5))
plt.bar([x for x in imp_dict.keys()], [y for y in imp_dict.values()])
plt.xticks(rotation=90)
plt.show()

#print(imp_dict)


# ### Let's try one more ensemble model based on boosting technique called XGboost
# - XGboost is based on boosting technique
# - Each model in ensemble learns from previous model
# - XGboost tries to fit residuals from previous model
# - The sum of the individual predictions is considered the prediction of the overall system
# - To select a best set of hyperparameters we can form a grid of best parameter combinations we can think of using GridsearchCV with a cross validation of 3 to come up with best set of parameters
# - In our case best parameters are:
#     - Estimators=70 - Number of trees
#     - max_depth = 3 - Each tree will have a maximum of 15 levels of splits
#     - Every other parameter is set to default value

# In[60]:


# param_grid = [{'n_estimators':[60, 70, 80, 100, 120], 'max_depth':[3,5,7]}]
# gbrt = GradientBoostingRegressor()
# grid_search_gbrt = GridSearchCV(gbrt, param_grid, cv=3, scoring="r2")
# grid_search_gbrt.fit(X_train, y_train)
# final_gbrt = grid_search_gbrt.best_params_
# print(final_gbrt)
# grid_search_gbrt.best_estimator_


# #### Fitting and predicting on both test and train
# - Let's fit the model on train data and evaluate on both train and test data

# In[61]:


gbrt = GradientBoostingRegressor(max_depth=3, n_estimators=70, learning_rate=1)
gbrt.fit(X_train, y_train)
y_pred_train_xg = gbrt.predict(X_train)
y_pred_test_xg = gbrt.predict(X_test)


# #### RMSE values
# - As we can see, RMSE value on training is around 0.47 and test set is around 0.75
# - This model is performing in between linear regression and random forest

# In[62]:


print('RMSE on training data: ', np.sqrt(mean_squared_error(y_train, y_pred_train_xg)))
print('RMSE on test data: ', np.sqrt(mean_squared_error(y_test, y_pred_test_xg)))
print('R2 on training data: ', r2_score(y_train, y_pred_train_xg))
print('R2 on test data: ', r2_score(y_test, y_pred_test_xg))


# #### Feature importances

# - Gives same insight as random forest

# In[63]:


imp_dict = {}
for i,j in list(zip(X_test, gbrt.feature_importances_)):
    imp_dict[i] = j
imp_dict = {k:v for k,v in sorted(imp_dict.items(), key =lambda item: item[1])}

plt.figure(figsize = (10,5))
plt.bar([x for x in imp_dict.keys()], [y for y in imp_dict.values()])
plt.xticks(rotation=90)
plt.show()

#print(imp_dict)


# ### Let's try to enhance the models

# - As we have seen from all the three models, grade and subgrade seems to influence the model heavily
# - In order to decrease the complexity of model, let's try running our best model yet, i.e random forest on just subgrade and see what the results looks like
# - It turned out to be a decent model with RMSE of 1.5 on test data
# - If we reduce model complexity and data collection efforts (only single variable needs to be collected for this model) and can compromise for little less accuracy, we can use this model

# In[64]:


X_train1 = np.array(X_train.loc[:,'sub_grade']).reshape(-1,1)
X_test1 = np.array(X_test.loc[:,'sub_grade']).reshape(-1,1)
rf1 = RandomForestRegressor(n_estimators=80, max_depth=15, bootstrap=True, max_features=None)
rf1.fit(X_train1, y_train)


# #### Fitting and predicting on both test and train
# - Let's fit the model on train data and evaluate on both train and test data

# In[65]:


#rf.fit(X_train, y_train)
y_pred_train_rf_simple = rf1.predict(X_train1)
y_pred_test_rf_simple = rf1.predict(X_test1)


# #### RMSE values
# - As we can see, RMSE value on training is around 1.38 compared to around 1.17 in linear regression and RMSE value on test set is around 1.52 compared to around 1.18 in linear regression
# - Hence this model performance is similar to linear regression model with all the variables

# In[66]:


print('RMSE on training data: ', np.sqrt(mean_squared_error(y_train, y_pred_train_rf_simple)))
print('RMSE on test data: ', np.sqrt(mean_squared_error(y_test, y_pred_test_rf_simple)))
print('R2 on training data: ', r2_score(y_train, y_pred_train_rf_simple))
print('R2 on test data: ', r2_score(y_test, y_pred_test_rf_simple))


# ### Model Summary

# In[67]:


rmse_train = [round(np.sqrt(mean_squared_error(y_train, train_pred)),4),       round(np.sqrt(mean_squared_error(y_train, y_pred_train_rf)),4),       round(np.sqrt(mean_squared_error(y_train, y_pred_train_xg)),4),       round(np.sqrt(mean_squared_error(y_train, y_pred_train_rf_simple)),4)]

rmse_train = pd.DataFrame(np.array(rmse_train).reshape(1,4))

rmse_test = [round(np.sqrt(mean_squared_error(y_test, ridge_test_pred)),4),       round(np.sqrt(mean_squared_error(y_test, y_pred_test_rf)),4),       round(np.sqrt(mean_squared_error(y_test, y_pred_test_xg)),4),       round(np.sqrt(mean_squared_error(y_test, y_pred_test_rf_simple)),4)]

rmse_test = pd.DataFrame(np.array(rmse_test).reshape(1,4))

r2_train = [round(r2_score(y_train, train_pred),4),       round(r2_score(y_train, y_pred_train_rf),4),       round(r2_score(y_train, y_pred_train_xg),4),       round(r2_score(y_train, y_pred_train_rf_simple),4)]

r2_train = pd.DataFrame(np.array(r2_train).reshape(1,4))

r2_test = [round(r2_score(y_test, ridge_test_pred),4),       round(r2_score(y_test, y_pred_test_rf),4),       round(r2_score(y_test, y_pred_test_xg),4),       round(r2_score(y_test, y_pred_test_rf_simple),4)]

r2_test = pd.DataFrame(np.array(r2_test).reshape(1,4))

model_summ = rmse_train.append(rmse_test).append(r2_train).append(r2_test)
model_summ.columns = ['Linear Ridge','Random Forest','XGBoost','Simplified Random Forest']
model_summ.index = ['Train_RMSE','Test_RMSE','Train_R2','Test_R2']


# In[68]:


model_summ


# In[69]:


fig,axes = plt.subplots(1,4,figsize = (20,5))
sns.barplot(ax = axes[0],y=model_summ.iloc[:,0],x=model_summ.index)
sns.barplot(ax = axes[1],y=model_summ.iloc[:,1],x=model_summ.index)
sns.barplot(ax = axes[2],y=model_summ.iloc[:,2],x=model_summ.index)
sns.barplot(ax = axes[3],y=model_summ.iloc[:,3],x=model_summ.index)
axes[0].title.set_text('Linear Ridge')
axes[1].title.set_text('Random Forest')
axes[2].title.set_text('XGBoost')
axes[3].title.set_text('Simplified Random Forest')
axes[0].set_ylabel('')
axes[1].set_ylabel('')
axes[2].set_ylabel('')
axes[3].set_ylabel('')
plt.show()


# ### Scope for improvement

# - 10,000 rows is usually less to build an effective machine learning model. So first step would be data augmentation
# - Imputation of missing values can also be done with unsupervised clustering algorithm like KNN. Sklearn provides direct implementation of the same
# - We can spend more time on hyperparameter tuning and get the best set of parameters to generalize the model predictions well
# - We could manually look at job titles and combine them into fewer categories. For example manager and product manager can be combined and so on
# - We can think of doing feature engineering by leveraging domain expertise of employees
# - We can create pipeline of all data processing steps so that it makes it easy for us to deploy models into production

# In[82]:


os.system('jupyter nbconvert --to html Predicting_interest_rate.ipynb --no-input')


# In[71]:


#### Linear Ridge regression


# In[72]:


# print('RMSE on training data: ', round(np.sqrt(mean_squared_error(y_train, train_pred)),4))
# print('RMSE on test data: ', round(np.sqrt(mean_squared_error(y_test, ridge_test_pred)),4))
# print('R2 on training data: ', round(r2_score(y_train, train_pred),4))
# print('R2 on test data: ', round(r2_score(y_test, ridge_test_pred),4))


# In[73]:


#### Random forest


# In[74]:


# print('RMSE on training data: ', round(np.sqrt(mean_squared_error(y_train, y_pred_train_rf)),4))
# print('RMSE on test data: ', round(np.sqrt(mean_squared_error(y_test, y_pred_test_rf)),4))
# print('R2 on training data: ', round(r2_score(y_train, y_pred_train_rf),4))
# print('R2 on test data: ', round(r2_score(y_test, y_pred_test_rf),4))


# In[75]:


#### XG Boost


# In[76]:


# print('RMSE on training data: ', round(np.sqrt(mean_squared_error(y_train, y_pred_train_xg)),4))
# print('RMSE on test data: ', round(np.sqrt(mean_squared_error(y_test, y_pred_test_xg)),4))
# print('R2 on training data: ', round(r2_score(y_train, y_pred_train_xg),4))
# print('R2 on test data: ', round(r2_score(y_test, y_pred_test_xg),4))


# In[77]:


#### Random forest with just one variable Subgrade


# In[78]:


# print('RMSE on training data: ', round(np.sqrt(mean_squared_error(y_train, y_pred_train_rf_simple)),4))
# print('RMSE on test data: ', round(np.sqrt(mean_squared_error(y_test, y_pred_test_rf_simple)),4))
# print('R2 on training data: ', round(r2_score(y_train, y_pred_train_rf_simple),4))
# print('R2 on test data: ', round(r2_score(y_test, y_pred_test_rf_simple),4))


# In[79]:


# rf = RandomForestRegressor(n_estimators=120, max_depth=25, bootstrap=True, max_features=3)
# plot_learning_curves(rf)
# plt.title('Model Learning curve')
# plt.xlabel('Training set size')
# plt.ylabel('RMSE')
# plt.show()


# In[80]:


# fig, axes = plt.subplots(2, 3,figsize=(12,10))

# sns.pointplot(ax = axes[0][0],x= ['zero','non-zero'],\
#               y = [raw.delinq_2y.isin([0]).mean()*100,100-raw.delinq_2y.isin([0]).mean()*100])

# sns.pointplot(ax = axes[0][1],x= ['zero','non-zero'],\
#               y = [raw.num_collections_last_12m.isin([0]).mean()*100,100-raw.num_collections_last_12m.isin([0]).mean()*100])
# sns.pointplot(ax = axes[1][0],x= ['zero','non-zero'],\
#                y = [raw.current_accounts_delinq.isin([0]).mean()*100,100-raw.current_accounts_delinq.isin([0]).mean()*100])
# sns.pointplot(ax = axes[1][1],x= ['zero','non-zero'],\
#                y = [raw.num_accounts_120d_past_due.isin([0]).mean()*100,100-raw.num_accounts_120d_past_due.isin([0]).mean()*100])
# axes[0][0].set_title('Delinquencies in last 2 years',loc =  'left')
# axes[0][1].set_title('Number of collections in the last 12 months',loc='left')
# axes[1][0].set_title('Number of current accounts which are deliquent',loc='left')
# axes[1][1].set_title('Number of current accounts that are 120 days past due.',loc='left')
# plt.show()


# In[81]:


#categoricals = [x for x in data.columns if x not in numericals and x not in strings]

# for category in categoricals:
#     try:
#         data[category] = data[category].astype('category')
#     except:
#         pass


# In[ ]:




