import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

housing = pd.read_excel("C:/Users/asus/Downloads/ml.xlsx")
print(housing.head())
print(housing.info())
print(housing['CHAS'].value_counts())
housing.hist(bins=50, figsize=(20,15)) 
plt.show()
#“Split the dataset such that the proportion of values in CHAS remains the same in both train and test sets.”
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)# create a stratified shuffle split object with 1 split, 20% test size, and a random state for reproducibility
for train_index, test_index in split.split(housing, housing['CHAS']):# stratified sampling based on the 'CHAS' column
    strat_train_set = housing. loc[train_index]# create the stratified training set using the train indices
    strat_test_set = housing.loc[test_index]# create the stratified test set using the test indices
print(strat_train_set['CHAS'].value_counts())# print the value counts of the 'CHAS' column in the stratified training set to verify the distribution
print(strat_test_set['CHAS'].value_counts())# print the value counts of the 'CHAS' column in the stratified test set to verify the distribution

#Attribute Combination
housing['TAXRM'] = housing['TAX']/housing['RM']# create a new attribute 'TAXRM' by dividing the 'TAX' column by the 'RM' column

attributes = ['MEDV','RM','ZN','LSTAT','TAXRM']# select the attributes to be included in the scatter matrix
scatter_matrix(housing[attributes], figsize=(12,8))# create a scatter matrix of the selected attributes with a specified figure size
plt.show()

# Ensure derived attribute TAXRM exists in both train and test sets
strat_train_set['TAXRM'] = strat_train_set['TAX'] / strat_train_set['RM']
strat_test_set['TAXRM'] = strat_test_set['TAX'] / strat_test_set['RM']

housing = strat_train_set.copy()# create a copy of the stratified training set to work with for further analysis and modeling


#Looking for Correlations
corr_matrix = housing.corr()# calculate the correlation matrix of the housing dataset
print(corr_matrix['MEDV'].sort_values(ascending=False))# print the correlation of the 'MEDV' column with all other columns, sorted in ascending order



from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)

#Creating Pipeline


my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),# add an imputer step to handle missing values using the median strategy])
    ('std_scaler', StandardScaler()),# add a standard scaler step to standardize the features

])  
housing_num_tr = my_pipeline.fit_transform(housing.drop('MEDV', axis=1))# apply the pipeline to the housing dataset, dropping the 'MEDV' column to create a transformed feature set
print(housing_num_tr)# print the transformed feature set after applying the pipeline

# Show feature names and count
feature_names = list(housing.drop('MEDV', axis=1).columns)
print(f"\nFeature names ({len(feature_names)} total): {feature_names}")
print(f"Shape of training data: {housing_num_tr.shape}")

#SELECTING A DESIRED MODEL FOR DRAGON REAL ESTATES

#model = LinearRegression()# create an instance of the Linear Regression model
# model = DecisionTreeRegressor()# create an instance of the Decision Tree Regressor model
model = RandomForestRegressor()# create an instance of the Random Forest Regressor model
model.fit(housing_num_tr,housing['MEDV'])# fit the model to the transformed feature set and the target variable 'MEDV'
print(model)# print the trained model to verify that it has been fitted correctly

some_data = housing.iloc[:5]# select the first 5 rows of the housing dataset to create a sample data set
some_labels = housing['MEDV'].iloc[:5]# select the corresponding 'MEDV' values for the sample data set to create a sample labels set
print("Predictions:", model.predict(my_pipeline.transform(some_data.drop('MEDV', axis=1))))# make predictions using the trained model on the sample data set, after applying the same transformations as the training data
print("Labels:",list(some_labels))# print the actual labels for the sample data set to compare with the predictions

#EVALUATING THE MODEL

housing_predictions = model.predict(housing_num_tr)# make predictions on the entire training set using the trained model
lin_mse = mean_squared_error(housing['MEDV'],housing_predictions)# calculate the mean squared error between the actual 'MEDV' values and the predicted values
lin_rmse = np.sqrt(lin_mse)# calculate the root mean squared error by taking the square root of the mean squared erro
print(lin_rmse)# print the root mean squared error to evaluate the performance of the model on the training set

# since on using lineaar regression  modekwe see that our mean squared erro(mse) is 17.78 which is quiet high and not good for our model so we will use decision tree regressor to see if we can get better results
# now when we used decision tree regreesor we se that our mean square error is 0 which means it is OVERFITTING our model is performing perfectly on the training data but it may not generalize well to unseen data.

# CROSS VALIDATION (to check if our model is overfitting or not)
scores = cross_val_score(model, housing_num_tr, housing['MEDV'], scoring="neg_mean_squared_error", cv=10)# perform cross-validation on the model using 10 folds and calculate the negative mean squared error for each fold
rmse_scores = np.sqrt(-scores)
def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

print_scores(rmse_scores)# print the cross-validation scores, mean, and standard deviation to evaluate the performance of the model across different folds

#  Testing the model on test data
X_test = strat_test_set.drop('MEDV', axis=1)# create the test feature set by dropping the 'MEDV' column from the stratified test set
y_test = strat_test_set['MEDV'].copy()# create the test labels set by copying the 'MEDV' column from the stratified test set
X_test_prepared = my_pipeline.transform(X_test)# apply the same transformations to the test feature set as were applied to the training data using the pipeline
final_predictions = model.predict(X_test_prepared)# make predictions on the prepared test feature set using the trained model
final_mse = mean_squared_error(y_test, final_predictions)# calculate the mean squared error between the actual test labels and the final predictions
final_rmse = np.sqrt(final_mse)# calculate the root mean squared error by taking the square root of the mean squared error
print(final_rmse)# print the final root mean squared error to evaluate the performance of the model on the test set

#USING THE MODEL
from joblib import dump, load
import os

model_filename = 'Dragon_real_estate_price_predictor.joblib'
# If a saved model exists, load it; otherwise save the currently trained model
if os.path.exists(model_filename):
    model = load(model_filename)
    print(f"\nLoaded model from {model_filename}")
else:
    dump(model, model_filename)
    print(f"\nSaved trained model to {model_filename}")

# Make prediction using actual test data (has all 14 features)
# Pick first sample from test set
test_sample = X_test.iloc[:1]
test_sample_prepared = my_pipeline.transform(test_sample)
prediction = model.predict(test_sample_prepared)
print(f"Test sample prediction: {prediction[0]:.2f}")
print(f"Actual test label: {y_test.iloc[0]:.2f}")
