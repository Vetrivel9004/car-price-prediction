# Importing the libraries
import numpy as np
import pickle
import pandas as pd

df = pd.read_csv('car data.csv')

final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
# removing carname because its of no use in the mathematical calculations

final_dataset['Current Year']=2021

final_dataset['no_year']=final_dataset['Current Year']- final_dataset['Year']

final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset=final_dataset.drop(['Current Year'],axis=1)

# dealing with categorical data

final_dataset=pd.get_dummies(final_dataset,drop_first=True)

X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor()

# Hyperparameter tuning and model training

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)

from sklearn.model_selection import RandomizedSearchCV

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)

pickle.dump(rf_random,open('model.pkl','wb'))

#loading model to compare the results
model = pickle.load(open('model.pkl','rb'))