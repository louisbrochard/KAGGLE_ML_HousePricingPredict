#!usr/local/bin/python3
# -*- coding = utf-8 -*


import pandas as pd 
from sklearn.ensemble import RandomForestRegressor




train = pd.read_csv('train2.csv')
features = ['MSSubClass', 'LotArea', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea','MiscVal', 'MoSold', 'YrSold', 'SaleType']

X = train[features]
y = train.SalePrice

print('On commence le programme qui va calculer les prix des maisons du fichier test...')

model = RandomForestRegressor(random_state = 1)
model.fit(X,y)

test = pd.read_csv('test2.csv')

X_test = test[features]
predictions = model.predict(X_test)

output = pd.DataFrame({'Id' : test.Id, 'SalePrice' : predictions})
output.to_csv('my_submission2.csv', index=False)
print("Your submission has been successfully saved. Find it on the 'HousePricingPredict' file")
