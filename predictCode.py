import pandas as pd
data = pd.read_csv("IndianCompanies.csv")


import matplotlib.pyplot as plt
import seaborn as sns
print("Welcome to Profit Predictor")
print("--------------------------------------------------------------")
print("Let us Check how mnay Null Values we Have")
##############################################################################
#############   Data Preprocessing    #########################

#creating a subplot for count of Data before and after Cleaning
plt.subplot(1,2,1)
sns.heatmap(data.isnull(),yticklabels = False, cbar = True, cmap = "viridis")
plt.title("Pre-Null Value Count")

for i in data.columns:
    if(i!="State"):
        data[i].fillna(value = data[i].mean(),inplace = True)
plt.subplot(1,2,2)
sns.heatmap(data.isnull(),yticklabels = False, cbar = True, cmap = "viridis")
plt.title("Post-Null Value Count")
plt.savefig("Cleaned Data")
plt.show()
print("----------------------------------------------------------")
print("Let us understand State wise Profit Generated")
companies = data
plt.subplots(figsize=(10,4))
p = sns.barplot(x=companies["State"],y=companies["Profit"], saturation=2, edgecolor = "yellow", linewidth = 2.5,)
p.axes.set_title("\n State\n", fontsize=35)
plt.ylabel("Profit" , fontsize = 25)
plt.xlabel("\nState" , fontsize = 25)
# plt.yscale("log")
plt.xticks(rotation = 90)
for container in p.containers:
    p.bar_label(container,label_type = "center",padding = 8,size = 25,color = "black",rotation = 0,
    bbox={"boxstyle": "round", "pad": 0.6, "facecolor": "pink", "edgecolor": "black", "alpha": 1})

sns.despine(left=True, bottom=True)
plt.savefig("State vs Companies")
plt.show()
print("------------------------------------------------")
print("Let us understand Marketing Spend - it's almost Uniform!")
sns.displot(data=companies, x="Marketing Spend", kde=True, bins = 100,color = "red", facecolor = "green",height = 2, aspect = 3.5)

plt.savefig("Almost Uniform Marketing Spend of Companies.jpg")
plt.show()

print("----------------------------------------")
print()
print("Displaying Correlation Matrix between Varioud Factors")
plt.figure(figsize=(8,6))
floatDB = companies.drop(["State"],axis = 1)
sns.heatmap(floatDB.corr(),annot=True,cmap='Blues')
plt.title("Correlation Matrix between Varioud Factors")
plt.savefig("Correlation Matrix")
plt.show()

print("As noticed, R&D, Administration, Marketinhg Spend have  an high positive correlation with Profit of a Company")
print("Hence all these attributes will be considered for Predicting the Profit")
print("--------------------------------------------")
print("--------------------------------------------")
print()
print()
print("Now let's fit into models!")

#######################################################################
##########Independent and Dependent Attributes##################
                 
x = companies.iloc[:,:-1]
y = companies.iloc[:,-1]


###################################################################
############One Hot Encoding for Categorical Data####################
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
label_encoder_x_1 = LabelEncoder()
x.iloc[: , 3] = label_encoder_x_1.fit_transform(x.iloc[:,3])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)
x = x[:,1:]

from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.3,random_state=122)

######################################################################
############Linear Regression###############################
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error,mean_absolute_error
model_LinearRegression = LinearRegression()
model_LinearRegression.fit(xTrain, yTrain)
pred_test_LinearRegression = model_LinearRegression.predict(xTest)

r2LinearRegression = r2_score(yTest, pred_test_LinearRegression).round(4)
mseLinearRegression = mean_squared_error(yTest, pred_test_LinearRegression).round(4)
rmseLinearRegression = np.sqrt(mean_squared_error(yTest, pred_test_LinearRegression)).round(4)
maeLinearRegression = mean_absolute_error(yTest, pred_test_LinearRegression).round(4)

model_LinearRegression = LinearRegression()
model_LinearRegression.fit(xTrain, yTrain)

##########################Lasso Regression################################
from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha = 0.0001)
model_lasso.fit(xTrain, yTrain)
pred_test_lasso = model_lasso.predict(xTest)

r2_test_lasso = r2_score(yTest, pred_test_lasso).round(4)
mse_test_lasso = mean_squared_error(yTest, pred_test_lasso).round(4)
rmse_test_lasso = np.sqrt(mean_squared_error(yTest, pred_test_lasso)).round(4)
mae_test_lasso = mean_absolute_error(yTest, pred_test_lasso).round(4)

################################XGBoost#################################
from xgboost import XGBRegressor
model_xgb = XGBRegressor(random_state=42)
model_xgb.fit(xTrain, yTrain)

pred_test_xgb = model_xgb.predict(xTest)

r2_test_xgb = r2_score(yTest, pred_test_xgb).round(4)
mse_test_xgb = mean_squared_error(yTest, pred_test_xgb).round(4)
rmse_test_xgb = np.sqrt(mean_squared_error(yTest, pred_test_xgb)).round(4)
mae_test_xgb = mean_absolute_error(yTest, pred_test_xgb).round(4)


###############Model Comparison###################################
models = pd.DataFrame({'Model':["Linear Regression","Lasso","XGBoost"], "Root Mean Square Error": [rmseLinearRegression,rmse_test_lasso, rmse_test_xgb], "R2 Test": [r2LinearRegression, r2_test_lasso, r2_test_xgb],"Mean Absolute Error Test": [maeLinearRegression, mae_test_lasso, mae_test_xgb]})
print(models)
p = plt.figure(figsize = (8,6))
p = sns.set_theme(style ="white")
p= models=models.sort_values(by='Root Mean Square Error',ascending=False)

p = sns.barplot(y= 'Model', x= 'Root Mean Square Error', data= models)
for container in p.containers:
    p.bar_label(container,label_type = 'center',padding = 2,size = 15,color = "Red",rotation = 0,
    bbox={"boxstyle": "round", "pad": 0.3, "facecolor": "yellow", "edgecolor": "black", "alpha": 1})
plt.title('COMPARE THE MODEL')
plt.xlabel('RMSE Value')
plt.ylabel('Model');
plt.show()
'''
As noticed, from Linear Regression's 7838.3020 , our accuracy increased to 3586.7639 in XGBoost.
Decreased the error by 54.24 %.
'''
