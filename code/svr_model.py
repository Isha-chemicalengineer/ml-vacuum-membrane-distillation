import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('VMD_Data.csv')
X = df[['Feed Temperature', 'Permeate Side Pressure', 'Feed Flow Rate']]
# Target variable
Y = df['Permeate Flux']
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled,Y,test_size=0.33,random_state=42)
param_grid = {
    'C': [10, 100, 500],
    'gamma': ['scale', 0.01, 0.1],
    'epsilon': [0.001, 0.01, 0.1]}
# SVR model
svr = SVR(kernel='rbf')
grid = GridSearchCV(svr,param_grid, cv=5,scoring='r2')
grid.fit(X_train, Y_train)
best_model = grid.best_estimator_
Y_pred = best_model.predict(X_test)
r2 = r2_score(Y_test, Y_pred)
rmse = (mean_squared_error(Y_test, Y_pred)) ** 0.5
mae = mean_absolute_error(Y_test, Y_pred)
print(f'R² = {r2:.5f}')
print(f'RMSE = {rmse:.5f}')
print(f'MAE = {mae:.5f}')
plt.figure(figsize=(7, 5))
plt.scatter(Y_test,Y_pred,c='blue',alpha=0.6)
plt.plot([0, max(Y_test)],[0, max(Y_test)],'r--',lw=2)
plt.xlabel('Actual Permeate Flux (kg/m²·s)')
plt.ylabel('Predicted Permeate Flux (kg/m²·s)')
plt.title('SVR (RBF) Predicted vs Actual Flux')
plt.grid(True)
plt.show()
