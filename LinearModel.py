import time  # Import time module
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Start timer
start_time = time.time()

# Load dataset
df = pd.read_csv("student_health_data.csv")
print("Processing Data")

# Drop non-numeric or irrelevant columns
df = df.drop(columns=["Student_ID"])  
print(df.columns)
# Convert categorical values to numeric
activity_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
df['Physical_Activity'] = df['Physical_Activity'].map(activity_mapping)

sleep_mapping = {'Poor': 1, 'Moderate': 2, 'Good': 3}
df['Sleep_Quality'] = df['Sleep_Quality'].map(sleep_mapping)

gender_mapping = {'M': 1, 'F': 2}
df['Gender'] = df['Gender'].map(gender_mapping)

mood_mapping = {'Stressed': 1, 'Neutral': 2, 'Happy': 3}
df['Mood'] = df['Mood'].map(mood_mapping)

HealthRisk_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
df['Health_Risk_Level'] = df['Health_Risk_Level'].map(HealthRisk_mapping)

# Separate features and target variable
X = df.drop(columns=["Health_Risk_Level"])
y = df["Health_Risk_Level"]

# Split into training and testing sets
print("Training...")
split_start = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
split_end = time.time()
print(f"Data split time: {split_end - split_start:.4f} seconds")

# Standardize features
scale_start = time.time()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
scale_end = time.time()
print(f"Scaling time: {scale_end - scale_start:.4f} seconds")

# Train a Linear Regression model
train_start = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
train_end = time.time()
print(f"Training time: {train_end - train_start:.4f} seconds")

# Make predictions
predict_start = time.time()
y_pred = model.predict(X_test)
predict_end = time.time()
print(f"Prediction time: {predict_end - predict_start:.4f} seconds")

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")

# End timer
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print("------------------------------------------------------------")


import numpy as np

def jacobi(A, b, tol=1e-6, max_iterations=1000):
    n = len(A)
    x = np.zeros(n)  # Initial guess
    x_new = np.zeros(n)

    for _ in range(max_iterations):
        for i in range(n):
            sum_except_i = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_except_i) / A[i][i]

        if np.linalg.norm(x_new - x) < tol:  # Convergence check
            break
        x = x_new.copy()

    return x


def gauss_seidel(A, b, tol=1e-6, max_iterations=1000):
    n = len(A)
    x = np.zeros(n)

    for _ in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            sum_except_i = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_except_i) / A[i][i]

        if np.linalg.norm(x - x_old) < tol:
            break

    return x
    
def timed_solver(solver_func, A, b):
    start = time.time()
    w = solver_func(A, b)
    end = time.time()
    return w, end - start


# Add bias term to X
X_train_aug = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test_aug = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Normal equation components
A = X_train_aug.T @ X_train_aug
b = X_train_aug.T @ y_train.values

lr_start = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
lr_end = time.time()
print(f"[Linear Regression] Training Time: {lr_end - lr_start:.4f} seconds")

# Train and time Random Forest
rf_start = time.time()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_end = time.time()

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"[Random Forest] MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}, Training Time: {rf_end - rf_start:.4f} seconds")

# Solve using Jacobi method
w_jacobi, time_jacobi = timed_solver(jacobi, A, b)
y_pred_jacobi = X_test_aug @ w_jacobi
mse_jacobi = mean_squared_error(y_test, y_pred_jacobi)
r2_jacobi = r2_score(y_test, y_pred_jacobi)
print(f"[Jacobi] MSE: {mse_jacobi:.4f}, R²: {r2_jacobi:.4f}, Time: {time_jacobi:.4f} seconds")

# Solve using Gauss-Seidel method with timing
w_gs, time_gs = timed_solver(gauss_seidel, A, b)
y_pred_gs = X_test_aug @ w_gs
mse_gs = mean_squared_error(y_test, y_pred_gs)
r2_gs = r2_score(y_test, y_pred_gs)
print(f"[Gauss-Seidel] MSE: {mse_gs:.4f}, R²: {r2_gs:.4f}, Time: {time_gs:.4f} seconds")

