import time  # Import time module
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

start_time = time.time()

# Load dataset
df = pd.read_csv("student_health_data.csv")
print("Processing Data")

#Organize Data
df = df.drop(columns=["Student_ID"])  
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

# Create Matrix System Elements
X = df.drop(columns=["Health_Risk_Level"])
y = df["Health_Risk_Level"]

#Training and Testing
print("Training...")
split_start = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
split_end = time.time()
print(f"Data split time: {split_end - split_start:.4f} seconds")

scale_start = time.time()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
scale_end = time.time()
print(f"Scaling time: {scale_end - scale_start:.4f} seconds")

train_start = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
train_end = time.time()
print(f"Training time: {train_end - train_start:.4f} seconds")

predict_start = time.time()
y_pred = model.predict(X_test)
predict_end = time.time()
print(f"Prediction time: {predict_end - predict_start:.4f} seconds")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.4f} seconds")

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

