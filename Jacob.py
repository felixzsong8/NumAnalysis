import time  # Import time module
import pandas as pd
import numpy as np 

df = pd.read_csv("student_health_data.csv")
print("Processing Data")

#Organize data
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

#Create Matrix System Elements
X = df.drop(columns=["Health_Risk_Level"])
y = df["Health_Risk_Level"]


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

