import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([
    [1000, 2, 1, 1],
    [1500, 3, 2, 2],
    [2000, 4, 3, 2],
    [800, 2, 1, 0],
    [1200, 3, 2, 1]
])
y = np.array([10000000, 15000000, 20000000, 8500000, 13000000])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save to Desktop/House_price folder
import os

# Replace 'YourUsername' with your actual Windows username
save_path = r"C:\Users\neeth\Desktop\House_price\house.pkl"

with open(save_path, 'wb') as f:
    pickle.dump(model, f)

print("Model saved to:", save_path)

