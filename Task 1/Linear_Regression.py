import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read dataset
housing = pd.read_csv('/kaggle/input/train-csv/train.csv')

# Keep only required columns
house = housing[['price', 'area', 'bedrooms', 'bathrooms']].rename(
    columns={'price': 'Price', 'area': 'Area', 'bedrooms': 'Bedrooms', 'bathrooms': 'Bathrooms'}
)

# Check for null values
print("Null values:\n", house.isnull().sum())

# Outlier removal for all columns
for col in ['Price', 'Area', 'Bedrooms', 'Bathrooms']:
    Q1 = house[col].quantile(0.25)
    Q3 = house[col].quantile(0.75)
    IQR = Q3 - Q1
    house = house[(house[col] >= Q1 - 1.5 * IQR) & (house[col] <= Q3 + 1.5 * IQR)]

# Boxplots after outlier removal
fig, axs = plt.subplots(2, 2, figsize=(10, 5))
sns.boxplot(x=house['Price'], ax=axs[0, 0])
sns.boxplot(x=house['Area'], ax=axs[0, 1])
sns.boxplot(x=house['Bedrooms'], ax=axs[1, 0])
sns.boxplot(x=house['Bathrooms'], ax=axs[1, 1])
plt.tight_layout()
plt.show()

# Features and target
X = house[['Area', 'Bedrooms', 'Bathrooms']]
y = house['Price']

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Model parameters
print("Intercept:", lr.intercept_)
print("Coefficients:", lr.coef_)

# Predictions
y_pred_train = lr.predict(x_train)
y_pred_test = lr.predict(x_test)

# Training R²
print("Training R²:", r2_score(y_train, y_pred_train))
# Test R²
print("Test R²:", r2_score(y_test, y_pred_test))

# Visualization of predictions (Test Data)
plt.scatter(y_test, y_pred_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect fit line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
