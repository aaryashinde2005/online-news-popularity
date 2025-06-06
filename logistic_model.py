import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('C:\\Users\\aarya\\OneDrive\\Desktop\\onlinenewspopularity\\OnlineNewsPopularity.csv')


# Clean column names
df.columns = df.columns.str.strip()

# Drop non-numeric or irrelevant columns
df = df.drop(['url', 'timedelta'], axis=1)

# Create a binary target: popular = 1 if shares > 1400, else 0
df['popular'] = (df['shares'] > 1400).astype(int)

# Drop the original target column
df = df.drop('shares', axis=1)

# Define input and output
X = df.drop('popular', axis=1)
y = df['popular']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
