import pandas as pd

# Load dataset
df = pd.read_csv("f1_enhanced_dataset_for_analysis.csv")

# Create target
df['podium'] = df['FinishPosition'].apply(lambda x: 1 if x <= 3 else 0)

# Take only required columns (change names if needed)
df = df[['Driver','Team','StartPosition','QualifyingPosition','Weather','TireStrategy','podium']]

# Convert categorical to numeric
df = pd.get_dummies(df)

# Take 150 rows → 100 train + rest test
df = df.sample(n=150, random_state=21)

# X and Y
X = df.drop(labels=['podium'], axis=1)
Y = df[['podium']]

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().set_output(transform='pandas')
X_pre = scaler.fit_transform(X)

# Train-Test split (EXACT 100 train)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X_pre, Y, train_size=100, random_state=21)

print(xtrain.shape)   # (100, columns)
print(xtest.shape)

# Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(xtrain, ytrain)

# Scores
print(model.score(xtrain, ytrain))
print(model.score(xtest, ytest))

# Predictions
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(ytest, ts_pred)
print(cf)

# Save model
import pickle
with open('model.pkl','wb') as file:
    pickle.dump(model,file)

# Load model
with open('model.pkl','rb') as file:
    m = pickle.load(file)