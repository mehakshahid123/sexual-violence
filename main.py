
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load the dataset
data = pd.read_csv("NCRB_Table_3A.3_1755096074268.csv")

print("First 5 rows of data:")
print(data.head())

# Step 2: Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Step 3: Define features (X) and target (y)
# Change 'Type_of_Violence' to your actual target column name in CSV
target_column = 'Type_of_Violence'
if target_column not in data.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset. Please update target_column variable.")

X = data.drop(columns=[target_column])
y = data[target_column]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save model and encoders
joblib.dump(model, "sexual_violence_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\nModel and encoders saved successfully!")

# Step 8: Sample prediction
print("\nSample prediction from first row of test set:")
sample_input = X_test.iloc[0:1]
print("Input:", sample_input)
print("Predicted Violence Type:", model.predict(sample_input)[0])
