import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset
data = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)
print("\nFirst five rows:")
print(data.head())

# 2. Separate features and target
X = data.drop("price_range", axis=1)
y = data["price_range"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 5. Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTraining completed!")
print(f"Model accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# 7. Save the trained model
joblib.dump(model, "smartphone_model.pkl")

# Save feature names for use by the application
joblib.dump(list(X.columns), "feature_names.pkl")

print("\nModel saved as smartphone_model.pkl")
print("Feature names saved as feature_names.pkl")
print("\nTraining completed successfully!")
