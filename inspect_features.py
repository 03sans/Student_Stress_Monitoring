import joblib

feature_columns = joblib.load("model/feature_columns.pkl")

print("Number of features:", len(feature_columns))
print("\nFirst 20 feature names:")
for col in feature_columns[:20]:
    print(col)
