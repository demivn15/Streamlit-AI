import joblib
from sklearn.datasets import load_digits
from sklearn.svm import SVC

# Load standard digits
digits = load_digits()
X, y = digits.data, digits.target

# Train a robust Support Vector Classifier
model = SVC(gamma=0.001, C=10.0, probability=True)
model.fit(X, y)

# Save
joblib.dump(model, 'model.joblib')
print("New balanced model.joblib saved!")
