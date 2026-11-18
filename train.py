import joblib
from sklearn.datasets import load_digits
from sklearn.svm import SVC

print("Loading dataset...")
digits = load_digits()
X, y = digits.data, digits.target

print("Training pre-tuned classifier...")
# Using a Support Vector Classifier which performs very well on digit recognition
model = SVC(gamma=0.001, C=100.0, probability=True)
model.fit(X, y)

# Save the trained model to disk
joblib.dump(model, 'model.joblib')
print("Model saved successfully as 'model.joblib'!")
