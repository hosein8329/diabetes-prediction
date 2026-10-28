
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. خواندن دیتابیس
df = pd.read_csv("diabetes.csv")

# 2. جدا کردن ورودی‌ها از نتیجه
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# 3. تقسیم داده‌ها به آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. ساخت مدل
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 5. آموزش مدل
model.fit(X_train, y_train)

# 6. ارزیابی عملکرد
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 7. ذخیره مدل آموزش‌دیده
joblib.dump(model, "diabetes_model.joblib")

print("\nModel saved successfully.")
