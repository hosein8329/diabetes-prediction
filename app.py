
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.joblib")

model = load_model()

st.title("🩺 سیستم پیش‌بینی دیابت")
st.write(
    "اطلاعات زیر را برای پیش‌بینی آزمایشی وارد کنید."
)

# نام ویژگی‌ها باید با دیتاست زمان آموزش یکسان باشد
features = list(model.feature_names_in_)

labels = {
    "Pregnancies": "تعداد بارداری",
    "Glucose": "گلوکز",
    "BloodPressure": "فشار خون",
    "SkinThickness": "ضخامت چین پوستی",
    "Insulin": "انسولین",
    "BMI": "شاخص توده بدنی",
    "DiabetesPedigreeFunction": "شاخص سابقه خانوادگی",
    "Age": "سن"
}

defaults = {
    "Pregnancies": 1.0,
    "Glucose": 120.0,
    "BloodPressure": 70.0,
    "SkinThickness": 20.0,
    "Insulin": 80.0,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30.0
}

with st.form("prediction_form"):
    values = {}

    for feature in features:
        values[feature] = st.number_input(
            labels.get(feature, feature),
            min_value=0.0,
            value=defaults.get(feature, 0.0),
            step=1.0
        )

    submitted = st.form_submit_button(
        "پیش‌بینی"
    )

if submitted:
    X_new = pd.DataFrame(
        [values],
        columns=features
    )

    prediction = int(model.predict(X_new)[0])

    if hasattr(model, "predict_proba"):
        probability = float(
            model.predict_proba(X_new)[0][1]
        )
        st.write(
            f"امتیاز احتمال مدل برای کلاس ۱: "
            f"{probability:.1%}"
        )

    if prediction == 1:
        st.warning(
            "مدل این نمونه را در کلاس دیابت قرار داده است."
        )
    else:
        st.success(
            "مدل این نمونه را در کلاس بدون دیابت قرار داده است."
        )

    st.caption(
        "این ابزار آموزشی است و برای تشخیص پزشکی "
        "یا تصمیم درمانی استفاده نشود."
    )
