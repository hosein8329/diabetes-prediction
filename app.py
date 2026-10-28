
import streamlit as st
import pandas as pd
import joblib

# -------------------- تنظیمات صفحه --------------------
st.set_page_config(
    page_title="Diabetes AI | تحلیل سلامت",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- طراحی ظاهری --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');

/* تنظیمات کلی */
html, body, [class*="css"], [data-testid="stApp"],
[data-testid="stAppViewContainer"], input, textarea, button, label {
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #fffaf3 0%, #fff1df 55%, #fff8ed 100%);
    color: #3f2a1d;
    direction: rtl;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stMain"],
[data-testid="stAppViewContainer"] {
    direction: rtl;
    text-align: right;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
label {
    direction: rtl;
    text-align: right;
}

/* سربرگ اصلی */
.hero {
    background: linear-gradient(120deg, #ea580c 0%, #f97316 52%, #fb923c 100%);
    padding: 34px;
    border-radius: 24px;
    color: #ffffff;
    margin-bottom: 25px;
    box-shadow: 0 12px 28px rgba(234, 88, 12, 0.19);
    border: 1px solid rgba(255,255,255,0.25);
}

.hero h1 {
    color: #ffffff !important;
    font-size: 30px;
    font-weight: 800;
    margin: 0 0 12px 0;
}

.hero p {
    color: #fff7ed !important;
    font-size: 15px;
    line-height: 2.1;
    margin: 0;
}

/* کارت‌های اطلاعات */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #f4d7b5;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(124, 65, 20, 0.05);
    text-align: right;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"],
div[data-testid="stMetricDelta"] {
    text-align: right;
}

div[data-testid="stMetricLabel"] {
    color: #8b5e3c;
}

div[data-testid="stMetricValue"] {
    color: #ea580c;
    font-weight: 800;
}

/* کارت توضیحات */
.section-card {
    background: rgba(255,255,255,0.88);
    padding: 23px;
    border: 1px solid #f2d7b8;
    border-radius: 19px;
    margin: 18px 0;
    box-shadow: 0 5px 18px rgba(124, 65, 20, 0.035);
}

.section-title {
    font-size: 20px;
    font-weight: 800;
    color: #c2410c;
    margin-bottom: 8px;
}

.section-subtitle {
    font-size: 13px;
    color: #846b58;
    line-height: 2;
}

/* ورودی‌ها */
input, textarea {
    direction: rtl !important;
    text-align: right !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 11px !important;
    border-color: #edcfad !important;
    background-color: #fffefa !important;
}

div[data-testid="stNumberInput"] button {
    color: #c2410c;
}

/* دکمه‌های اصلی */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(100deg, #ea580c, #f97316) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 45px;
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
    font-weight: 700 !important;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(100deg, #c2410c, #ea580c) !important;
    box-shadow: 0 5px 14px rgba(234, 88, 12, 0.18);
}

/* گزینه‌های انتخابی */
[data-testid="stCheckbox"] {
    direction: rtl;
}

[data-testid="stCheckbox"] label {
    color: #513522 !important;
    display: flex;
    flex-direction: row-reverse;
    justify-content: flex-end;
    gap: 9px;
}

/* پیام‌ها */
[data-testid="stAlert"] {
    direction: rtl;
    text-align: right;
    border-radius: 14px;
}

[data-testid="stAlert"] p {
    text-align: right;
}

/* جدول و جزئیات */
[data-testid="stDataFrame"] {
    direction: rtl;
}

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.7);
    border: 1px solid #f2d7b8;
    border-radius: 13px;
}

/* خطوط جداکننده */
hr {
    border-color: #f0d9bd;
}

/* پاورقی */
.footer {
    text-align: center;
    color: #967963;
    font-size: 12px;
    line-height: 2.2;
    padding: 24px 0 8px;
}

/* مخفی کردن فضای سایدبار */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* نمایش در موبایل */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .hero {
        padding: 22px;
        border-radius: 18px;
    }

    .hero h1 {
        font-size: 23px;
    }

    .section-card {
        padding: 17px;
    }
}
</style>
""", unsafe_allow_html=True)


# -------------------- بارگذاری مدل --------------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.joblib")


try:
    model = load_model()
except Exception:
    st.error(
        "فایل مدل بارگذاری نشد. بررسی کن فایل "
        "diabetes_model.joblib در مسیر پروژه موجود باشد."
    )
    st.stop()


# -------------------- ویژگی‌های مدل --------------------
features = list(
    getattr(model, "feature_names_in_", [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])
)

labels = {
    "Pregnancies": "تعداد بارداری",
    "Glucose": "سطح گلوکز",
    "BloodPressure": "فشار خون دیاستولیک",
    "SkinThickness": "ضخامت چین پوستی",
    "Insulin": "سطح انسولین",
    "BMI": "شاخص توده بدنی (BMI)",
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

descriptions = {
    "Pregnancies": "تعداد دفعات بارداری",
    "Glucose": "معمولاً بر حسب mg/dL",
    "BloodPressure": "معمولاً بر حسب mmHg",
    "SkinThickness": "معمولاً بر حسب میلی‌متر",
    "Insulin": "معمولاً بر حسب μU/mL",
    "BMI": "وزن تقسیم بر مجذور قد بر حسب متر",
    "DiabetesPedigreeFunction": "شاخص عددی سابقه خانوادگی در دیتاست",
    "Age": "سن بر حسب سال"
}


# -------------------- سربرگ --------------------
st.markdown("""
<div class="hero">
    <h1>🩺 سامانه هوشمند پیش‌بینی دیابت</h1>
    <p>
        به داشبورد تحلیل سلامت خوش آمدید.
        <br>
        اطلاعات را وارد کنید و خروجی مدل یادگیری ماشین را مشاهده کنید.
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------- تنظیمات داخل صفحه --------------------
with st.expander("⚙️ تنظیمات نمایش و راهنما", expanded=False):
    setting_col1, setting_col2, setting_col3 = st.columns(3)

    with setting_col1:
        show_chart = st.checkbox(
            "نمایش نمودار اطلاعات",
            value=True
        )

    with setting_col2:
        show_details = st.checkbox(
            "نمایش جزئیات مدل",
            value=True
        )

    with setting_col3:
        show_help = st.checkbox(
            "نمایش راهنمای ورودی‌ها",
            value=True
        )


# -------------------- کارت‌های داشبورد --------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="تعداد ویژگی‌های ورودی",
        value=len(features)
    )

with col2:
    st.metric(
        label="وضعیت مدل",
        value="آماده"
    )

with col3:
    st.metric(
        label="نوع سامانه",
        value="آموزشی"
    )


# -------------------- معرفی فرم --------------------
st.markdown("""
<div class="section-card">
    <div class="section-title">📋 اطلاعات فرد مورد بررسی</div>
    <div class="section-subtitle">
        مقادیر را مطابق ویژگی‌های مورد نیاز مدل وارد کنید.
        واحدهای اندازه‌گیری باید با دیتاست آموزش مدل سازگار باشند.
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------- فرم ورودی --------------------
personal_features = [
    "Pregnancies",
    "Age",
    "BMI",
    "DiabetesPedigreeFunction"
]

lab_features = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin"
]

other_features = [
    f for f in features
    if f not in personal_features + lab_features
]

with st.form("prediction_form", clear_on_submit=False):

    values = {}

    left_col, right_col = st.columns(2, gap="large")

    with right_col:
        st.markdown("### 👤 مشخصات فردی و بدنی")

        for feature in personal_features:
            if feature in features:

                if show_help:
                    st.caption(descriptions.get(feature, ""))

                values[feature] = st.number_input(
                    labels.get(feature, feature),
                    min_value=0.0,
                    max_value=200.0 if feature == "Age" else 1000.0,
                    value=float(defaults.get(feature, 0.0)),
                    step=1.0,
                    key=f"input_{feature}"
                )

    with left_col:
        st.markdown("### 🧪 اطلاعات آزمایشگاهی")

        for feature in lab_features:
            if feature in features:

                if show_help:
                    st.caption(descriptions.get(feature, ""))

                values[feature] = st.number_input(
                    labels.get(feature, feature),
                    min_value=0.0,
                    max_value=10000.0,
                    value=float(defaults.get(feature, 0.0)),
                    step=1.0,
                    key=f"input_{feature}"
                )

    if other_features:
        st.divider()
        st.markdown("### ➕ سایر ویژگی‌های مدل")

        extra_cols = st.columns(2)

        for i, feature in enumerate(other_features):
            with extra_cols[i % 2]:
                values[feature] = st.number_input(
                    labels.get(feature, feature),
                    min_value=0.0,
                    value=float(defaults.get(feature, 0.0)),
                    step=1.0,
                    key=f"input_{feature}"
                )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 تحلیل اطلاعات و پیش‌بینی",
        use_container_width=True
    )


# -------------------- پردازش و پیش‌بینی --------------------
if submitted:
    try:
        X_new = pd.DataFrame(
            [[values[f] for f in features]],
            columns=features
        )

        prediction = int(model.predict(X_new)[0])

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_new)[0]
            classes = list(
                getattr(model, "classes_", [0, 1])
            )

            if 1 in classes:
                probability = float(
                    probabilities[classes.index(1)]
                )

        st.divider()
        st.markdown("## 📊 نتیجه تحلیل")

        result_col, probability_col = st.columns(
            2,
            gap="large"
        )

        with result_col:
            if prediction == 1:
                st.warning(
                    "⚠️ خروجی مدل: کلاس ۱"
                )
                st.write(
                    "مدل این نمونه را در گروه دارای برچسب دیابت "
                    "قرار داده است."
                )
            else:
                st.success(
                    "✅ خروجی مدل: کلاس ۰"
                )
                st.write(
                    "مدل این نمونه را در گروه دارای برچسب بدون دیابت "
                    "قرار داده است."
                )

        with probability_col:
            if probability is not None:
                st.markdown("### 🎯 امتیاز مدل برای کلاس ۱")
                st.progress(
                    max(0.0, min(1.0, probability))
                )
                st.markdown(f"## {probability:.1%}")
                st.caption(
                    "این مقدار امتیاز احتمالی مدل است، نه احتمال قطعی "
                    "ابتلای فرد به دیابت."
                )
            else:
                st.info(
                    "این مدل قابلیت ارائه احتمال را ندارد؛ "
                    "فقط برچسب پیش‌بینی نمایش داده می‌شود."
                )

        # -------------------- نمودار --------------------
        if show_chart:
            st.markdown("### 📈 نمودار مقادیر ورودی")

            chart_df = pd.DataFrame({
                "ویژگی": [
                    labels.get(f, f) for f in features
                ],
                "مقدار": [
                    values[f] for f in features
                ]
            }).set_index("ویژگی")

            st.bar_chart(
                chart_df,
                color="#f97316"
            )

            st.caption(
                "ویژگی‌ها مقیاس‌های متفاوتی دارند؛ "
                "بزرگ‌تر بودن ستون به معنی اهمیت بیشتر آن نیست."
            )

        # -------------------- جزئیات --------------------
        if show_details:
            with st.expander(
                "🔎 مشاهده جزئیات ورودی و خروجی",
                expanded=False
            ):
                st.markdown("**اطلاعات ارسال‌شده به مدل**")

                st.dataframe(
                    X_new,
                    use_container_width=True,
                    hide_index=True
                )

                st.write("برچسب خروجی مدل:", prediction)

                if probability is not None:
                    st.write(
                        f"امتیاز کلاس ۱: {probability:.4f}"
                    )

        st.info(
            "این ابزار آموزشی است و برای تشخیص پزشکی یا تصمیم درمانی "
            "استفاده نشود. برای ارزیابی دیابت به پزشک و آزمایش‌های معتبر "
            "مراجعه کنید."
        )

    except Exception:
        st.error(
            "در پردازش اطلاعات خطایی رخ داد. "
            "ساختار ورودی‌ها و ویژگی‌های مدل را بررسی کنید."
        )
        st.exception(Exception)


# -------------------- پاورقی --------------------
st.markdown("""
<div class="footer">
    Diabetes AI · سامانه آموزشی تحلیل داده و یادگیری ماشین
    <br>
    طراحی‌شده با Python و Streamlit
</div>
""", unsafe_allow_html=True)
