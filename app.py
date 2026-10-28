
import streamlit as st
import pandas as pd
import joblib

# -------------------- تنظیمات صفحه --------------------
st.set_page_config(
    page_title="Diabetes AI | تحلیل سلامت",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- طراحی ظاهری --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');

/* فونت وزیر و راست‌چین شدن کل برنامه */
html, body, [class*="css"],
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"],
input, textarea, button, label {
    font-family: 'Vazirmatn', sans-serif !important;
}

html, body, [data-testid="stApp"] {
    direction: rtl;
    text-align: right;
}

[data-testid="stAppViewContainer"] {
    direction: rtl;
}

[data-testid="stMain"] {
    direction: rtl;
    text-align: right;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] li {
    direction: rtl;
    text-align: right;
}

/* سایدبار */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff0cc 0%, #fff8e8 100%);
    border-left: 1px solid #f5d69a;
    border-right: none;
}

[data-testid="stSidebar"] > div {
    direction: rtl;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small {
    color: #422b16 !important;
    text-align: right;
    direction: rtl;
}

/* متن توضیحات */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #76532d !important;
}

/* چک‌باکس‌ها */
[data-testid="stSidebar"] [data-testid="stCheckbox"] {
    direction: rtl;
}

[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    display: flex;
    flex-direction: row-reverse;
    justify-content: flex-end;
    gap: 10px;
    color: #422b16 !important;
}

/* ورودی‌ها و انتخابگرها */
input, textarea {
    direction: rtl !important;
    text-align: right !important;
}

[data-baseweb="select"] > div {
    direction: rtl;
}

[data-testid="stNumberInput"] button {
    color: #b45309;
}

/* دکمه‌ها */
.stButton > button,
.stFormSubmitButton > button {
    font-family: 'Vazirmatn', sans-serif !important;
    font-weight: 700;
}

/* کارت‌های داشبورد */
div[data-testid="stMetric"] {
    direction: rtl;
    text-align: right;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"] {
    text-align: right;
}

/* هشدارها و پیام‌ها */
[data-testid="stAlert"] {
    direction: rtl;
    text-align: right;
}

[data-testid="stAlert"] p {
    text-align: right;
}

/* جدول */
[data-testid="stDataFrame"] {
    direction: rtl;
}

/* جلوگیری از به‌هم‌ریختگی اعداد و کدهای انگلیسی */
code, pre {
    direction: ltr;
    text-align: left;
}

/* موبایل */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
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
except Exception as e:
    st.error(
        "مدل بارگذاری نشد. مطمئن شو فایل "
        "`diabetes_model.joblib` در مسیر پروژه وجود دارد."
    )
    st.exception(e)
    st.stop()


# -------------------- مشخصات ویژگی‌ها --------------------
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
    "Glucose": "معمولاً بر اساس mg/dL",
    "BloodPressure": "معمولاً بر اساس mmHg",
    "SkinThickness": "معمولاً بر اساس میلی‌متر",
    "Insulin": "معمولاً بر اساس μU/mL",
    "BMI": "وزن تقسیم بر مجذور قد بر حسب متر",
    "DiabetesPedigreeFunction": "شاخص عددی موجود در دیتاست",
    "Age": "سن بر حسب سال"
}


# -------------------- نوار کناری --------------------
with st.sidebar:
    st.markdown("## 🩺 Diabetes AI")
    st.caption("پنل تحلیل و پیش‌بینی آزمایشی")
    st.divider()

    st.markdown("### ⚙️ تنظیمات نمایش")

    show_chart = st.checkbox(
        "نمایش نمودار اطلاعات",
        value=True
    )

    show_details = st.checkbox(
        "نمایش جزئیات مدل",
        value=True
    )

    show_help = st.checkbox(
        "نمایش راهنمای ورودی‌ها",
        value=True
    )

    st.divider()
    st.markdown("### 📌 درباره سامانه")
    st.write("نسخه آموزشی رابط پیش‌بینی دیابت")
    st.write(f"تعداد ویژگی‌های ورودی: {len(features)}")

    st.warning(
        "این سامانه ابزار تشخیص پزشکی نیست."
    )


# -------------------- سربرگ --------------------
st.markdown("""
<div class="hero">
    <h1>🩺 سامانه هوشمند پیش‌بینی دیابت</h1>
    <p>
        اطلاعات نمونه را وارد کن، ورودی‌ها را بررسی کن
        و خروجی مدل یادگیری ماشین را مشاهده کن.
        <br>
        طراحی مدرن با رابط کاربری نارنجی و زرد
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------- کارت‌های داشبورد --------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="ویژگی‌های ورودی",
        value=len(features),
        help="تعداد متغیرهای مورد استفاده مدل"
    )

with col2:
    st.metric(
        label="وضعیت مدل",
        value="آماده" if model else "نامشخص"
    )

with col3:
    st.metric(
        label="نوع سامانه",
        value="آموزشی"
    )


st.write("")

st.markdown("""
<div class="section-card">
    <div class="section-title">📋 اطلاعات فرد مورد بررسی</div>
    <div class="section-subtitle">
        مقادیر را مطابق داده‌های مورد نیاز مدل وارد کن.
        در صورت استفاده از داده واقعی، واحدها باید با دیتاست آموزش یکسان باشند.
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------- فرم ورودی --------------------
with st.form("prediction_form", clear_on_submit=False):

    values = {}

    # تقسیم ورودی‌ها در دو ستون
    left_col, right_col = st.columns(2, gap="large")

    # گروه‌بندی براساس ویژگی‌های رایج دیتاست دیابت
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

    # ویژگی‌های اضافه یا سفارشی مدل
    other_features = [
        f for f in features
        if f not in personal_features + lab_features
    ]

    with right_col:
        st.markdown("#### 👤 مشخصات فردی و بدنی")

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
        st.markdown("#### 🧪 اطلاعات آزمایشگاهی")

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
        st.markdown("#### ➕ سایر ویژگی‌های مدل")

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


# -------------------- پیش‌بینی --------------------
if submitted:
    try:
        # ترتیب ستون‌ها دقیقاً مطابق آموزش مدل
        X_new = pd.DataFrame(
            [[values[f] for f in features]],
            columns=features
        )

        prediction = int(model.predict(X_new)[0])

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_new)[0]
            classes = list(getattr(model, "classes_", [0, 1]))

            # پیدا کردن احتمال کلاس 1 به جای فرض کردن اندیس آن
            if 1 in classes:
                probability = float(probabilities[classes.index(1)])

        st.divider()
        st.markdown("## 📊 نتیجه تحلیل")

        result_col, probability_col = st.columns(2, gap="large")

        with result_col:
            if prediction == 1:
                st.warning(
                    "⚠️ خروجی مدل: کلاس ۱",
                    icon="⚠️"
                )
                st.markdown(
                    "مدل این نمونه را در گروه دارای برچسب دیابت "
                    "قرار داده است."
                )
            else:
                st.success(
                    "✅ خروجی مدل: کلاس ۰",
                    icon="✅"
                )
                st.markdown(
                    "مدل این نمونه را در گروه دارای برچسب بدون دیابت "
                    "قرار داده است."
                )

        with probability_col:
            if probability is not None:
                st.markdown("#### 🎯 امتیاز خروجی مدل")
                st.progress(
                    max(0.0, min(1.0, probability))
                )
                st.markdown(
                    f"### {probability:.1%}"
                )
                st.caption(
                    "احتمال تخمینی مدل برای کلاس ۱؛ "
                    "نه احتمال قطعی ابتلای فرد."
                )
            else:
                st.info(
                    "این مدل قابلیت ارائه خروجی احتمال را ندارد؛ "
                    "فقط برچسب پیش‌بینی نمایش داده شده است."
                )

        # -------------------- نمودار ورودی‌ها --------------------
        if show_chart:
            st.markdown("### 📈 نمودار مقادیر واردشده")

            chart_df = pd.DataFrame({
                "ویژگی": [labels.get(f, f) for f in features],
                "مقدار": [values[f] for f in features]
            }).set_index("ویژگی")

            st.bar_chart(
                chart_df,
                color="#f59e0b"
            )

            st.caption(
                "مقیاس ویژگی‌ها متفاوت است؛ "
                "بلندتر بودن یک ستون به معنی اهمیت بیشتر آن نیست."
            )

        # -------------------- جزئیات --------------------
        if show_details:
            with st.expander("🔎 مشاهده جزئیات ورودی و خروجی"):
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
            "این نتیجه فقط خروجی یک مدل یادگیری ماشین است. "
            "برای تشخیص دیابت یا تصمیم درمانی، باید به پزشک "
            "و آزمایش‌های معتبر مراجعه کرد."
        )

    except Exception as e:
        st.error(
            "در پردازش اطلاعات خطایی رخ داد. "
            "ساختار ورودی و ویژگی‌های مدل را بررسی کن."
        )
        st.exception(e)


# -------------------- پاورقی --------------------
st.markdown("""
<div class="footer">
    Diabetes AI · سامانه آموزشی تحلیل داده و یادگیری ماشین
    <br>
    ساخته‌شده با Python و Streamlit
</div>
""", unsafe_allow_html=True)
