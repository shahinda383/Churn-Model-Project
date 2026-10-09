import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go

# --------------------------- Page setup ---------------------------
st.set_page_config(
    page_title="ChurnSense AI | Customer Retention",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "best_churn_model.pkl"

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
if "lang" not in st.session_state:
    st.session_state.lang = "EN"

# --------------------------- Language / theme ---------------------------
with st.sidebar:
    st.markdown("## ✦ ChurnSense AI")
    st.caption("CUSTOMER RETENTION INTELLIGENCE")
    lang = st.radio("Language / اللغة", ["EN", "AR"], horizontal=True,
                    index=0 if st.session_state.lang == "EN" else 1)
    st.session_state.lang = lang
    theme = st.radio("Appearance / المظهر", ["Dark", "Light"], horizontal=True,
                     index=0 if st.session_state.theme == "Dark" else 1)
    st.session_state.theme = theme
    st.divider()
    st.markdown("**Navigation / التنقل**")
    page = st.radio(
        "Go to",
        ["Overview", "Predict Churn", "Batch Analysis", "About the Model"]
        if lang == "EN" else
        ["نظرة عامة", "توقع عميل", "تحليل ملف", "عن الموديل"],
        label_visibility="collapsed",
    )

is_ar = lang == "AR"
is_dark = theme == "Dark"
bg = "#0b1020" if is_dark else "#f4f7fb"
panel = "#131c31" if is_dark else "#ffffff"
text = "#edf2ff" if is_dark else "#172033"
muted = "#a5b2cc" if is_dark else "#64748b"
border = "#273653" if is_dark else "#e2e8f0"
accent = "#7c5cff"
accent2 = "#21c8a8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Cairo:wght@400;600;700;800&display=swap');
:root {{ color-scheme: {'dark' if is_dark else 'light'}; }}
.stApp {{ background: radial-gradient(ellipse at top left, {'#18284b' if is_dark else '#e0eaff'} 0%, {bg} 48%); color: {text}; }}
html, body, [class*="css"] {{ font-family: {'Cairo' if is_ar else 'Inter'}, sans-serif; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1450px; }}
section[data-testid="stSidebar"] {{ background: {panel}; border-right: 1px solid {border}; }}
section[data-testid="stSidebar"] * {{ color: {text}; }}
.hero {{ padding: 30px 32px; border: 1px solid {border}; border-radius: 24px;
background: linear-gradient(125deg, {'#172746' if is_dark else '#ffffff'} 0%, {'#241c4c' if is_dark else '#eeeaff'} 100%);
box-shadow: 0 18px 55px rgba(30,50,100,.13); margin-bottom: 20px; }}
.hero h1 {{ font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.08; margin: 0 0 12px 0; font-weight: 800; letter-spacing: -1.4px; color: {text}; }}
.hero p {{ color: {muted}; font-size: 1.05rem; margin: 0; }}
.eyebrow {{ color: {accent2}; font-weight: 800; letter-spacing: 2px; font-size: .76rem; margin-bottom: 10px; }}
.metric-card {{ background: {panel}; border: 1px solid {border}; border-radius: 18px; padding: 20px 22px; min-height: 120px; }}
.metric-label {{ color: {muted}; font-size: .86rem; font-weight: 600; }}
.metric-value {{ color: {text}; font-size: 1.8rem; font-weight: 800; margin-top: 7px; }}
.metric-note {{ color: {muted}; font-size: .8rem; margin-top: 4px; }}
.section-title {{ font-size: 1.25rem; font-weight: 800; color: {text}; margin: 1.1rem 0 .7rem 0; }}
.info-card {{ background: {panel}; border: 1px solid {border}; border-radius: 18px; padding: 20px; }}
div[data-testid="stForm"] {{ background: {panel}; border: 1px solid {border}; border-radius: 18px; padding: 22px; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 12px; border: 0; font-weight: 700; min-height: 2.7rem; }}
div[data-testid="stProgressBar"] > div > div {{ background: linear-gradient(90deg, {accent}, {accent2}); }}
.small-muted {{ color: {muted}; font-size: .88rem; }}
</style>
""", unsafe_allow_html=True)

def tr(en, ar):
    return ar if is_ar else en

@st.cache_resource
def load_bundle(path, mtime):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.markdown(f"""
    <div class="hero">
      <div class="eyebrow">CHURNSENSE AI · MODEL SETUP</div>
      <h1>{tr("Your retention intelligence workspace", "منصة ذكية لفهم فقدان العملاء")}</h1>
      <p>{tr("Place best_churn_model.pkl in the same folder as app.py, then restart the app.",
              "حطي ملف best_churn_model.pkl في نفس فولدر app.py وبعدها شغّلي التطبيق.")}</p>
    </div>""", unsafe_allow_html=True)
    st.error(tr("Model file not found.", "ملف الموديل مش موجود."))
    st.code("project_folder/\n├── app.py\n└── best_churn_model.pkl")
    st.stop()

bundle = load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
model = bundle["model"]
threshold = float(bundle.get("threshold", 0.5))
model_name = bundle.get("model_name", "Selected model")
features = bundle.get("feature_names", [])

def add_engineered_features(row):
    row = row.copy()
    # Keep these transformations identical to the notebook's feature engineering.
    if "Age" in row:
        row["AgeGroup"] = pd.cut(
            pd.to_numeric(row["Age"], errors="coerce"),
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        ).astype(str)
    if "Balance" in row:
        row["HasBalance"] = (pd.to_numeric(row["Balance"], errors="coerce") > 0).astype(int)
    if {"Balance", "NumOfProducts"}.issubset(row.columns):
        row["BalancePerProduct"] = row["Balance"] / row["NumOfProducts"].replace(0, 1)
    if {"Age", "IsActiveMember"}.issubset(row.columns):
        row["AgeActive"] = row["Age"] * row["IsActiveMember"]
    if "NumOfProducts" in row:
        row["ProductGroup"] = row["NumOfProducts"].apply(lambda x: "High" if x >= 3 else str(int(x)))
    if {"Balance", "EstimatedSalary"}.issubset(row.columns):
        row["BalanceSalaryRatio"] = row["Balance"] / (row["EstimatedSalary"] + 1)
    if {"Tenure", "Age"}.issubset(row.columns):
        row["TenureByAge"] = row["Tenure"] / row["Age"].replace(0, 1)
    if {"CreditScore", "Age"}.issubset(row.columns):
        row["CreditScoreByAge"] = row["CreditScore"] / row["Age"].replace(0, 1)
    if {"Age", "IsActiveMember"}.issubset(row.columns):
        row["InactiveSenior"] = ((row["Age"] >= 45) & (row["IsActiveMember"] == 0)).astype(int)
    if {"Geography", "HasBalance"}.issubset(row.columns):
        row["GermanyBalance"] = ((row["Geography"] == "Germany") & (row["HasBalance"] == 1)).astype(int)
    # Restore exact feature order from training.
    return row.reindex(columns=features)

def predict_row(raw):
    X = add_engineered_features(pd.DataFrame([raw]))
    prob = float(model.predict_proba(X)[:, 1][0])
    return prob, int(prob >= threshold)

def hero(title, subtitle, eyebrow=""):
    direction = "rtl" if is_ar else "ltr"
    st.markdown(f'<div class="hero" dir="{direction}"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)
def metric(label, value, note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

# --------------------------- Overview ---------------------------
if page in ["Overview", "نظرة عامة"]:
    hero(tr("See churn before it happens.", "اكتشف احتمالية فقدان العميل قبل ما يمشي."),
         tr("A decision-support dashboard that turns customer data into clear retention priorities.",
            "لوحة ذكية بتحوّل بيانات العملاء لمؤشرات واضحة تساعدك تحدد مين محتاج تدخل سريع."))
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric(tr("Selected model", "الموديل المختار"), model_name, tr("Loaded and ready", "جاهز للاستخدام"))
    with c2: metric(tr("Decision threshold", "حد القرار"), f"{threshold:.2f}", tr("Calibrated in training", "تم اختياره أثناء التدريب"))
    with c3: metric(tr("Input features", "عدد الخصائص"), str(len(features)), tr("Expected model inputs", "الخصائص المطلوبة للموديل"))
    with c4: metric(tr("Prediction mode", "نوع التوقع"), tr("Risk scoring", "تقييم المخاطر"), tr("Probability + decision", "احتمال + قرار"))
    st.markdown(f'<div class="section-title">{tr("What you can do", "تقدر تعمل إيه؟")}</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a:
        st.markdown(f'<div class="info-card"><h3>◈ {tr("Individual scoring", "تقييم عميل")}</h3><p class="small-muted">{tr("Enter a customer profile and get a churn probability, risk band, and recommended next action.", "أدخل بيانات عميل واعرف احتمال المغادرة ومستوى الخطورة والخطوة المقترحة.")}</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="info-card"><h3>▦ {tr("Batch analysis", "تحليل مجموعة")}</h3><p class="small-muted">{tr("Upload a CSV to score many customers and download a prioritised list.", "ارفع CSV لتقييم مجموعة عملاء وتنزيل قائمة مرتبة حسب الأولوية.")}</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown(f'<div class="info-card"><h3>✧ {tr("Explainable workflow", "نتائج واضحة")}</h3><p class="small-muted">{tr("Review risk bands and understand that predictions support decisions, not guarantee outcomes.", "راجع مستويات الخطورة وافتكر إن التوقع بيساعد القرار لكنه مش ضمان للنتيجة.")}</p></div>', unsafe_allow_html=True)
    st.info(tr("Use predictions to prioritise human follow-up. Validate performance on current business data before operational use.",
               "استخدم النتائج لترتيب أولوية التواصل مع العملاء، وراجع أداء الموديل على بيانات حديثة قبل الاعتماد التشغيلي."))

# --------------------------- Individual prediction ---------------------------
elif page in ["Predict Churn", "توقع عميل"]:
    hero(tr("Customer risk studio", "مساحة تقييم مخاطر العملاء"),
         tr("Build a customer profile and receive an instant, threshold-aware prediction.",
            "كوّن ملف العميل واحصل على توقع فوري باستخدام حد القرار المختار أثناء التدريب."))
    with st.form("customer_form"):
        st.markdown(f"### {tr('Customer profile', 'بيانات العميل')}")
        c1,c2,c3 = st.columns(3)
        with c1:
            credit = st.slider(tr("Credit score", "التقييم الائتماني"), 350, 850, 650)
            geography = st.selectbox(tr("Country", "الدولة"), ["France", "Germany", "Spain"])
            gender = st.selectbox(tr("Gender", "النوع"), ["Female", "Male"])
            age = st.slider(tr("Age", "العمر"), 18, 92, 38)
        with c2:
            tenure = st.slider(tr("Years with the bank", "سنوات التعامل مع البنك"), 0, 10, 5)
            balance = st.number_input(tr("Account balance", "رصيد الحساب"), min_value=0.0, value=75000.0, step=5000.0)
            products = st.selectbox(tr("Number of products", "عدد المنتجات"), [1,2,3,4], index=1)
            salary = st.number_input(tr("Estimated annual salary", "الراتب السنوي المتوقع"), min_value=0.0, value=65000.0, step=5000.0)
        with c3:
            credit_card = st.selectbox(tr("Has credit card?", "هل لديه بطاقة ائتمان؟"), [1,0],
                                       format_func=lambda x: tr("Yes" if x else "No", "أيوه" if x else "لأ"))
            active = st.selectbox(tr("Active member?", "عميل نشط؟"), [1,0],
                                  format_func=lambda x: tr("Yes" if x else "No", "أيوه" if x else "لأ"))
            st.caption(tr("All fields are illustrative customer inputs.", "كل البيانات دي مدخلات توضيحية لملف العميل."))
        submitted = st.form_submit_button(tr("✦ Analyse customer", "✦ حلّل العميل"), use_container_width=True)

    if submitted:
        raw = {
            "CreditScore": credit, "Geography": geography, "Gender": gender,
            "Age": age, "Tenure": tenure, "Balance": balance,
            "NumOfProducts": products, "HasCrCard": credit_card,
            "IsActiveMember": active, "EstimatedSalary": salary
        }
        probability, prediction = predict_row(raw)
        risk = "High" if probability >= max(threshold, 0.5) else ("Medium" if probability >= threshold * 0.65 else "Low")
        risk_ar = {"High":"مرتفع", "Medium":"متوسط", "Low":"منخفض"}[risk]
        r1,r2,r3 = st.columns([1.2,1,1])
        with r1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{tr("Churn probability", "احتمال مغادرة العميل")}</div><div class="metric-value">{probability:.1%}</div><div class="metric-note">{tr("Model-estimated probability", "احتمال مقدّر بواسطة الموديل")}</div></div>', unsafe_allow_html=True)
            st.progress(float(np.clip(probability,0,1)))
        with r2:
            metric(tr("Risk level", "مستوى الخطورة"), risk if not is_ar else risk_ar, tr("Relative risk band", "تصنيف إرشادي للمخاطر"))
        with r3:
            metric(tr("Model decision", "قرار الموديل"),
                   tr("Likely to leave" if prediction else "Likely to stay",
                      "مرشح للمغادرة" if prediction else "مرشح للاستمرار"),
                   f"{tr('Threshold', 'حد القرار')}: {threshold:.2f}")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=probability*100,
            number={"suffix":"%", "font":{"size":30}},
            title={"text":tr("Churn risk score", "مؤشر خطر المغادرة")},
            gauge={"axis":{"range":[0,100]}, "bar":{"color":accent},
                   "steps":[{"range":[0,30],"color":"#d9f8ef"},
                            {"range":[30,60],"color":"#fff0c2"},
                            {"range":[60,100],"color":"#ffd9df"}],
                   "threshold":{"line":{"color":"#ff5470","width":4},"thickness":0.8,"value":threshold*100}}
        ))
        fig.update_layout(height=300, margin=dict(l=20,r=20,t=55,b=10),
                          paper_bgcolor=panel, font_color=text)
        st.plotly_chart(fig, use_container_width=True)
        if prediction:
            st.warning(tr("Suggested action: prioritise a personal retention contact, review service fit, and offer a relevant plan—not an automatic discount.",
                          "الخطوة المقترحة: ابدأ بتواصل شخصي، وراجع مناسبة الخدمات للعميل، وقدّم عرض مناسب بدل خصم تلقائي."))
        else:
            st.success(tr("Suggested action: maintain engagement and monitor future changes; low predicted risk does not mean zero risk.",
                          "الخطوة المقترحة: حافظ على التواصل وراقب أي تغيّر مستقبلي؛ انخفاض الخطر لا يعني إنه صفر."))

# --------------------------- Batch analysis ---------------------------
elif page in ["Batch Analysis", "تحليل ملف"]:
    hero(tr("Portfolio scanner", "ماسح محفظة العملاء"),
         tr("Upload customer records, score churn risk, and export an actionable priority list.",
            "ارفع بيانات العملاء، واحسب احتمالات المغادرة، ونزّل قائمة مرتبة حسب الأولوية."))
    st.markdown(tr("Upload a CSV with the original bank-customer columns. Target column `Exited` is optional and will not be used for prediction.",
                   "ارفع CSV بأعمدة بيانات عملاء البنك الأصلية. عمود Exited اختياري ولن يُستخدم في التوقع."))
    uploaded = st.file_uploader(tr("Choose CSV file", "اختار ملف CSV"), type=["csv"])
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            st.write(tr("Preview", "معاينة البيانات"), data.head())
            expected = ["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
            missing = [col for col in expected if col not in data.columns]
            if missing:
                st.error(tr(f"Missing required columns: {', '.join(missing)}",
                            f"الأعمدة المطلوبة الناقصة: {', '.join(missing)}"))
            else:
                if st.button(tr("Analyse all customers", "حلّل كل العملاء"), type="primary"):
                    base = data.drop(columns=["Exited"], errors="ignore").copy()
                    rows = []
                    for _, row in base.iterrows():
                        p, pred = predict_row(row.to_dict())
                        rows.append((p, pred))
                    data["ChurnProbability"] = [x[0] for x in rows]
                    data["PredictedExited"] = [x[1] for x in rows]
                    data["RiskBand"] = pd.cut(data["ChurnProbability"], bins=[-0.001,0.3,0.6,1.001],
                                               labels=["Low","Medium","High"]).astype(str)
                    st.success(tr(f"Scored {len(data):,} customers.", f"تم تقييم {len(data):,} عميل."))
                    k1,k2,k3 = st.columns(3)
                    with k1: metric(tr("High risk", "خطورة مرتفعة"), f"{(data['RiskBand']=='High').sum():,}")
                    with k2: metric(tr("Predicted churn", "متوقع مغادرتهم"), f"{data['PredictedExited'].sum():,}")
                    with k3: metric(tr("Average risk", "متوسط الاحتمال"), f"{data['ChurnProbability'].mean():.1%}")
                    fig = go.Figure(go.Histogram(x=data["ChurnProbability"], nbinsx=20))
                    fig.update_layout(title=tr("Customer risk distribution", "توزيع مخاطر العملاء"),
                                      xaxis_title=tr("Churn probability", "احتمال المغادرة"),
                                      yaxis_title=tr("Customers", "عدد العملاء"),
                                      paper_bgcolor=panel, plot_bgcolor=panel, font_color=text)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(data.sort_values("ChurnProbability", ascending=False), use_container_width=True)
                    st.download_button(tr("Download scored CSV", "تنزيل ملف النتائج CSV"),
                        data.to_csv(index=False).encode("utf-8-sig"),
                        file_name="churn_risk_results.csv", mime="text/csv", type="primary")
        except Exception as e:
            st.error(tr(f"Could not read this file: {e}", f"مقدرناش نقرأ الملف: {e}"))

# --------------------------- About ---------------------------
elif page in ["About the Model", "عن الموديل"]:
    hero(tr("Know what powers the prediction.", "اعرف إيه اللي ورا التوقع."),
         tr("Transparent model metadata and practical guidance for responsible use.",
            "معلومات الموديل وإرشادات عملية للاستخدام المسؤول."))
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="info-card"><h3>{tr("Model details", "تفاصيل الموديل")}</h3><p><b>{tr("Algorithm", "الخوارزمية")}:</b> {model_name}</p><p><b>{tr("Threshold", "حد القرار")}:</b> {threshold:.3f}</p><p><b>{tr("Input features", "عدد الخصائص")}:</b> {len(features)}</p><p><b>{tr("Target", "الهدف")}:</b> Exited (1 = left)</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="info-card"><h3>{tr("Responsible use", "الاستخدام المسؤول")}</h3><p class="small-muted">{tr("Predictions reflect patterns in historical data and may be wrong. Use them to support—not replace—human judgement. Check data quality, monitor drift, and evaluate fairness before deployment.", "التوقعات مبنية على أنماط في بيانات سابقة وممكن تكون غلط. استخدمها لدعم القرار مش استبدال الحكم البشري، وراجع جودة البيانات وتغيرها وعدالة النتائج قبل التشغيل الفعلي.")}</p></div>', unsafe_allow_html=True)
    with st.expander(tr("Features expected by the model", "الخصائص المطلوبة للموديل")):
        st.write(features)
    st.caption(tr("Model file is loaded locally from the same folder as app.py.", "يتم تحميل ملف الموديل محليًا من نفس فولدر app.py."))

