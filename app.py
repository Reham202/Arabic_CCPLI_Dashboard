import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="مرصد القوة اللغوية العالمي CCPLI v2.0",
    page_icon="🌐",
    layout="wide",
)

# 2. مصفوفة الأبعاد العشرة
DIMENSIONS = [
    "D1: الرقمي والخوارزمي",
    "D2: تعليم اللغة L2",
    "D3: الخصائص والبناء",
    "D4: الديني والقداسي",
    "D5: الحضاري والتاريخي",
    "D6: المعرفي والعلمي",
    "D7: الاقتصادي",
    "D8: الديموغرافي",
    "D9: الجيوسياسي",
    "D10: الإعلامي",
]

DEFAULT_WEIGHTS = [0.14, 0.08, 0.10, 0.09, 0.08, 0.11, 0.12, 0.10, 0.10, 0.08]


# 3. دالة حساب القوة اللغوية المحدثة
def calculate_ccpli(scores, weights, m_momentum=1.05, m_geo=1.05):
    S = np.array(scores)
    W = np.array(weights)
    core = np.exp(np.sum(W * np.log(1 + 0.99 * S)))
    return round(core * m_momentum * m_geo, 2)


# 4. قاعدة البيانات المرجعية المحدثة
BENCHMARK_LANGUAGES = {
    "العربية": {
        "scores": [68.0, 78.0, 98.0, 98.0, 92.0, 62.0, 74.0, 82.0, 80.0, 72.0],
        "m_m": 1.085,
        "m_g": 1.0817,
    },
    "إنجليزية": {
        "scores": [98.5, 92.0, 75.0, 50.0, 80.0, 96.0, 94.0, 85.0, 92.0, 95.0],
        "m_m": 1.05,
        "m_g": 1.0638,
    },
    "الصينية": {
        "scores": [88.0, 60.0, 70.0, 55.0, 85.0, 88.0, 92.0, 96.0, 80.0, 75.0],
        "m_m": 1.03,
        "m_g": 1.0224,
    },
    "الفرنسية": {
        "scores": [76.0, 85.0, 74.0, 50.0, 82.0, 75.0, 68.0, 62.0, 88.0, 80.0],
        "m_m": 1.04,
        "m_g": 1.0654,
    },
    "الإسبانية": {
        "scores": [72.0, 80.0, 72.0, 60.0, 75.0, 68.0, 70.0, 84.0, 76.0, 78.0],
        "m_m": 1.03,
        "m_g": 1.0660,
    },
    "الروسية": {
        "scores": [78.0, 65.0, 75.0, 60.0, 88.0, 82.0, 70.0, 65.0, 86.0, 74.0],
        "m_m": 1.03,
        "m_g": 1.0450,
    },
    "الألمانية": {
        "scores": [80.0, 72.0, 70.0, 45.0, 86.0, 90.0, 88.0, 55.0, 82.0, 78.0],
        "m_m": 1.025,
        "m_g": 1.0350,
    },
    "اليابانية": {
        "scores": [85.0, 55.0, 68.0, 40.0, 82.0, 86.0, 84.0, 58.0, 70.0, 72.0],
        "m_m": 1.02,
        "m_g": 1.0280,
    },
}

# --- الواجهة التفاعلية ---
st.title("🌐 مرصد القوة الشاملة للغات العالمي (CCPLI v2.0)")
st.markdown(
    "**نظام محاكاة التنافسية الدولية مع التحكم الديناميكي بالأوزان وتصدير التقارير**"
)
st.markdown("---")

# الشريط الجانبي
st.sidebar.header("⚙️ إعدادات المحاكاة والأوزان")

# قسم تعديل الأوزان
with st.sidebar.expander("⚖️ تعديل الأوزان المعيارية للأبعاد (اختياري)"):
    st.caption("يمكنك تعديل أوزان الأبعاد لتقييم سيناريوهات مختلفة:")
    raw_weights = []
    for i, dim in enumerate(DIMENSIONS):
        w_val = st.slider(
            f"وزن {dim}",
            0.0,
            0.30,
            float(DEFAULT_WEIGHTS[i]),
            step=0.01,
            key=f"weight_{i}",
        )
        raw_weights.append(w_val)

    sum_w = sum(raw_weights)
    if sum_w > 0:
        active_weights = [w / sum_w for w in raw_weights]
    else:
        active_weights = DEFAULT_WEIGHTS

    st.info(f"المجموع الخام للأوزان: **{sum_w*100:.1f}%**")

# اختيار اللغة للمحاكاة
selected_lang = st.sidebar.selectbox(
    "اختر اللغة المراد محاكاة أبعادها:", list(BENCHMARK_LANGUAGES.keys())
)

st.sidebar.markdown(f"--- \n**تعديل أبعاد ({selected_lang}):**")

base_scores = BENCHMARK_LANGUAGES[selected_lang]["scores"]
user_scores = []

for i, dim in enumerate(DIMENSIONS):
    val = st.sidebar.slider(
        dim, 0.0, 100.0, float(base_scores[i]), step=1.0, key=f"score_{i}"
    )
    user_scores.append(val)

# حساب النتائج
results = {}
for lang, data in BENCHMARK_LANGUAGES.items():
    if lang == selected_lang:
        score = calculate_ccpli(
            user_scores, active_weights, data["m_m"], data["m_g"]
        )
        results[f"{lang} (المحاكاة)"] = score
    else:
        score = calculate_ccpli(
            data["scores"], active_weights, data["m_m"], data["m_g"]
        )
        results[lang] = score

# جدول الترتيب
df_ranks = pd.DataFrame(
    list(results.items()), columns=["اللغة", "الدرجة الكلية"]
).sort_values(by="الدرجة الكلية", ascending=False)
df_ranks["الترتيب"] = [f"#{i+1}" for i in range(len(df_ranks))]

# عرض المؤشرات
col1, col2, col3 = st.columns(3)
active_score = results[f"{selected_lang} (المحاكاة)"]
active_rank = df_ranks[df_ranks["اللغة"] == f"{selected_lang} (المحاكاة)"][
    "الترتيب"
].values[0]

col1.metric(f"درجة {selected_lang} المحاكاة", f"{active_score:.2f}")
col2.metric("الترتيب العالمي الحالي", active_rank)
col3.metric("عدد اللغات بالمقارنة", len(BENCHMARK_LANGUAGES))

st.markdown("---")

# الرسم الراداري
st.subheader("🕸️ التحليل الراداري والتنافسي")

default_compares = (
    [selected_lang, "إنجليزية", "الصينية"]
    if selected_lang != "إنجليزية"
    else ["إنجليزية", "العربية", "الصينية"]
)
compare_langs = st.multiselect(
    "اختر اللغات التي تريد إدراجها في الرسم الراداري:",
    options=list(BENCHMARK_LANGUAGES.keys()),
    default=[c for c in default_compares if c in BENCHMARK_LANGUAGES],
)

fig_radar = go.Figure()

for lang in compare_langs:
    if lang == selected_lang:
        r_vals = user_scores
        label = f"{lang} (المحاكاة)"
    else:
        r_vals = BENCHMARK_LANGUAGES[lang]["scores"]
        label = lang

    fig_radar.add_trace(
        go.Scatterpolar(r=r_vals, theta=DIMENSIONS, fill="toself", name=label)
    )

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True
)
st.plotly_chart(fig_radar, use_container_width=True)

# جدول الترتيب العالمي
st.subheader("📊 لائحة التنافسية الدولية الحالية")
st.dataframe(df_ranks[["الترتيب", "اللغة", "الدرجة الكلية"]], use_container_width=True)

# --- قسم تصدير البيانات ---
st.markdown("---")
st.subheader("📥 تصدير النتائج والتقارير")

csv_data = df_ranks[["الترتيب", "اللغة", "الدرجة الكلية"]].to_csv(
    index=False, encoding="utf-8-sig"
)

df_details = pd.DataFrame(
    {
        "البُعد": DIMENSIONS,
        "الوزن المعياري": [f"{w*100:.1f}%" for w in active_weights],
        "الدرجة الممنوحة": user_scores,
    }
)
csv_details = df_details.to_csv(index=False, encoding="utf-8-sig")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.download_button(
        label="📄 تنزيل جدول التنافسية العامة (CSV)",
        data=csv_data,
        file_name="CCPLI_Global_Rankings.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_exp2:
    st.download_button(
        label=f"📊 تنزيل تفاصيل أبعاد ({selected_lang})",
        data=csv_details,
        file_name=f"CCPLI_Details_{selected_lang}.csv",
        mime="text/csv",
        use_container_width=True,
    )
