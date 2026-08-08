

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="مرصد القوة اللغوية CCPLI v2.0", page_icon="🌐", layout="wide"
)

# الأوزان المعيارية للأبعاد العشرة
WEIGHTS = np.array([0.14, 0.08, 0.10, 0.09, 0.08, 0.11, 0.12, 0.10, 0.10, 0.08])
DIMENSIONS = [
    "D1: الرقمي والخوارزمي",
    "D2: تعليم العربية L2",
    "D3: الخصائص والبناء",
    "D4: الديني والقداسي",
    "D5: الحضاري والتاريخي",
    "D6: المعرفي والعلمي",
    "D7: الاقتصادي",
    "D8: الديموغرافي",
    "D9: الجيوسياسي",
    "D10: الإعلامي",
]


def calculate_ccpli(scores, m_momentum=1.085, m_geo=1.0817):
    S = np.array(scores)
    core = np.exp(np.sum(WEIGHTS * np.log(1 + 0.99 * S)))
    return round(core * m_momentum * m_geo, 2)


# واجهة التطبيق
st.title("🌐 مرصد القوة الشاملة للغات (CCPLI v2.0)")
st.markdown(
    "**النموذج الرقمي التفاعلي لاختبار أثر السياسات اللغوية والتخطيط الاستراتيجي**"
)
st.markdown("---")

# الشريط الجانبي
st.sidebar.header("⚙️ محاكاة أثر القرارات (العربية)")

user_scores = []
default_arabic = [68.0, 78.0, 98.0, 98.0, 92.0, 62.0, 74.0, 82.0, 80.0, 72.0]

for i, dim in enumerate(DIMENSIONS):
    val = st.sidebar.slider(dim, 0.0, 100.0, default_arabic[i], step=1.0)
    user_scores.append(val)

# حساب الدرجة
arabic_final = calculate_ccpli(user_scores)

# عرض المؤشرات
col1, col2 = st.columns(2)
col1.metric("درجة العربية المحاكاة", f"{arabic_final:.2f}")
col2.metric("الترتيب العالمي المتوقع", "#2")

st.markdown("---")

# الرسم البياني الراداري
st.subheader("🕸️ التحليل الراداري للأبعاد العشرة")
fig_radar = go.Figure()
fig_radar.add_trace(
    go.Scatterpolar(
        r=user_scores, theta=DIMENSIONS, fill="toself", name="العربية (المحاكاة)"
    )
)
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True
)
st.plotly_chart(fig_radar, use_container_width=True)