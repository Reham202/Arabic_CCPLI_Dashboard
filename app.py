import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. إعدادات الصفحة والتصميم العامة (Page Config & CSS)
# ==========================================
st.set_page_config(
    page_title="مرصد المؤشر الشامل لقوة اللغة (CCPLI v4.0)",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تطبيق تنسيقات CSS لدعم الاتجاه العربي والتصميم العصري
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* بطاقات الإحصائيات */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-right: 5px solid #1E3A8A;
        margin-bottom: 15px;
    }
    
    .metric-title {
        color: #6B7280;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .metric-value {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 5px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_content_policy=True)

# ==========================================
# 2. بناء قواعد البيانات وهيكلية الجيل الرابع (v4.0)
# ==========================================
DIMENSIONS = [
    "البُعد الديموغرافي والجغرافي",
    "البُعد الاقتصادي والتجاري",
    "البُعد الأكاديمي والبحثي",
    "البُعد الرقمي والتقني",
    "البُعد الدبلوماسي والمؤسسي",
    "البُعد الثقافي والإعلامي",
    "البُعد التعليمي (تعليم اللغة لغير الناطقين)",
    "البُعد الترجمي والنشر",
    "البُعد المعياري والتشريعي",
    "البُعد الابتكاري والذكاء الاصطناعي"
]

SUB_INDICATORS = {
    dim: [f"مؤشر فرعي {i+1}: {dim.split()[1]} {i+1}" for i in range(5)]
    for dim in DIMENSIONS
}

@st.cache_data
def load_ccpli_v4_data():
    """توليد هيكل بيانات معياري محاكي للإصدار الرابع CCPLI v4.0"""
    languages = ["العربية", "الإنجليزية", "الفرنسية", "الصينية", "الإسبانية"]
    
    # أوزان الأبعاد العشرة (تساوي 100%)
    weights = [0.10, 0.15, 0.12, 0.12, 0.08, 0.10, 0.10, 0.08, 0.07, 0.08]
    
    # درجات افتراضية أساسية
    np.random.seed(42)
    
    dims_data = []
    sub_data = []
    summary_data = []
    
    base_scores = {
        "الإنجليزية": 92.5,
        "العربية": 86.4,
        "الصينية": 82.1,
        "الفرنسية": 78.3,
        "الإسبانية": 74.0
    }
    
    for lang in languages:
        total_ccpli = 0
        base = base_scores[lang]
        
        for d_idx, dim in enumerate(DIMENSIONS):
            # درجة البُعد من 100
            dim_score = np.clip(base + np.random.normal(0, 5), 40, 100)
            weight = weights[d_idx]
            dim_contrib = dim_score * weight
            total_ccpli += dim_contrib
            
            dims_data.append({
                "lang_name": lang,
                "dim_name": dim,
                "dim_score": round(dim_score, 2),
                "weight": weight,
                "dim_contribution": round(dim_contrib, 2)
            })
            
            # المؤشرات الفرعية الخمسة لكل بُعد
            for s_idx, sub_name in enumerate(SUB_INDICATORS[dim]):
                sub_score = np.clip(dim_score + np.random.normal(0, 3), 30, 100)
                sub_data.append({
                    "lang_name": lang,
                    "dim_name": dim,
                    "sub_indicator_name": f"{dim} - مؤشر فرعي {s_idx+1}",
                    "sub_score": round(sub_score, 2)
                })
                
        # تحديد الفئة المعيارية بناءً على النتيجة الكلية
        if total_ccpli >= 80:
            tier = "الفئة العالمية المتقدمة (Advanced Global Tier)"
        elif total_ccpli >= 60:
            tier = "الفئة الإقليمية العالية (High Regional Tier)"
        elif total_ccpli >= 40:
            tier = "الفئة المتوسطة (Intermediate Tier)"
        else:
            tier = "الفئة النامية (Developing Tier)"
            
        summary_data.append({
            "lang_name": lang,
            "ccpli_score": round(total_ccpli, 2),
            "tier": tier
        })
        
    return pd.DataFrame(summary_data), pd.DataFrame(dims_data), pd.DataFrame(sub_data)

ccpli_summary, ccpli_dims, ccpli_subs = load_ccpli_v4_data()

# ==========================================
# 3. القائمة الجانبية وعناصر التحكم (Sidebar Controls)
# ==========================================
st.sidebar.title("🌐 مرصد CCPLI v4.0")
st.sidebar.markdown("**الإصدار الرابع من المؤشر الشامل**")
st.sidebar.caption("نظام قياس قوة ونفوذ اللغات الكونية")
st.sidebar.divider()

selected_lang = st.sidebar.selectbox(
    "اختر اللغة للتحليل التفصيلي:",
    ccpli_summary['lang_name'].unique(),
    index=1 # افتراضياً اللغة العربية
)

st.sidebar.info("""
**مكونات الإطار النظري (v4.0):**
- **10** أبعاد رئيسية متوازنة.
- **5** مؤشرات فرعية لكل بُعد (50 مؤشراً).
- قياس مركب يجمع بين القوة الصلبة والناعمة.
""")

# ==========================================
# 4. الواجهة الرئيسية واللوحات البيانية (Main Content)
# ==========================================
st.title("🏛️ مرصد المؤشر الشامل لقوة اللغة (CCPLI)")
st.markdown("لوحة تحكم تفاعلية لاستكشاف وتقييم الأبعاد العشرة والمؤشرات الفرعية للغات.")

# بطاقات ملخص النتيجة للغة المختارة
lang_summary = ccpli_summary[ccpli_summary['lang_name'] == selected_lang].iloc[0]

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">اللغة المختارة</div>
        <div class="metric-value">{selected_lang}</div>
    </div>
    """, unsafe_content_policy=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">النتيجة الكلية للمؤشر</div>
        <div class="metric-value">{lang_summary['ccpli_score']} / 100</div>
    </div>
    """, unsafe_content_policy=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">التصنيف المعياري العالمي</div>
        <div class="metric-value" style="font-size: 1.2rem; color: #1E3A8A;">{lang_summary['tier']}</div>
    </div>
    """, unsafe_content_policy=True)

st.divider()

# تبويبات العرض التحليلي
tab1, tab2, tab3 = st.tabs([
    "📊 لوحة النتائج التوضيحية", 
    "🔬 تفكيك الأبعاد والمؤشرات الفرعية (50 مؤشراً)", 
    "📈 المقارنة الرادارية والمصفوفة الكلية"
])

# ------------------------------------------
# التبويب الأول: لوحة النتائج التوضيحية (Gauge & Contribution Bar)
# ------------------------------------------
with tab1:
    st.subheader(f"التحليل التوضيحي المباشر لنتائج: {selected_lang}")
    
    col_g, col_b = st.columns([1, 1.3])
    
    # 1. Gauge Chart - العداد
    with col_g:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=lang_summary['ccpli_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "مقياس الدرجة الكلية (CCPLI)", 'font': {'size': 18, 'family': 'Tajawal'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1E3A8A"},
                'bar': {'color': "#1E3A8A"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E5E7EB",
                'steps': [
                    {'range': [0, 40], 'color': '#FEE2E2'},   # الفئة النامية
                    {'range': [40, 60], 'color': '#FEF3C7'},  # الفئة المتوسطة
                    {'range': [60, 80], 'color': '#E0F2FE'},  # الفئة الإقليمية العالية
                    {'range': [80, 100], 'color': '#DCFCE7'}  # الفئة العالمية المتقدمة
                ],
            }
        ))
        fig_gauge.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # 2. Horizontal Contribution Bar Chart
    with col_b:
        lang_dims = ccpli_dims[ccpli_dims['lang_name'] == selected_lang].sort_values(by='dim_contribution', ascending=True)
        
        fig_contrib = px.bar(
            lang_dims,
            x='dim_contribution',
            y='dim_name',
            orientation='h',
            title="نقاط إسهام الأبعاد الرئيسية في النتيجة الكلية",
            labels={'dim_contribution': 'النقاط المكتسبة', 'dim_name': ''},
            color='dim_score',
            color_continuous_scale='Blues',
            text_auto='.2f'
        )
        fig_contrib.update_layout(
            height=360,
            margin=dict(l=10, r=20, t=50, b=20),
            font=dict(family="Tajawal"),
            xaxis=dict(title="النقاط المساهمة (الوزن × الدرجة)"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

# ------------------------------------------
# التبويب الثاني: تفكيك الأبعاد والمؤشرات الفرعية الخمسة
# ------------------------------------------
with tab2:
    st.subheader(f"🔍 التفاصيل الدقيقة لمكونات الجيل الرابع (5 مؤشرات فرعية لكل بُعد)")
    
    selected_dim = st.selectbox(
        "اختر البُعد الرئيسي لاستعراض مؤشراته الفرعية الخمسة:",
        DIMENSIONS
    )
    
    # تصفية البيانات للبُعد واللغة المححدين
    filtered_subs = ccpli_subs[
        (ccpli_subs['lang_name'] == selected_lang) & 
        (ccpli_subs['dim_name'] == selected_dim)
    ]
    
    col_sub_chart, col_sub_table = st.columns([1.2, 1])
    
    with col_sub_chart:
        fig_sub = px.bar(
            filtered_subs,
            x='sub_score',
            y='sub_indicator_name',
            orientation='h',
            title=f"تقييم المؤشرات الفرعية لـ ({selected_dim})",
            labels={'sub_score': 'الدرجة من 100', 'sub_indicator_name': 'المؤشر الفرعي'},
            color='sub_score',
            color_continuous_scale='Teal'
        )
        fig_sub.update_layout(
            height=350,
            font=dict(family="Tajawal"),
            xaxis=dict(range=[0, 100]),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_sub, use_container_width=True)
        
    with col_sub_table:
        st.markdown("##### 📋 جدول القيم والمعايير التفصيلية")
        st.dataframe(
            filtered_subs[['sub_indicator_name', 'sub_score']],
            column_config={
                "sub_indicator_name": "اسم المؤشر الفرعي",
                "sub_score": st.column_config.NumberColumn("الدرجة المستحقة", format="%.2f")
            },
            hide_index=True,
            use_container_width=True
        )

# ------------------------------------------
# التبويب الثالث: المقارنة الرادارية والمصفوفة الشاملة
# ------------------------------------------
with tab3:
    st.subheader("📈 المقارنة الرادارية المتعددة للأبعاد العشرة")
    
    selected_langs_radar = st.multiselect(
        "اختر اللغات للمقارنة الرادارية:",
        ccpli_summary['lang_name'].unique(),
        default=["العربية", "إنجليزية", "الصينية"] if "إنجليزية" in ccpli_summary['lang_name'].unique() else ["العربية", "إنجليزية"] if "إنجليزية" in ccpli_summary['lang_name'].unique() else ccpli_summary['lang_name'].unique()[:3]
    )
    
    if selected_langs_radar:
        fig_radar = go.Figure()
        
        for l_name in selected_langs_radar:
            l_df = ccpli_dims[ccpli_dims['lang_name'] == l_name]
            fig_radar.add_trace(go.Scatterpolar(
                r=l_df['dim_score'],
                theta=l_df['dim_name'],
                fill='toself',
                name=l_name
            ))
            
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=500,
            font=dict(family="Tajawal")
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    st.divider()
    st.subheader("📊 ترتيب اللغات في التقييم العالمي العام")
    st.dataframe(
        ccpli_summary.sort_values(by="ccpli_score", ascending=False),
        column_config={
            "lang_name": "اللغة",
            "ccpli_score": st.column_config.NumberColumn("النتيجة الكلية (CCPLI)", format="%.2f"),
            "tier": "التصنيف والطبقة الكونية"
        },
        hide_index=True,
        use_container_width=True
    )
