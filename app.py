import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. تهيئة صفحة التطبيق والتنسيق العربي (RTL)
# ==========================================
st.set_page_config(
    page_title="مرصد CCPLI v4.0 - مقارنة اللغات العشر",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تنسيق CSS آمن يتكيف تلقائياً مع الوضعين الفاتح والداكن (Dark/Light Mode)
st.markdown("""
    <style>
    .stAppViewContainer, .stHeader, .stSidebar { direction: rtl; text-align: right; }
    div[data-testid="stMetric"] {
        background-color: rgba(30, 136, 229, 0.08) !important;
        border-radius: 10px;
        padding: 15px;
        border-right: 5px solid #1E88E5 !important;
    }
    .stTable { direction: rtl; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. مسميات الأبعاد والأوزان القياسية (CCPLI v4.0)
# ==========================================
DIMENSIONS = {
    "D1": "القوة الرقمية والخوارزمية",
    "D7": "الوزن الاقتصادي والسوقي",
    "D6": "القوة المعرفية والنشر العلمي",
    "D3": "البناء الصرفي والخصائص اللغوية", 
    "D8": "الانتشار الديموغرافي والجغرافي",
    "D9": "التموضع الجيوسياسي والدبلوماسي",
    "D4": "القوة الدينية والقداسية",
    "D2": "تعليم اللغة كـ L2", 
    "D5": "العمق الحضاري والأرشيف التاريخي",
    "D10": "الإشغال الإعلامي والصناعات الثقافية"
}

WEIGHTS = {
    "D1": 0.14, "D7": 0.12, "D6": 0.11, "D3": 0.10, "D8": 0.10,
    "D9": 0.10, "D4": 0.09, "D2": 0.08, "D5": 0.08, "D10": 0.08
}

# ==========================================
# 3. قاعدة البيانات الشاملة للغات العشر (50 مؤشراً فرعياً + المضاعفات)
# ==========================================
LANGUAGES_DATA = {
    "العربية": {
        "m_momentum": 1.085, "m_geo": 1.082,
        "scores": {
            "D1": [65, 70, 60, 75, 68], "D7": [70, 65, 60, 55, 62], "D6": [50, 45, 55, 60, 58],
            "D3": [95, 90, 85, 92, 88], "D8": [80, 75, 82, 70, 85], "D9": [75, 70, 68, 72, 65],
            "D4": [98, 95, 90, 96, 92], "D2": [60, 65, 58, 70, 62], "D5": [92, 88, 90, 85, 89],
            "D10": [65, 70, 62, 55, 68]
        }
    },
    "الإنجليزية": {
        "m_momentum": 1.150, "m_geo": 1.180,
        "scores": {
            "D1": [98, 96, 95, 97, 96], "D7": [95, 94, 96, 92, 95], "D6": [96, 98, 94, 95, 95],
            "D3": [75, 72, 78, 70, 75], "D8": [90, 88, 92, 85, 90], "D9": [98, 97, 96, 99, 98],
            "D4": [40, 35, 42, 38, 40], "D2": [99, 98, 97, 99, 98], "D5": [80, 82, 78, 80, 80],
            "D10": [96, 95, 98, 94, 97]
        }
    },
    "الصينية (المندرين)": {
        "m_momentum": 1.120, "m_geo": 1.020,
        "scores": {
            "D1": [90, 92, 88, 91, 89], "D7": [98, 97, 96, 99, 98], "D6": [88, 90, 86, 88, 87],
            "D3": [80, 82, 78, 80, 80], "D8": [95, 98, 94, 92, 96], "D9": [85, 86, 84, 85, 85],
            "D4": [30, 28, 32, 25, 30], "D2": [75, 78, 72, 76, 74], "D5": [85, 88, 82, 86, 84],
            "D10": [80, 82, 78, 80, 80]
        }
    },
    "الإسبانية": {
        "m_momentum": 1.060, "m_geo": 1.100,
        "scores": {
            "D1": [78, 80, 76, 79, 77], "D7": [75, 76, 74, 75, 75], "D6": [70, 72, 68, 71, 69],
            "D3": [82, 84, 80, 83, 81], "D8": [88, 90, 86, 89, 87], "D9": [80, 82, 78, 81, 79],
            "D4": [50, 52, 48, 51, 49], "D2": [85, 87, 83, 86, 84], "D5": [82, 84, 80, 83, 81],
            "D10": [85, 86, 84, 85, 85]
        }
    },
    "الفرنسية": {
        "m_momentum": 1.020, "m_geo": 1.080,
        "scores": {
            "D1": [80, 82, 78, 81, 79], "D7": [82, 84, 80, 83, 81], "D6": [82, 84, 80, 83, 81],
            "D3": [80, 82, 78, 81, 79], "D8": [60, 62, 58, 61, 59], "D9": [90, 92, 88, 91, 89],
            "D4": [35, 37, 33, 36, 34], "D2": [88, 90, 86, 89, 87], "D5": [88, 90, 86, 89, 87],
            "D10": [82, 84, 80, 83, 81]
        }
    },
    "الروسية": {
        "m_momentum": 1.000, "m_geo": 1.040,
        "scores": {
            "D1": [75, 77, 73, 76, 74], "D7": [70, 72, 68, 71, 69], "D6": [78, 80, 76, 79, 77],
            "D3": [80, 82, 78, 81, 79], "D8": [65, 67, 63, 66, 64], "D9": [75, 77, 73, 76, 74],
            "D4": [30, 32, 28, 31, 29], "D2": [60, 62, 58, 61, 59], "D5": [78, 80, 76, 79, 77],
            "D10": [65, 67, 63, 66, 64]
        }
    },
    "الألمانية": {
        "m_momentum": 1.010, "m_geo": 1.030,
        "scores": {
            "D1": [82, 84, 80, 83, 81], "D7": [88, 90, 86, 89, 87], "D6": [89, 91, 87, 90, 88],
            "D3": [85, 87, 83, 86, 84], "D8": [40, 42, 38, 41, 39], "D9": [78, 80, 76, 79, 77],
            "D4": [25, 27, 23, 26, 24], "D2": [70, 72, 68, 71, 69], "D5": [80, 82, 78, 81, 79],
            "D10": [70, 72, 68, 71, 69]
        }
    },
    "اليابانية": {
        "m_momentum": 1.010, "m_geo": 1.010,
        "scores": {
            "D1": [85, 87, 83, 86, 84], "D7": [84, 86, 82, 85, 83], "D6": [82, 84, 80, 83, 81],
            "D3": [78, 80, 76, 79, 77], "D8": [45, 47, 43, 46, 44], "D9": [65, 67, 63, 66, 64],
            "D4": [20, 22, 18, 21, 19], "D2": [55, 57, 53, 56, 54], "D5": [75, 77, 73, 76, 74],
            "D10": [80, 82, 78, 81, 79]
        }
    },
    "البرتغالية": {
        "m_momentum": 1.040, "m_geo": 1.060,
        "scores": {
            "D1": [65, 67, 63, 66, 64], "D7": [68, 70, 66, 69, 67], "D6": [60, 62, 58, 61, 59],
            "D3": [75, 77, 73, 76, 74], "D8": [70, 72, 68, 71, 69], "D9": [65, 67, 63, 66, 64],
            "D4": [40, 42, 38, 41, 39], "D2": [58, 60, 56, 59, 57], "D5": [70, 72, 68, 71, 69],
            "D10": [68, 70, 66, 69, 67]
        }
    },
    "الهندية": {
        "m_momentum": 1.080, "m_geo": 1.010,
        "scores": {
            "D1": [60, 62, 58, 61, 59], "D7": [72, 74, 70, 73, 71], "D6": [55, 57, 53, 56, 54],
            "D3": [78, 80, 76, 79, 77], "D8": [92, 94, 90, 93, 91], "D9": [60, 62, 58, 61, 59],
            "D4": [65, 67, 63, 66, 64], "D2": [45, 47, 43, 46, 44], "D5": [75, 77, 73, 76, 74],
            "D10": [60, 62, 58, 61, 59]
        }
    }
}

# ==========================================
# 4. دالة حساب مؤشر CCPLI v4.0 غير الخطية
# ==========================================
def calculate_ccpli(lang_entry, weights):
    """حساب المؤشر باستخدام التجميع الأُسي-اللوغاريتمي مع المضاعفات الاستراتيجية"""
    sub_scores = lang_entry["scores"]
    m_momentum = lang_entry.get("m_momentum", 1.0)
    m_geo = lang_entry.get("m_geo", 1.0)
    
    log_sum = sum(
        weights[d] * np.log(1.0 + 0.99 * np.mean(sub_scores[d])) 
        for d in weights if d in sub_scores
    )
    base_score = np.exp(log_sum)
    final_score = base_score * m_momentum * m_geo
    return final_score

def get_tier(score):
    if score >= 80:
        return "عالمية متقدمة (Tier 1)"
    elif score >= 60:
        return "دولية مؤثرة (Tier 2)"
    else:
        return "إقليمية محصورة (Tier 3)"

# ==========================================
# 5. واجهة التطبيق وتبويبات العرض
# ==========================================
st.title("🌐 مرصد CCPLI v4.0 - المقارنة العالمية بين اللغات العشر")
st.caption("أداة القياس والتحليل المقارن لنفوذ اللغات وفق 10 أبعاد استراتيجية و50 مؤشراً فرعياً")

tab1, tab2 = st.tabs(["📊 التحليل المقارن الثنائي", "🏆 جدول الترتيب والتصنيف العالمي"])

# ------------------------------------------
# التبويب الأول: التحليل المقارن الثنائي
# ------------------------------------------
with tab1:
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        lang1 = st.selectbox("اللغة الأساسية للمقارنة:", list(LANGUAGES_DATA.keys()), index=0)
    with col_sel2:
        lang2_options = ["لا يوجد"] + [l for l in LANGUAGES_DATA.keys() if l != lang1]
        lang2 = st.selectbox("اللغة الثانية (للمقارنة):", lang2_options, index=1)

    # حساب الدرجات
    score1 = calculate_ccpli(LANGUAGES_DATA[lang1], WEIGHTS)
    
    st.markdown("---")
    cols = st.columns(2)

    # العمود الأول: البطاقات الإحصائية والرادار
    with cols[0]:
        st.subheader("📈 مؤشر القوة والرسم الراداري")
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(f"مؤشر {lang1}", f"{score1:.2f}", delta=get_tier(score1))
        
        if lang2 != "لا يوجد":
            score2 = calculate_ccpli(LANGUAGES_DATA[lang2], WEIGHTS)
            m_col2.metric(f"مؤشر {lang2}", f"{score2:.2f}", delta=get_tier(score2))

        # إنشاء الرسم البياني الراداري
        fig = go.Figure()
        
        # اللغة الأولى
        r1_vals = [np.mean(LANGUAGES_DATA[lang1]["scores"][d]) for d in DIMENSIONS]
        fig.add_trace(go.Scatterpolar(
            r=r1_vals + [r1_vals[0]],
            theta=list(DIMENSIONS.values()) + [list(DIMENSIONS.values())[0]],
            fill='toself',
            name=lang1,
            line_color='#1E88E5'
        ))
        
        # اللغة الثانية (إن وجدت)
        if lang2 != "لا يوجد":
            r2_vals = [np.mean(LANGUAGES_DATA[lang2]["scores"][d]) for d in DIMENSIONS]
            fig.add_trace(go.Scatterpolar(
                r=r2_vals + [r2_vals[0]],
                theta=list(DIMENSIONS.values()) + [list(DIMENSIONS.values())[0]],
                fill='toself',
                name=lang2,
                line_color='#FF7F0E',
                opacity=0.6
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=480,
            margin=dict(l=40, r=40, t=30, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    # العمود الثاني: تفاصيل البيانات والمقارنة البعدية
    with cols[1]:
        st.subheader("📋 تفاصيل الأداء عبر الأبعاد العشرة")
        
        df_comp = pd.DataFrame({
            "البُعد الاستراتيجي": list(DIMENSIONS.values()),
            f"{lang1} (الدرجة الخام)": [round(np.mean(LANGUAGES_DATA[lang1]["scores"][d]), 1) for d in DIMENSIONS]
        })
        
        if lang2 != "لا يوجد":
            df_comp[f"{lang2} (الدرجة الخام)"] = [round(np.mean(LANGUAGES_DATA[lang2]["scores"][d]), 1) for d in DIMENSIONS]
            # إبراز الأعلى لكل بعد
            st.dataframe(df_comp.style.highlight_max(subset=[f"{lang1} (الدرجة الخام)", f"{lang2} (الدرجة الخام)"], axis=1, color="rgba(46, 125, 50, 0.2)"), use_container_width=True, height=420)
        else:
            st.dataframe(df_comp, use_container_width=True, height=420)

# ------------------------------------------
# التبويب الثاني: جدول الترتيب والتصنيف العالمي
# ------------------------------------------
with tab2:
    st.subheader("🏆 الترتيب العالمي الشامل للغات العشر")
    st.write("جدول مرتب تنازلياً حسب القوة الشاملة لمؤشر CCPLI v4.0:")
    
    ranking_list = []
    for lang_name, lang_info in LANGUAGES_DATA.items():
        sc = calculate_ccpli(lang_info, WEIGHTS)
        ranking_list.append({
            "اللغة": lang_name,
            "درجة CCPLI": round(sc, 2),
            "التصنيف الاستراتيجي": get_tier(sc),
            "مضاعف الزخم (M_momentum)": lang_info["m_momentum"],
            "مضاعف الانتشار (M_geo)": lang_info["m_geo"]
        })
        
    df_ranking = pd.DataFrame(ranking_list).sort_values(by="درجة CCPLI", ascending=False).reset_index(drop=True)
    df_ranking.index += 1 # الترتيب يبدأ من 1
    
    st.dataframe(df_ranking, use_container_width=True)
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
