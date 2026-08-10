import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. تهيئة صفحة التطبيق والواجهة
# ==========================================
st.set_page_config(
    page_title="مرصد القوة اللغوية الشاملة | CCPLI v4.0",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص اتجاه الصفحة والتنسيق العربي (RTL)
st.markdown("""
    <style>
    .main { text-align: right; direction: rtl; }
    .stMetric { background-color: #f8f9fa; border-radius: 10px; padding: 15px; border-right: 5px solid #1E88E5; }
    .stSelectbox, .stSlider { direction: rtl; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. البيانات المرجعية والهيكلية (10 أبعاد - 50 مؤشراً)
# ==========================================
DIMENSIONS = {
    "D1": {"name": "القوة الرقمية والخوارزمية", "weight": 0.14, "color": "#1f77b4"},
    "D7": {"name": "الوزن الاقتصادي والسوقي", "weight": 0.12, "color": "#ff7f0e"},
    "D6": {"name": "القوة المعرفية والنشر العلمي", "weight": 0.11, "color": "#2ca02c"},
    "D3": {"name": "البناء الصرفي والخصائص اللغوية", "weight": 0.10, "color": "#d62728"},
    "D8": {"name": "الانتشار الديموغرافي والجغرافي", "weight": 0.10, "color": "#9467bd"},
    "D9": {"name": "التموضع الجيوسياسي والدبلوماسي", "weight": 0.10, "color": "#8c564b"},
    "D4": {"name": "القوة الدينية والقداسية", "weight": 0.09, "color": "#e377c2"},
    "D2": {"name": "تعليم اللغة كـ L2", "weight": 0.08, "color": "#7f7f7f"},
    "D5": {"name": "العمق الحضاري والأرشيف التاريخي", "weight": 0.08, "color": "#bcbd22"},
    "D10": {"name": "الإشغال الإعلامي والصناعات الثقافية", "weight": 0.08, "color": "#17becf"}
}

SUB_INDICATORS = {
    "D1": ["LLM Token Density (كثافة النماذج)", "أدوات NLP المفتوحة المصدر", "المحتوى الرقمي والموسوعي", "جودة الترجمة العصبية NMT", "كفاءة التعرف الصوتي ASR"],
    "D7": ["حصة الناتج المحلي GDP", "التجارة البينية وعابرة الحدود", "العائد المالي لإتقان اللغة Wage Premium", "تدفقات الاستثمار الأجنبي FDI", "النفوذ الملاحي واللوجستي"],
    "D6": ["النشر العلمي المحكم Scopus/WoS", "براءات الاختراع والابتكارات", "معدل الترجمة الأكاديمية", "الحضور في المؤتمرات الدولية", "المعاجم العلمية المتخصصة"],
    "D3": ["الطاقة الاشتقاقية والتوليدية", "الاتساع المعجمي والتمايز الدلالي", "انضباط القياس النحوي والصرفي", "مرونة التراكيب والتنوع الأسلوبي", "القدرة الذاتية على التعريب والتبيئة"],
    "D8": ["حجم المتحدثين الأصليين L1", "توزع الجاليات Diaspora", "معدلات النمو السكاني الطبيعي", "التعدد القاري والسيادي", "نسبة الفئة العمرية الشابة"],
    "D9": ["الاعتماد في المنظمات الدولية", "التوثيق في المعاهدات الدولية", "ثقل التحالفات والتكتلات", "الدبلوماسية الثقافية والوساطات", "صياغة التشريعات عابرة الحدود"],
    "D4": ["الكثافة البشرية للممارسة التعبدية", "المركزية النصية اللاهوتية", "انتشار الجامعات والدور المرجعية", "حركة حفظ واستظهار النص المقدس", "الأثر الرمزي في المواسم الجامعة"],
    "D2": ["انتشار المراكز اللغوية الدولية", "إقبال الطلاب الدوليين جامعياً", "تقنين الاختبارات المعيارية L2", "جودة السلسلات والمناهج التعليمية", "منصات التعلم الذاتي MOOCs"],
    "D5": ["حجم الأرشيف المخطوط والوثائق", "مواقع التراث العالمي UNESCO", "الاستمرارية القرائية عبر القرون", "المساهمة التاريخية في نقل العلوم", "الجماليات والتراث الفني"],
    "D10": ["الإنتاج السينمائي والموسيقي", "الشبكات الإخبارية العالمية", "مبيعات الكتب والنشر الثقافي", "صناعة الألعاب الإلكترونية", "التأثير في شبكات التواصل الاجتماعي"]
}

# درجات افتراضية معيرة للنموذج المحاكى (مثال لغة تقدمية)
DEFAULT_SCORES = {
    "D1": [65, 70, 60, 75, 68],
    "D7": [70, 65, 60, 55, 62],
    "D6": [50, 45, 55, 60, 58],
    "D3": [95, 90, 85, 92, 88],
    "D8": [80, 75, 82, 70, 85],
    "D9": [75, 70, 68, 72, 65],
    "D4": [98, 95, 90, 96, 92],
    "D2": [60, 65, 58, 70, 62],
    "D5": [92, 88, 90, 85, 89],
    "D10": [65, 70, 62, 55, 68]
}

# ==========================================
# 3. الدوال الرياضياتية والتحليلية
# ==========================================
def calculate_ccpli(sub_scores, weights, m_momentum, m_geo):
    """حساب مؤشر CCPLI v4.0 باستخدام الدالة غير الخطية الأُسية-اللوغاريتمية"""
    dim_scores = {}
    log_sum = 0.0
    
    for d_code, scores in sub_scores.items():
        # S_i هو متوسط المؤشرات الفرعية الخمسة
        s_i = np.mean(scores)
        dim_scores[d_code] = s_i
        w_i = weights[d_code]
        # w_i * ln(1 + 0.99 * S_i)
        log_sum += w_i * np.log(1.0 + 0.99 * s_i)
        
    base_score = np.exp(log_sum)
    final_score = base_score * m_momentum * m_geo
    return final_score, dim_scores

def run_monte_carlo(sub_scores, weights, m_momentum, m_geo, iterations=500, noise_std=0.15):
    """معمل محاكاة مونت كارلو لاختبار الصلابة الإحصائية"""
    results = []
    for _ in range(iterations):
        noisy_sub_scores = {}
        for d_code, scores in sub_scores.items():
            # إضافة ضوضاء عشوائية بنسبة noise_std على الدرجات
            noise = np.random.normal(0, noise_std * 100, len(scores))
            noisy_scores = np.clip(np.array(scores) + noise, 0, 100)
            noisy_sub_scores[d_code] = noisy_scores
            
        score, _ = calculate_ccpli(noisy_sub_scores, weights, m_momentum, m_geo)
        results.append(score)
    return np.array(results)

# ==========================================
# 4. الشريط الجانبي (Sidebar Control)
# ==========================================
st.sidebar.title("🎛️ لوحة التحكم والمعايرة")
st.sidebar.markdown("---")

st.sidebar.subheader("🚀 المضاعفات الاستراتيجية")
m_momentum = st.sidebar.slider("مضاعف الزخم والتطور (M_momentum)", 1.000, 1.200, 1.085, step=0.005)
m_geo = st.sidebar.slider("مضاعف الانتشار الجغرافي (M_geo)", 1.000, 1.200, 1.082, step=0.005)

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ تعديل الأوزان النسبية")
custom_weights = {}
total_w = 0.0
for d_code, d_info in DIMENSIONS.items():
    w = st.sidebar.number_input(f"وزن {d_info['name']} ({d_code})", 0.01, 0.30, d_info['weight'], step=0.01)
    custom_weights[d_code] = w
    total_w += w

if round(total_w, 2) != 1.00:
    st.sidebar.warning(f"⚠️ مجموع الأوزان الحالي: {total_w:.2f} (يجب أن يكون 1.00)")

# ==========================================
# 5. الواجهة الرئيسية وتبويبات التطبيق
# ==========================================
st.title("🌐 مرصد القوة اللغوية الشاملة (CCPLI v4.0)")
st.caption("نموذج إحصائي محكم لقياس القوة الاستراتيجية للغات وفق 10 أبعاد متكاملة و50 مؤشراً فرعياً")

# جمع مدخلات المؤشرات الفرعية الخمسين
user_sub_scores = {}

tabs = st.tabs([
    "📊 لوحة النتائج والتحليل الراداري", 
    "🧬 المؤشرات الفرعية الخمسون (50 Sub-indicators)", 
    "🎲 محاكاة مونت كارلو والصلابة", 
    "📐 محاكاة السياسات والسيناريوهات"
])

# --- التبويب الثاني: إدخال الدرجات الخام للمؤشرات الخمسين ---
with tabs[1]:
    st.markdown("### 📝 إدخال وتقييم الدرجات المعيارية للمؤشرات الفرعية [0 - 100]")
    st.info("قم بتعديل قيم المؤشرات الفرعية بناءً على أداء اللغة المراد تقييمها.")
    
    col_a, col_b = st.columns(2)
    dim_keys = list(DIMENSIONS.keys())
    
    for i, d_code in enumerate(dim_keys):
        target_col = col_a if i % 2 == 0 else col_b
        with target_col:
            with st.expander(f"📌 {DIMENSIONS[d_code]['name']} ({d_code}) - الوزن: {custom_weights[d_code]}"):
                scores_list = []
                for j, sub_name in enumerate(SUB_INDICATORS[d_code]):
                    val = st.slider(
                        f"{sub_name}",
                        min_value=0, max_value=100,
                        value=DEFAULT_SCORES[d_code][j],
                        key=f"{d_code}_sub_{j}"
                    )
                    scores_list.append(val)
                user_sub_scores[d_code] = scores_list

# إجراء الحسابات الأساسية
final_ccpli, dim_scores = calculate_ccpli(user_sub_scores, custom_weights, m_momentum, m_geo)

# --- التبويب الأول: لوحة النتائج والرادار ---
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("درجة المؤشر الشاملة (CCPLI)", f"{final_ccpli:.2f}")
    
    # تصنيف المستوى
    tier = "عالمية متقدمة (Tier 1)" if final_ccpli >= 80 else ("دولية مؤاثرة (Tier 2)" if final_ccpli >= 60 else "إقليمية محصورة (Tier 3)")
    col2.metric("التصنيف الاستراتيجي للغة", tier)
    col3.metric("مضاعف الزخم (M_momentum)", f"{m_momentum:.3f}")
    col4.metric("مضاعف التشتت الجغرافي (M_geo)", f"{m_geo:.3f}")
    
    st.markdown("---")
    
    r_col1, r_col2 = st.columns([1.2, 0.8])
    
    with r_col1:
        st.subheader("🕸️ الرسم البياني الراداري للأبعاد العشرة")
        categories = [DIMENSIONS[d]['name'] for d in dim_scores.keys()]
        values = list(dim_scores.values())
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='أداء اللغة',
            line_color='#1E88E5',
            fillcolor='rgba(30, 136, 229, 0.3)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=450,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with r_col2:
        st.subheader("📋 تفصيل درجات الأبعاد الموزونة")
        df_summary = pd.DataFrame({
            "البُعد الاستراتيجي": [DIMENSIONS[d]['name'] for d in dim_scores.keys()],
            "الوزن": [custom_weights[d] for d in dim_scores.keys()],
            "الدرجة الخام (S_i)": [round(dim_scores[d], 2) for d in dim_scores.keys()],
            "المساهمة": [round(dim_scores[d] * custom_weights[d], 2) for d in dim_scores.keys()]
        })
        st.dataframe(df_summary, hide_index=True, height=400, use_container_width=True)

# --- التبويب الثالث: محاكاة مونت كارلو ---
with tabs[2]:
    st.markdown("### 🎲 اختبار الحساسية والصلابة الإحصائية (Monte Carlo Lab)")
    st.write("يقوم هذا المعمل بإحداث تذبذب وضوضاء عشوائية بنسبة (±15%) على الدرجات لبيان صمود الترتيب وثبات النتائج.")
    
    m_col1, m_col2 = st.columns([1, 3])
    with m_col1:
        sim_iterations = st.selectbox("عدد دورات المحاكاة", [500, 1000, 2000], index=0)
        run_sim = st.button("🚀 تشغيل محاكاة مونت كارلو")
        
    if run_sim:
        with st.spinner("جاري تشغيل محاكاة مونت كارلو..."):
            mc_results = run_monte_carlo(user_sub_scores, custom_weights, m_momentum, m_geo, iterations=sim_iterations)
            
            mean_score = np.mean(mc_results)
            std_dev = np.std(mc_results)
            ci_lower = np.percentile(mc_results, 2.5)
            ci_upper = np.percentile(mc_results, 97.5)
            
            st.success(f"تمت المحاكاة بنجاح عبر {sim_iterations} دورة!")
            
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("متوسط درجة المحاكاة", f"{mean_score:.2f}")
            res_c2.metric("الانحراف المعياري (Std Dev)", f"{std_dev:.2f}")
            res_c3.metric("فاصل الثقة 95%", f"[{ci_lower:.2f} - {ci_upper:.2f}]")
            
            # رسم التوزيع الاحتمالي
            fig_hist = px.histogram(
                mc_results, nbins=40, 
                title="توزيع الدرجات المحاكاة عبر مونت كارلو",
                labels={'value': 'درجة CCPLI'},
                color_discrete_sequence=['#4CAF50']
            )
            fig_hist.add_vline(x=mean_score, line_dash="dash", line_color="red", annotation_text="المتوسط")
            st.plotly_chart(fig_hist, use_container_width=True)

# --- التبويب الرابع: محاكاة السياسات والسيناريوهات ---
with tabs[3]:
    st.markdown("### 📐 محاكاة سيناريوهات الاستثمار والتخطيط اللغوي")
    st.write("اختبر أثر ضخ استثمارات موجهة لتطوير بعد معين (مثل رفع البُعد الرقمي D1 أو المعرفي D6) على الدرجة النهائية للمؤشر.")
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        target_dim = st.selectbox("اختر البُعد الموجه للاستثمار والتطوير", list(DIMENSIONS.keys()), format_func=lambda x: DIMENSIONS[x]['name'])
        boost_amount = st.slider("نسبة التحسين المترتبة على المشروع (%)", 5, 50, 20)
        
    # حساب السيناريو الجديد
    boosted_sub_scores = {k: list(v) for k, v in user_sub_scores.items()}
    boosted_sub_scores[target_dim] = [min(100, x * (1 + boost_amount/100)) for x in boosted_sub_scores[target_dim]]
    
    new_ccpli, new_dim_scores = calculate_ccpli(boosted_sub_scores, custom_weights, m_momentum, m_geo)
    diff = new_ccpli - final_ccpli
    
    with s_col2:
        st.markdown("#### 🎯 نتيجة محاكاة القرار الاستراتيجي:")
        st.metric(f"الدرجة الجديدة المتوقعة لـ CCPLI", f"{new_ccpli:.2f}", delta=f"+{diff:.2f} نقطة")
        st.info(f"استثمار نسبة {boost_amount}% في تطوير {DIMENSIONS[target_dim]['name']} سينعكس بارتفاع قدره {diff:.2f} نقطة في المؤشر الشامل.")
