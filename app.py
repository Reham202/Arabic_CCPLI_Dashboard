
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import streamlit as st

# ---------------------------------------------------------
# 1. تهيئة الصفحة والإعدادات العامة
# ---------------------------------------------------------
st.set_page_config(
    page_title="مؤشر قوة اللغات المركب (CCPLI v4.0)",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# تطبيق تنسيقات CSS لتحسين الواجهة باللغة العربية (RTL)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-right: 5px solid #1E3A8A;
    }
    </style>
""",
    unsafe_text_html=True,
)

# ---------------------------------------------------------
# 2. البيانات الأساسية والهيكل العام (10 أبعاد × 5 مؤشرات)
# ---------------------------------------------------------
DIMENSIONS = [
    "الثقل الديموغرافي",
    "القوة الاقتصادية",
    "النفوذ الجيوسياسي",
    "الإنتاج العلمي",
    "المحتوى الرقمي",
    "الجاذبية التعليمية",
    "الحضور الإعلامي",
    "الترجمة والتنوع",
    "رأس المال الثقافي",
    "المرونة المؤسسية",
]

SUB_INDICATORS_DEFAULT = {
    "الثقل الديموغرافي": [
        "عدد الناطقين الأصليين",
        "عدد الناطقين كبلد ثاني",
        "معدل النمو الديموغرافي",
        "الانتشار الجغرافي للسكّان",
        "نسبة الشباب الناطقين",
    ],
    "القوة الاقتصادية": [
        "الناتج المحلي الإجمالي المجمع",
        "حجم التجارة الدولية باللغة",
        "الثقل في أسواق العمل",
        "الابتكار وريادة الأعمال",
        "معدل الإنفاق السياحي",
    ],
    "النفوذ الجيوسياسي": [
        "الاعتماد في المنظمات الدولية",
        "عدد الدول ذات الصفة الرسمية",
        "الاتفاقيات والتحالفات الدولية",
        "الدبلوماسية الثقافية",
        "المساعدات والتنمية العابرة للحدود",
    ],
    "الإنتاج العلمي": [
        "عدد الأبحاث المنشورة سنوياً",
        "الأوراق المؤرشفة في قاعدة بيانات عالمية",
        "حجم براءات الاختراع",
        "الاستشهادات العلمية باللغة",
        "مشاريع البحث والتطوير (R&D)",
    ],
    "المحتوى الرقمي": [
        "نسبة المحتوى على شبكة الإنترنت",
        "حجم المدونات والبيانات الرقمية",
        "المعالم اللغوية في أبحاث الذكاء الاصطناعي",
        "تفاعل منصات التواصل الاجتماعي",
        "جودة المحتوى البرمجي والأدوات",
    ],
    "الجاذبية التعليمية": [
        "أعداد متعلميها كلفة ثانية/أجنبية",
        "البرامج الأكاديمية بالجامعات العالمية",
        "مراكز الاختبارات والشهادات الدولية",
        "المنح الدراسية المتاحة",
        "تطوير المناهج وتأهيل المعلمين",
    ],
    "الحضور الإعلامي": [
        "القنوات القائمة والشبكات الدولية",
        "معدل المشاهدات والاستماع العالمي",
        "صناعة السينما والدراما",
        "البودكاست والإنتاج الصوتي",
        "النشر الصحفي الإلكتروني والمطبوع",
    ],
    "الترجمة والتنوع": [
        "حجم الكتب المترجمة منها وإليها",
        "المؤتمرات والدوريات المترجمة",
        "دعم أدوات الترجمة الآلية",
        "التنوع اللهجي واللغوي المخصب",
        "المشاريع القومية للترجمة",
    ],
    "رأس المال الثقافي": [
        "المواقع المسجلة في اليونسكو",
        "الجوائز الأدبية والفكرية العالمية",
        "المكانة الدينية والتاريخية",
        "الفعاليات والمعارض الثقافية",
        "المصنفات الفنية والأدبية الخالدة",
    ],
    "المرونة المؤسسية": [
        "قدرة المجامع اللغوية والتحديث",
        "السياسات والتشريعات اللغوية",
        "القدرة الاشتقاقية والمعجمية",
        "استيعاب المصطلحات الحديثة",
        "تمويل وتطوير البنية اللغوية",
    ],
}

# تهيئة حالة الجلسة (Session State)
if "sub_indicators_df" not in st.session_state:
  rows = []
  for dim in DIMENSIONS:
    for sub in SUB_INDICATORS_DEFAULT[dim]:
      rows.append({
          "البعد الرئيسي": dim,
          "المؤشر الفرعي": sub,
          "القيمة (0-100)": 50.0,
          "الوزن النسبي": 1.0,
      })
  st.session_state.sub_indicators_df = pd.DataFrame(rows)


# ---------------------------------------------------------
# 3. دوال التصدير والطباعة
# ---------------------------------------------------------
def generate_pdf(summary_df, sub_df):
  """إنشاء ملف PDF للطباعة يضم النتائج والجداول التفصيلية"""
  buffer = io.BytesIO()
  doc = SimpleDocTemplate(
      buffer, pagesize=landscape(A4), rightMargin=20, leftMargin=20
  )
  elements = []

  # تحويل ملخص الأبعاد إلى قائمة لـ PDF
  summary_data = [["البعد الرئيسي", "نتيجة البعد (0-100)"]] + summary_df[
      ["البعد الرئيسي", "نتيجة البعد (0-100)"]
  ].values.tolist()

  t_summary = Table(summary_data, colWidths=[300, 150])
  t_summary.setStyle(
      TableStyle([
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
          ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
          ("ALIGN", (0, 0), (-1, -1), "CENTER"),
          ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
          ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
          ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F3F4F6")),
          ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#D1D5DB")),
      ])
  )

  elements.append(t_summary)
  doc.build(elements)
  buffer.seek(0)
  return buffer


# ---------------------------------------------------------
# 4. الشريط الجانبي والقائمة الرئيسية
# ---------------------------------------------------------
st.sidebar.title("📌 القائمة الرئيسية")
page = st.sidebar.radio(
    "انتقل إلى:",
    [
        "📝 إدخال وتعديل الـ 50 مؤشر",
        "📊 النتائج والتحليل العنكبوتي",
        "🖨️ التصدير والطباعة",
    ],
)

# ---------------------------------------------------------
# 5. الصفحة الأولى: إدخال وتعديل الـ 50 مؤشراً
# ---------------------------------------------------------
if page == "📝 إدخال وتعديل الـ 50 مؤشر":
  st.title("📝 إدخال وتعديل أبعاد المؤشر الـ 50")
  st.write(
      "يمكنك تعديل أسماء المؤشرات الفرعية، القيم الحالية (من 0 إلى 100)،"
      " والأوزان النسبية لكل مؤشر."
  )

  edited_df = st.data_editor(
      st.session_state.sub_indicators_df,
      num_rows="fixed",
      use_container_width=True,
      height=600,
      column_config={
          "البعد الرئيسي": st.column_config.TextColumn(disabled=True),
          "المؤشر الفرعي": st.column_config.TextColumn(
              label="اسم المؤشر الفرعي"
          ),
          "القيمة (0-100)": st.column_config.NumberColumn(
              min_value=0.0, max_value=100.0, step=0.5, format="%.2f"
          ),
          "الوزن النسبي": st.column_config.NumberColumn(
              min_value=0.1, max_value=5.0, step=0.1, format="%.1f"
          ),
      },
      key="sub_indicators_editor",
  )

  st.session_state.sub_indicators_df = edited_df
  st.success("✅ يتم حفظ وتحديث الحسابات تلقائياً عند تغيير أي قيمة.")

# ---------------------------------------------------------
# إجراء الحسابات التلقائية الموحدة
# ---------------------------------------------------------
df_calc = st.session_state.sub_indicators_df.copy()
df_calc["القيمة الموزونة"] = df_calc["القيمة (0-100)"] * df_calc["الوزن النسبي"]

summary_df = (
    df_calc.groupby("البعد الرئيسي", sort=False)
    .agg({"القيمة الموزونة": "sum", "الوزن النسبي": "sum"})
    .reset_index()
)

summary_df["نتيجة البعد (0-100)"] = (
    summary_df["القيمة الموزونة"] / summary_df["الوزن النسبي"]
)
overall_score = summary_df["نتيجة البعد (0-100)"].mean()

# ---------------------------------------------------------
# 6. الصفحة الثانية: عرض النتائج والرسم البياني العنكبوتي
# ---------------------------------------------------------
if page == "📊 النتائج والتحليل العنكبوتي":
  st.title("📊 ملخص النتائج والتمثيل العنكبوتي")

  col_score, col_blank = st.columns([1, 2])
  with col_score:
    st.metric(
        label="المؤشر العام المركب لقوة اللغة",
        value=f"{overall_score:.2f} / 100",
    )

  st.markdown("---")
  col_chart, col_table = st.columns([1.2, 1])

  with col_chart:
    st.subheader("🕸️ الرسم البياني العنكبوتي (Radar Chart)")

    # إعداد رسم Plotly التفاعلي للرادار
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=summary_df["نتيجة البعد (0-100)"].tolist()
            + [summary_df["نتيجة البعد (0-100)"].iloc[0]],
            theta=summary_df["البعد الرئيسي"].tolist()
            + [summary_df["البعد الرئيسي"].iloc[0]],
            fill="toself",
            name="مؤشر اللغة",
            line_color="#1E3A8A",
            fillcolor="rgba(30, 58, 138, 0.3)",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

  with col_table:
    st.subheader("📋 نتائج الأبعاد الرئيسية الـ 10")
    st.dataframe(
        summary_df[["البعد الرئيسي", "نتيجة البعد (0-100)"]].style.format(
            {"نتيجة البعد (0-100)": "{:.2f}"}
        ),
        use_container_width=True,
        height=450,
    )

# ---------------------------------------------------------
# 7. الصفحة الثالثة: التصدير والطباعة
# ---------------------------------------------------------
if page == "🖨️ التصدير والطباعة":
  st.title("🖨️ خيارات الطباعة والتصدير")
  st.write(
      "تتيح لك هذه الصفحة تحميل البيانات والجداول بصيغ مختلفة تناسب الطباعة"
      " والتقارير الأكاديمية."
  )

  st.subheader("1️⃣ طباعة سريعة مباشرة للجدول الرئيسي")

  # تحويل ملخص الأبعاد لـ HTML منسق للطباعة المباشرة
  html_table = summary_df[["البعد الرئيسي", "نتيجة البعد (0-100)"]].to_html(
      index=False, classes="styled-table"
  )
  print_component = f"""
    <style>
        .styled-table {{
            border-collapse: collapse;
            font-size: 1em;
            width: 100%;
            direction: rtl;
            text-align: right;
        }}
        .styled-table th {{
            background-color: #1E3A8A;
            color: #ffffff;
            padding: 10px;
        }}
        .styled-table td {{
            padding: 8px 12px;
            border: 1px solid #dddddd;
        }}
        @media print {{
            .no-print {{ display: none; }}
        }}
    </style>
    <button class="no-print" onclick="window.print()" style="padding: 10px 20px; font-size: 16px; background-color: #10B981; color: white; border: none; border-radius: 5px; cursor: pointer;">
        🖨️ فتح نافذة الطباعة (Print)
    </button>
    <br><br>
    {html_table}
    """
  st.components.v1.html(print_component, height=400, scrolling=True)

  st.markdown("---")
  st.subheader("2️⃣ تنزيل الملفات القابلة للطباعة والتعديل")

  c1, c2, c3 = st.columns(3)

  with c1:
    pdf_file = generate_pdf(summary_df, df_calc)
    st.download_button(
        label="📄 تحميل التقرير كـ PDF",
        data=pdf_file,
        file_name="Language_Power_Index_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

  with c2:
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
      summary_df.to_excel(writer, sheet_name="الأبعاد الرئيسية", index=False)
      df_calc.to_excel(writer, sheet_name="المؤشرات الـ 50", index=False)

    st.download_button(
        label="📊 تحميل كـ Excel شامل",
        data=excel_buffer.getvalue(),
        file_name="Language_Power_Index.xlsx",
        mime="application/vnd.ms-excel",
        use_container_width=True,
    )

  with c3:
    csv_data = df_calc.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📂 تحميل البيانات الـ 50 (CSV)",
        data=csv_data,
        file_name="sub_indicators_50.csv",
        mime="text/csv",
        use_container_width=True,
    )
