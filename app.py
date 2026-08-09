import io
import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. إعدادات الصفحة العامة
# ==========================================
st.set_page_config(
    page_title="مؤشر القوة اللغوية CCPLI v4.0 الشامل",
    layout="wide",
    page_icon="🏛️",
)

DATA_FILE = "ccpli_data.json"

# ==========================================
# 2. الثوابت وقواعد البيانات المرجعية الافتراضية
# ==========================================
DIMENSIONS_LIST = [
    "1. الثقل الديمغرافي",
    "2. القوة الاقتصادية",
    "3. النفوذ الجيوسياسي",
    "4. الإنتاج العلمي",
    "5. المحتوى الرقمي",
    "6. الجاذبية التعليمية",
    "7. الحضور الإعلامي",
    "8. الترجمة والتنوع",
    "9. رأس المال الثقافي",
    "10. المرونة المؤسسية",
]

DEFAULT_10_LANGUAGES = {
    "الإنجليزية": [85, 98, 95, 96, 94, 98, 90, 92, 88, 84],
    "الصينية": [98, 92, 80, 88, 85, 70, 72, 65, 75, 70],
    "الإسبانية": [82, 75, 78, 65, 70, 82, 78, 72, 85, 75],
    "العربية": [78, 65, 72, 52, 58, 68, 68, 60, 75, 55],
    "الفرنسية": [65, 78, 88, 80, 75, 88, 82, 85, 90, 80],
    "الهندية": [92, 70, 60, 58, 62, 45, 60, 50, 70, 50],
    "الروسية": [60, 68, 75, 72, 68, 60, 65, 68, 72, 62],
    "البرتغالية": [68, 65, 58, 55, 60, 55, 62, 58, 68, 60],
    "البنغالية": [75, 50, 40, 35, 42, 30, 45, 38, 55, 42],
    "الألمانية": [50, 85, 75, 88, 78, 75, 70, 88, 80, 82],
}

DEFAULT_10_PARAMS = {
    "الإنجليزية": {"cagr": 0.03, "n_eff": 45.0},
    "الصينية": {"cagr": 0.08, "n_eff": 15.0},
    "الإسبانية": {"cagr": 0.05, "n_eff": 21.0},
    "العربية": {"cagr": 0.12, "n_eff": 22.0},
    "الفرنسية": {"cagr": 0.06, "n_eff": 29.0},
    "الهندية": {"cagr": 0.09, "n_eff": 5.0},
    "الروسية": {"cagr": 0.02, "n_eff": 12.0},
    "البرتغالية": {"cagr": 0.04, "n_eff": 9.0},
    "البنغالية": {"cagr": 0.05, "n_eff": 3.0},
    "الألمانية": {"cagr": 0.02, "n_eff": 8.0},
}

RECOMMENDATIONS_DB = {
    "1. الثقل الديمغرافي": (
        "توسيع شبكات التبادل الثقافي ومراكز تعلم اللغة بالخارج، وتسهيل الإقامة"
        " والاندماج للمتعلمين الجدد."
    ),
    "2. القوة الاقتصادية": (
        "إلزام التوطين اللغوي في عقود الاستثمار والمنتجات التجارية، وربط استخدام"
        " اللغة في سوق العمل والتجارة الإلكترونية."
    ),
    "3. النفوذ الجيوسياسي": (
        "تعزيز السعي لزيادة اعتماد اللغة في المنظمات والدبلوماسية الإقليمية"
        " والدولية كأداة تواصل رسمية."
    ),
    "4. الإنتاج العلمي": (
        "تأسيس حوافز مالية ونقاط مكافأة للنشر العلمي باللغة، ودعم الدوريات"
        " العالمية المحكمة وتكليف برامج الترجمة العلمية."
    ),
    "5. المحتوى الرقمي": (
        "دعم وتطوير نماذج اللغات الضخمة (LLMs) بالذكاء الاصطناعي، ورعاية مبادرات"
        " إثراء المحتوى الرقمي الحر والموسوعات."
    ),
    "6. الجاذبية التعليمية": (
        "تطوير معايير حديثة لمناهج تعليم اللغة لغير الناطقين بها (TAFL)،"
        " وتقديم منح دراسية دولية لجذب الطلاب الأجانب."
    ),
    "7. الحضور الإعلامي": (
        "إطلاق منصات بث رقمي وسينمائي بدقة عالية، ودعم صناع المحتوى والتأثير"
        " الإعلامي عابر الحدود."
    ),
    "8. الترجمة والتنوع": (
        "تطوير محركات ترجمة آلية دقيقة تدعم اللغة، ورعاية مشاريع الترجمة"
        " العكسية للأمهات الفكرية والعلمية."
    ),
    "9. رأس المال الثقافي": (
        "استثمار التراث المادي واللامادي عبر معارض وسياحة ثقافية دولية تسوق"
        " الهوية اللغوية عالمياً."
    ),
    "10. المرونة المؤسسية": (
        "سن تشريعات حازمة لحماية اللغة وتفعيلها، وتحديث المجامع اللغوية لتوليد"
        " المصطلحات والتأقلم السريع مع التطورات."
    ),
}

SOURCES_DB = {
    "1. الثقل الديمغرافي": {
        "المصادر الرسمية": [
            {
                "name": "Ethnologue (SIL International)",
                "url": "https://www.ethnologue.com/",
            },
            {
                "name": "شعبة السكان بالأمم المتحدة (UN DESA)",
                "url": "https://www.un.org/development/desa/pd/",
            },
            {
                "name": "CIA World Factbook",
                "url": "https://www.cia.gov/the-world-factbook/",
            },
        ],
        "المؤشرات المقاسة": (
            "الناطقون الأصليون (L1)، المتحدثون كتقاطعات ثانية (L2)، والتوزيع"
            " الجغرافي والشتات."
        ),
        "دورية التحديث": "سنوية / كل سنتين",
    },
    "2. القوة الاقتصادية": {
        "المصادر الرسمية": [
            {
                "name": "مؤشرات التنمية للبنك الدولي (World Bank WDI)",
                "url": (
                    "https://databank.worldbank.org/source/world-development-indicators"
                ),
            },
            {
                "name": "قاعدة بيانات آفاق الاقتصاد العالمي (IMF WEO)",
                "url": (
                    "https://www.imf.org/en/Publications/SPROLLS/world-economic-outlook-databases"
                ),
            },
            {
                "name": "مرصد التعقيد الاقتصادي (OEC)",
                "url": "https://oec.world/",
            },
        ],
        "المؤشرات المقاسة": (
            "الناتج المحلي الإجمالي المعادل للشراء (GDP-PPP) والقوة الشرائية"
            " للمتحدثين."
        ),
        "دورية التحديث": "سنوية",
    },
    "3. النفوذ الجيوسياسي": {
        "المصادر الرسمية": [
            {
                "name": "اللغات الرسمية في الأمم المتحدة (UN)",
                "url": "https://www.un.org/en/our-work/official-languages",
            },
            {
                "name": "مؤشر الدبلوماسية العالمية (Lowy Institute)",
                "url": "https://globaldiplomacyindex.lowyinstitute.org/",
            },
            {
                "name": "Elcano Global Presence Index",
                "url": (
                    "https://www.globalpresence.realinstitutoelcano.org/"
                ),
            },
        ],
        "المؤشرات المقاسة": (
            "الاعتماد الدولي، البعثات الدبلوماسية، والتأثير الجيوسياسي."
        ),
        "دورية التحديث": "سنوية",
    },
    "4. الإنتاج العلمي": {
        "المصادر الرسمية": [
            {
                "name": "قاعدة بيانات Scopus (Elsevier)",
                "url": "https://www.scopus.com/",
            },
            {
                "name": "Web of Science (Clarivate)",
                "url": (
                    "https://clarivate.com/products/scientific-and-academic-research/research-discovery-and-workflow-solutions/web-of-science/"
                ),
            },
            {
                "name": "Google Scholar Metrics",
                "url": (
                    "https://scholar.google.com/citations?view_op=top_venues"
                ),
            },
        ],
        "المؤشرات المقاسة": (
            "الأوراق البحثية المحكمة، الاستشهادات المرجعية، والمجلات المعتمدة."
        ),
        "دورية التحديث": "مستمرة / سنوية",
    },
    "5. المحتوى الرقمي": {
        "المصادر الرسمية": [
            {
                "name": "W3Techs (Web Technology Surveys)",
                "url": (
                    "https://w3techs.com/technologies/overview/content_language"
                ),
            },
            {
                "name": "إحصائيات ومستودعات ويكيميديا (Wikimedia Data)",
                "url": "https://stats.wikimedia.org/",
            },
            {
                "name": "أرشيف Common Crawl الرقمي",
                "url": "https://commoncrawl.org/",
            },
        ],
        "المؤشرات المقاسة": (
            "نسبة لغة المواقع الإلكترونية، حجم ويكيبيديا، والتمثيل في مجموعات"
            " بيانات الذكاء الاصطناعي."
        ),
        "دورية التحديث": "شهرية",
    },
    "6. الجاذبية التعليمية": {
        "المصادر الرسمية": [
            {
                "name": "معهد اليونسكو للإحصاء (UNESCO UIS)",
                "url": "http://uis.unesco.org/",
            },
            {
                "name": "تقارير IIE Open Doors للطلاب الدوليين",
                "url": "https://opendoorsdata.org/",
            },
            {
                "name": (
                    "معايير المجلس الأمريكي لتعليم اللغات الأجنبية (ACTFL)"
                ),
                "url": "https://www.actfl.org/",
            },
        ],
        "المؤشرات المقاسة": (
            "متعلمو اللغة لغير الناطقين بها، الطلاب الوافدون، ومراكز"
            " الاختبارات."
        ),
        "دورية التحديث": "سنوية",
    },
    "7. الحضور الإعلامي": {
        "المصادر الرسمية": [
            {
                "name": "قطاع الاتصال والمعلومات باليونسكو",
                "url": "https://www.unesco.org/en/communication-information",
            },
            {
                "name": "Euromonitor International",
                "url": "https://www.euromonitor.com/",
            },
        ],
        "المؤشرات المقاسة": (
            "الوصول الإخباري للفضائيات، البث الرقمي، والإنتاج الدرامي والسينمائي."
        ),
        "دورية التحديث": "سنوية",
    },
    "8. الترجمة والتنوع": {
        "المصادر الرسمية": [
            {
                "name": "قاعدة بيانات الترجمة Index Translationum (اليونسكو)",
                "url": "https://www.unesco.org/xtrans/",
            },
            {
                "name": "المنظمة العالمية للملكية الفكرية (WIPO)",
                "url": "https://www.wipo.int/portal/en/index.html",
            },
            {
                "name": "تقارير سوق اللغات (Slator Intelligence)",
                "url": "https://slator.com/",
            },
        ],
        "المؤشرات المقاسة": (
            "الكتب المترجمة، براءات الاختراع المودعة، ودقة محركات الترجمة."
        ),
        "دورية التحديث": "سنوية",
    },
    "9. رأس المال الثقافي": {
        "المصادر الرسمية": [
            {
                "name": "مركز التراث العالمي (UNESCO WHC)",
                "url": "https://whc.unesco.org/",
            },
            {
                "name": "مؤشر القوة الناعمة العالمي (Brand Finance)",
                "url": (
                    "https://brandfinance.com/know-how/reports/global-soft-power-index"
                ),
            },
            {
                "name": "Anholt-Ipsos Nation Brands Index",
                "url": (
                    "https://www.ipsos.com/en/anholt-ipsos-nation-brands-index"
                ),
            },
        ],
        "المؤشرات المقاسة": (
            "التراث المادي واللامادي، والصناعات الثقافية الخلاقة."
        ),
        "دورية التحديث": "سنوية",
    },
    "10. المرونة المؤسسية": {
        "المصادر الرسمية": [
            {
                "name": "مقياس EGIDS لحيوية اللغات (Ethnologue)",
                "url": "https://www.ethnologue.com/about/language-status",
            },
            {
                "name": "الأكاديمية الدولية للقانون اللغوي (IALL)",
                "url": "https://www.language-policy.org/",
            },
        ],
        "المؤشرات المقاسة": (
            "التشريعات الحامية للغة، نشاط المجامع اللغوية، ودرجة الحيوية."
        ),
        "دورية التحديث": "كل 3 - 5 سنوات",
    },
}


# ==========================================
# 3. آليات الحفظ والدعم الدائم
# ==========================================
def save_persistent_data():
  """حفظ البيانات في ملف JSON محلي بشكل دائم"""
  data_to_save = {
      "languages_data": st.session_state.languages_data,
      "languages_params": st.session_state.languages_params,
      "weights": st.session_state.weights,
      "ahp_matrix": st.session_state.ahp_matrix.tolist(),
  }
  with open(DATA_FILE, "w", encoding="utf-8") as f:
    json.dump(data_to_save, f, ensure_ascii=False, indent=2)


def load_persistent_data():
  """تحميل البيانات من ملف JSON إن وجد، أو استخدام الافتراضيات"""
  if os.path.exists(DATA_FILE):
    try:
      with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        st.session_state.languages_data = data.get(
            "languages_data", DEFAULT_10_LANGUAGES.copy()
        )
        st.session_state.languages_params = data.get(
            "languages_params", DEFAULT_10_PARAMS.copy()
        )
        st.session_state.weights = data.get("weights", [0.10] * 10)
        st.session_state.ahp_matrix = np.array(
            data.get("ahp_matrix", np.ones((10, 10)).tolist()), dtype=float
        )
        return
    except Exception:
      pass

  st.session_state.languages_data = DEFAULT_10_LANGUAGES.copy()
  st.session_state.languages_params = DEFAULT_10_PARAMS.copy()
  st.session_state.weights = [0.10] * 10
  st.session_state.ahp_matrix = np.ones((10, 10), dtype=float)


if "data_loaded" not in st.session_state:
  load_persistent_data()
  st.session_state.data_loaded = True


# ==========================================
# 4. الدوال الرياضية وتصنيف اللغات
# ==========================================
def calculate_ccpli_v4(scores, weights, cagr_3yr, n_eff, n_max=50):
  """حساب مؤشر القوة اللغوية CCPLI v4.0 المعتمد"""
  scores = np.array(scores, dtype=float)
  weights = np.array(weights, dtype=float)

  log_sum = np.sum(weights * np.log(1 + scores))
  s_geom = np.exp(log_sum) - 1.0

  m_momentum = 1.0 + 0.05 * np.tanh(cagr_3yr)
  geo_ratio = (n_eff - 1.0) / (n_max - 1.0) if n_max > 1 else 0.0
  m_geo = 1.0 + 0.10 * np.clip(geo_ratio, 0.0, 1.0)

  raw_score = s_geom * m_momentum * m_geo
  final_ccpli = min(100.0, raw_score)

  return (
      round(s_geom, 2),
      round(m_momentum, 4),
      round(m_geo, 4),
      round(final_ccpli, 2),
  )


def get_language_tier(score):
  """تحديد مكانة اللغة وتصنيفها الاستراتيجي بناءً على درجة CCPLI v4.0"""
  if score >= 80.0:
    return "🌐 Tier 1: لغة عالمية مهيمنة (Global Superpower)"
  elif score >= 65.0:
    return "🌍 Tier 2: لغة عالمية كبرى (Major Global Power)"
  elif score >= 50.0:
    return "🏛️ Tier 3: لغة إقليمية مؤثّرة (Regional Power)"
  else:
    return "📍 Tier 4: لغة ذات نفوذ محلي / محدد (Local / Emerging)"


def get_all_languages_leaderboard():
  """إنشاء جدول الترتيب العالمي المحسوب لجميع اللغات المسجلة"""
  records = []
  for lang, scores in st.session_state.languages_data.items():
    params = st.session_state.languages_params.get(
        lang, {"cagr": 0.05, "n_eff": 10.0}
    )
    s_geom, m_mom, m_geo, final_ccpli = calculate_ccpli_v4(
        scores, st.session_state.weights, params["cagr"], params["n_eff"]
    )
    mean_raw = round(np.mean(scores), 1)
    tier_label = get_language_tier(final_ccpli)

    records.append({
        "اللغة": lang,
        "الدرجة الكلية (CCPLI v4.0)": final_ccpli,
        "المكانة والتصنيف الاستراتيجي": tier_label,
        "المتوسط البسيط للأبعاد": mean_raw,
        "التجميع الهندسي (S_geom)": s_geom,
        "معامل الزخم": m_mom,
        "معامل التشتت": m_geo,
    })
  df_lb = pd.DataFrame(records).sort_values(
      by="الدرجة الكلية (CCPLI v4.0)", ascending=False
  )
  df_lb.insert(0, "الترتيب العالمي", range(1, 1 + len(df_lb)))
  return df_lb


def compute_ahp_weights(matrix):
  """حساب متجه الأوزان واختبار الاتساق لمصفوفة AHP"""
  n = matrix.shape[0]
  col_sum = matrix.sum(axis=0)
  norm_matrix = matrix / col_sum
  weights = norm_matrix.mean(axis=1)

  weighted_sum = np.dot(matrix, weights)
  lambda_max = np.mean(weighted_sum / weights)

  ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
  ri_10 = 1.49
  cr = ci / ri_10 if ri_10 > 0 else 0.0

  return weights, round(lambda_max, 4), round(ci, 4), round(cr, 4)


def export_languages_to_excel(dimensions_list):
  """تحويل بيانات اللغات إلى ملف Excel بالذاكرة والتنزيل المباشر"""
  rows = []
  for lang, scores in st.session_state.languages_data.items():
    params = st.session_state.languages_params.get(
        lang, {"cagr": 0.05, "n_eff": 10.0}
    )
    row = {"اللغة": lang}
    for dim, score in zip(dimensions_list, scores):
      row[dim] = score
    row["CAGR"] = params.get("cagr", 0.05)
    row["N_eff"] = params.get("n_eff", 10.0)
    rows.append(row)

  df_export = pd.DataFrame(rows)
  output = io.BytesIO()
  with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df_export.to_excel(writer, index=False, sheet_name="بيانات_اللغات_CCPLI")
  return output.getvalue()


def import_languages_from_excel(uploaded_file, dimensions_list):
  """استيراد بيانات اللغات وتحديث ذاكرة الجلسة تلقائياً"""
  try:
    if uploaded_file.name.endswith(".csv"):
      df_imported = pd.read_csv(uploaded_file)
    else:
      df_imported = pd.read_excel(uploaded_file)

    if "اللغة" not in df_imported.columns:
      st.error("❌ الملف لا يحتوي على عمود باسم 'اللغة'. يرجى مراجعة المنسق.")
      return False

    for _, row in df_imported.iterrows():
      lang_name = str(row["اللغة"]).strip()
      if not lang_name or pd.isna(lang_name):
        continue

      scores = []
      for dim in dimensions_list:
        val = (
            row[dim]
            if dim in row and not pd.isna(row[dim])
            else 50.0
        )
        scores.append(float(val))

      cagr_val = (
          float(row["CAGR"])
          if "CAGR" in row and not pd.isna(row["CAGR"])
          else 0.05
      )
      neff_val = (
          float(row["N_eff"])
          if "N_eff" in row and not pd.isna(row["N_eff"])
          else 10.0
      )

      st.session_state.languages_data[lang_name] = scores
      st.session_state.languages_params[lang_name] = {
          "cagr": cagr_val,
          "n_eff": neff_val,
      }

    save_persistent_data()
    return True
  except Exception as e:
    st.error(f"❌ حدث خطأ أثناء معالجة الملف: {str(e)}")
    return False


def plot_language_comparison(df_dimensions, chart_type="Radar Chart"):
  """رسم مقارنة البصمة اللغوية للمؤشر"""
  categories = list(df_dimensions.index)
  languages = list(df_dimensions.columns)
  color_palette = px.colors.qualitative.Set1

  if chart_type == "Radar Chart":
    fig = go.Figure()
    for idx, lang in enumerate(languages):
      values = df_dimensions[lang].tolist()
      values_closed = values + [values[0]]
      categories_closed = categories + [categories[0]]
      color = color_palette[idx % len(color_palette)]

      fig.add_trace(
          go.Scatterpolar(
              r=values_closed,
              theta=categories_closed,
              fill="toself",
              name=lang,
              line_color=color,
              opacity=0.45,
              hovertemplate=(
                  "<b>%{theta}</b><br>اللغة: "
                  + lang
                  + "<br>الدرجة: %{r:.1f}/100<extra></extra>"
              ),
          )
      )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100], tickfont=dict(size=10)
            ),
            angularaxis=dict(direction="clockwise", tickfont=dict(size=11)),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        title=dict(
            text="<b>بصمة القوة اللغوية (Radar Comparison)</b>",
            x=0.5,
            font=dict(size=16),
        ),
        height=550,
        margin=dict(l=50, r=50, t=60, b=80),
    )
  else:
    fig = go.Figure()
    for idx, lang in enumerate(languages):
      color = color_palette[idx % len(color_palette)]
      fig.add_trace(
          go.Bar(
              x=categories,
              y=df_dimensions[lang],
              name=lang,
              marker_color=color,
              hovertemplate=(
                  "<b>%{x}</b><br>اللغة: "
                  + lang
                  + "<br>الدرجة: %{y:.1f}/100<extra></extra>"
              ),
          )
      )
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        title=dict(
            text="<b>مقارنة الأداء المباشر للأبعاد الـ 10</b>",
            x=0.5,
            font=dict(size=16),
        ),
        height=480,
        margin=dict(l=40, r=40, t=60, b=100),
    )
  return fig


@st.dialog("📚 التوثيق المباشر لمصادر البيانات")
def show_sources_dialog(dim_name):
  """نافذة منبثقة تفاعلية للمصادر بروابط خارجية موجهة"""
  data = SOURCES_DB.get(dim_name, {})
  st.subheader(f"البعد: {dim_name}")
  st.markdown(
      "##### 🌐 جهات الرصد وقواعد البيانات الدولية (روابط رسمية مباشرة):"
  )

  sources = data.get("المصادر الرسمية", [])
  for src in sources:
    st.link_button(
        label=f"🔗 {src['name']}", url=src["url"], use_container_width=True
    )

  st.markdown("---")
  st.markdown(
      f"**📈 المؤشرات الفرعية المقاسة:**\n{data.get('المؤشرات المقاسة', 'غ/م')}"
  )
  st.markdown(
      f"**⏱️ دورية تحديث البيانات:** `{data.get('دورية التحديث', 'غ/م')}`"
  )


# ==========================================
# 5. الشريط الجانبي (Sidebar)
# ==========================================
st.sidebar.title("🎛️ إدارة المحاكاة والبيانات")

if st.sidebar.button("🔄 إعادة ضبط المصنع (إعادة البيانات للافتراضي)"):
  st.session_state.languages_data = DEFAULT_10_LANGUAGES.copy()
  st.session_state.languages_params = DEFAULT_10_PARAMS.copy()
  st.session_state.weights = [0.10] * 10
  st.session_state.ahp_matrix = np.ones((10, 10), dtype=float)
  save_persistent_data()
  st.sidebar.success("تم إعادة ضبط القيمة المرجعية الأصلية ورسخ الملف!")
  st.rerun()

with st.sidebar.expander("➕ إضافة لغة جديدة", expanded=False):
  new_lang_name = st.text_input(
      "اسم اللغة الجديدة:", placeholder="مثال: الإيطالية"
  )
  if st.button("إضافة اللغة الآن", use_container_width=True):
    if (
        new_lang_name
        and new_lang_name not in st.session_state.languages_data
    ):
      st.session_state.languages_data[new_lang_name] = [50] * 10
      st.session_state.languages_params[new_lang_name] = {
          "cagr": 0.05,
          "n_eff": 10.0,
      }
      save_persistent_data()
      st.success(f"تمت إضافة ({new_lang_name}) وحفظ البيانات بنجاح!")
      st.rerun()

available_languages = list(st.session_state.languages_data.keys())
selected_lang = st.sidebar.selectbox(
    "اختر اللغة للتعديل والمحاكاة:",
    options=available_languages,
    index=0 if available_languages else None,
    key="selected_lang_key",
)

if len(available_languages) > 1 and selected_lang:
  with st.sidebar.expander("🗑️ حذف اللغة الحالية"):
    if st.button(f"تأكيد حذف ({selected_lang})", use_container_width=True):
      del st.session_state.languages_data[selected_lang]
      del st.session_state.languages_params[selected_lang]
      save_persistent_data()
      st.rerun()

st.sidebar.markdown("---")

if selected_lang:
  st.sidebar.subheader(f"⚙️ معاملات ({selected_lang})")
  cagr_val = st.sidebar.number_input(
      "معدل النمو المركب (CAGR 3yr)",
      value=st.session_state.languages_params[selected_lang]["cagr"],
      step=0.01,
      format="%.2f",
      key=f"cagr_{selected_lang}",
  )
  neff_val = st.sidebar.number_input(
      "عدد الدول الفعالة (N_eff)",
      value=st.session_state.languages_params[selected_lang]["n_eff"],
      min_value=1.0,
      max_value=50.0,
      step=1.0,
      key=f"neff_{selected_lang}",
  )

  if (
      cagr_val != st.session_state.languages_params[selected_lang]["cagr"]
      or neff_val != st.session_state.languages_params[selected_lang]["n_eff"]
  ):
    st.session_state.languages_params[selected_lang]["cagr"] = cagr_val
    st.session_state.languages_params[selected_lang]["n_eff"] = neff_val
    save_persistent_data()

  st.sidebar.markdown("---")
  st.sidebar.subheader("🛠️ تعديل درجات الأبعاد الـ 10")
  current_scores = st.session_state.languages_data[selected_lang]
  updated_scores = []
  scores_changed = False

  for i, dim in enumerate(DIMENSIONS_LIST):
    val = st.sidebar.slider(
        label=dim,
        min_value=0,
        max_value=100,
        value=current_scores[i],
        step=1,
        key=f"slider_{selected_lang}_{i}",
    )
    if val != current_scores[i]:
      scores_changed = True
    updated_scores.append(val)

  if scores_changed:
    st.session_state.languages_data[selected_lang] = updated_scores
    save_persistent_data()

st.sidebar.markdown("---")
st.sidebar.title("📁 إدارة قاعدة البيانات")

excel_bytes = export_languages_to_excel(DIMENSIONS_LIST)
st.sidebar.download_button(
    label="📥 تنزيل قاعدة البيانات (Excel)",
    data=excel_bytes,
    file_name="CCPLI_v4_Languages_Database.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

with st.sidebar.expander("📤 استيراد لغات من ملف خارجي"):
  uploaded_file = st.file_uploader(
      "اختر ملف Excel أو CSV:",
      type=["xlsx", "xls", "csv"],
      key="excel_uploader",
  )
  if uploaded_file is not None:
    if st.button("تأكيد الاستيراد ودمج البيانات", use_container_width=True):
      if import_languages_from_excel(uploaded_file, DIMENSIONS_LIST):
        st.success("✅ تم استيراد البيانات وتحديث الملف بنجاح!")
        st.rerun()

# ==========================================
# 6. الواجهة الرئيسية والتبويبات
# ==========================================
st.title("🏛️ لوحة تحكم ومحاكي مؤشر CCPLI v4.0")
st.caption(
    "نظام المحاكاة التفاعلي المتقدم — مع تحديد **تصنيف ومكانة اللغة"
    " الاستراتيجية** والربط الموحد للبيانات"
)

tab_sim, tab_stats, tab_ahp, tab_docs = st.tabs([
    "📊 المحاكاة والتحليل الاستراتيجي",
    "📈 الإحصاءات والترتيب العالمي للغات",
    "⚖️ حاسبة أوزان AHP",
    "📖 دليل المصادر والمنهجية",
])

# ------------------------------------------
# التبويب الأول: المحاكاة والتحليل الاستراتيجي
# ------------------------------------------
with tab_sim:
  df_current = pd.DataFrame(
      st.session_state.languages_data, index=DIMENSIONS_LIST
  )

  current_w_pct = [f"{round(w * 100, 1)}%" for w in st.session_state.weights]
  st.info(
      "💡 **الأوزان المطبقة حالياً على المؤشر:** "
      + " | ".join(
          [f"**د{i+1}:** {p}" for i, p in enumerate(current_w_pct)]
      )
  )

  st.subheader("📊 بصمة القوة اللغوية والتحليل التفاعلي")
  col_control, col_chart = st.columns([1, 3])

  with col_control:
    chart_type = st.radio(
        "نمط الرسم البياني:", ["Radar Chart", "Grouped Bar Chart"]
    )
    langs_to_show = st.multiselect(
        "اللغات المعروضة للمقارنة:",
        options=list(df_current.columns),
        default=list(df_current.columns)[:4],
        key="visible_langs_multiselect",
    )

  with col_chart:
    if langs_to_show:
      fig = plot_language_comparison(
          df_current[langs_to_show], chart_type=chart_type
      )
      st.plotly_chart(fig, use_container_width=True)

  if selected_lang:
    st.markdown("---")
    st.subheader(f"🧮 التقييم الاستراتيجي الشامل لـ ({selected_lang})")

    s_geom, m_mom, m_geo, final_score = calculate_ccpli_v4(
        st.session_state.languages_data[selected_lang],
        st.session_state.weights,
        st.session_state.languages_params[selected_lang]["cagr"],
        st.session_state.languages_params[selected_lang]["n_eff"],
    )

    tier_status = get_language_tier(final_score)
    df_lb_all = get_all_languages_leaderboard()
    rank_pos = (
        df_lb_all[df_lb_all["اللغة"] == selected_lang]["الترتيب العالمي"].values[
            0
        ]
        if selected_lang in df_lb_all["اللغة"].values
        else "-"
    )

    st.success(
        f"🏅 **مكانة اللغة والتصنيف الاستراتيجي:** `{tier_status}` | **الترتيب"
        f" العالمي:** المركز `{rank_pos}` من بين {len(df_lb_all)} لغات"
    )

    res1, res2, res3, res4 = st.columns(4)
    res1.metric("التجميع الهندسي (S_geom)", f"{s_geom}")
    res2.metric("معامل الزخم (M_momentum)", f"{m_mom}")
    res3.metric("معامل التشتت (M_geo)", f"{m_geo}")
    res4.metric("الدرجة الكلية (CCPLI v4.0)", f"{final_score} / 100")

    st.markdown("---")
    st.subheader(
        f"💡 التشخيص والتوصيات الاستراتيجية الحية لـ ({selected_lang})"
    )

    simulated_df = pd.DataFrame({
        "البعد": DIMENSIONS_LIST,
        "الدرجة": st.session_state.languages_data[selected_lang],
    })
    critical_gaps = simulated_df[simulated_df["الدرجة"] < 70].sort_values(
        by="الدرجة", ascending=True
    )

    if not critical_gaps.empty:
      st.warning(
          f"تم رصد **{len(critical_gaps)} أبعاد فرعية** دون عتبة الكفاءة (70"
          " نقطة):"
      )
      for _, row in critical_gaps.iterrows():
        dim_name = row["البعد"]
        score = row["الدرجة"]
        rec_text = RECOMMENDATIONS_DB.get(
            dim_name, "تطوير سياسات واضحة لجبر القصور."
        )
        with st.expander(
            f"⚠️ **{dim_name}** — الدرجة الحالية: **{score} / 100**",
            expanded=True,
        ):
          st.markdown(f"**التوصية الموجهة لجبر القصور:** {rec_text}")
    else:
      st.success("🎉 ممتاز! جميع الأبعاد الـ 10 تتجاوز عتبة الـ 70 نقطة.")

# ------------------------------------------
# التبويب الثاني: الإحصاءات والترتيب العالمي
# ------------------------------------------
with tab_stats:
  st.header("📈 الترتيب العالمي ومكانة اللغات العشر الكبرى")
  st.caption("جدول متكامل يربط بين الدرجات المركبة والتصنيف الاستراتيجي للغات")

  df_leaderboard = get_all_languages_leaderboard()

  st.subheader("🏆 جدول الترتيب العالمي لمؤشر CCPLI v4.0")
  st.dataframe(
      df_leaderboard.style.highlight_max(
          axis=0, color="#d4edda", subset=["الدرجة الكلية (CCPLI v4.0)"]
      ),
      use_container_width=True,
      hide_index=True,
  )

  st.markdown("---")

  col_bar, col_pie = st.columns([2, 1])

  with col_bar:
    fig_rank = px.bar(
        df_leaderboard,
        x="اللغة",
        y="الدرجة الكلية (CCPLI v4.0)",
        color="الدرجة الكلية (CCPLI v4.0)",
        color_continuous_scale="Viridis",
        text="الدرجة الكلية (CCPLI v4.0)",
        title="<b>ترتيب اللغات حسب الدرجة الكلية المركبة للـ CCPLI</b>",
    )
    fig_rank.update_traces(
        texttemplate="%{text:.1f}", textposition="outside"
    )
    fig_rank.update_layout(yaxis=dict(range=[0, 110]), height=420)
    st.plotly_chart(fig_rank, use_container_width=True)

  with col_pie:
    fig_pie = px.pie(
        df_leaderboard,
        names="اللغة",
        values="الدرجة الكلية (CCPLI v4.0)",
        title="<b>الحصة النسبية للقوة اللغوية</b>",
        hole=0.4,
    )
    fig_pie.update_layout(height=420)
    st.plotly_chart(fig_pie, use_container_width=True)

  st.markdown("---")

  st.subheader("🔥 الخريطة الحرارية لتوزيع الأداء عبر الأبعاد الـ 10")
  df_heatmap_data = pd.DataFrame(
      st.session_state.languages_data, index=DIMENSIONS_LIST
  )
  fig_heatmap = px.imshow(
      df_heatmap_data.T,
      labels=dict(x="البُعد الاستراتيجي", y="اللغة", color="الدرجة"),
      x=DIMENSIONS_LIST,
      y=list(df_heatmap_data.columns),
      color_continuous_scale="YlGnBu",
      aspect="auto",
      text_auto=True,
  )
  fig_heatmap.update_layout(height=480, margin=dict(l=40, r=40, t=30, b=80))
  st.plotly_chart(fig_heatmap, use_container_width=True)

  st.markdown("---")

  st.subheader("📐 الملخص الإحصائي الوصفي للأبعاد العشرة")
  col_desc, col_box = st.columns([1, 1])

  with col_desc:
    st.markdown("##### 📋 المؤشرات الإحصائية العامة للقطاع اللغوي:")
    df_stats_summary = df_heatmap_data.T.describe().T[
        ["mean", "std", "min", "50%", "max"]
    ]
    df_stats_summary.columns = [
        "المتوسط",
        "الانحراف المعياري",
        "الحد الأدنى",
        "الوسيط",
        "الحد الأقصى",
    ]
    st.dataframe(
        df_stats_summary.round(2), use_container_width=True, height=380
    )

  with col_box:
    st.markdown("##### 📦 رسم التشتت والمدى الربيعي لكل بُعد (Boxplot):")
    df_melted = df_heatmap_data.reset_index().melt(
        id_vars="index", var_name="اللغة", value_name="الدرجة"
    )
    df_melted.rename(columns={"index": "البعد"}, inplace=True)

    fig_box = px.box(
        df_melted,
        x="البعد",
        y="الدرجة",
        color="البعد",
        points="all",
        title="<b>مستوى التباين والتفاوت الدولي في كل بُعد</b>",
    )
    fig_box.update_layout(
        showlegend=False,
        height=380,
        xaxis=dict(tickangle=45),
        margin=dict(l=20, r=20, t=40, b=80),
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ------------------------------------------
# التبويب الثالث: حاسبة أوزان AHP
# ------------------------------------------
with tab_ahp:
  st.subheader(
      "⚖️ تحديد أوزان الأبعاد الـ 10 بأسلوب المقارنات الزوجية (AHP)"
  )
  st.markdown("""
    تتيح طريقة **التحليل الهرمي (AHP)** حساب وزن كل بعد معنوي بناءً على مصفوفة تفضيل زوجية $10 \\times 10$.
    * **مقياس الساتي (Saaty Scale 1-9):** القيمة 1 تعني تساوي الأهمية، 3 أهمية معتدلة، 5 أهمية قوية، 7 أهمية قوية جداً، 9 أهمية مطلقة.
    """)

  preset = st.selectbox(
      "اختر نموذج أوزان جاهز أو استخدم المصفوفة المخصصة:",
      [
          "أوزان متساوية (Equal Weights 10%)",
          "نموذج التركيز البحثي والأكاديمي",
          "مصفوفة AHP التفاعلية المخصصة",
      ],
  )

  if preset == "أوزان متساوية (Equal Weights 10%)":
    calculated_weights = [0.10] * 10
    cr_val = 0.0
    st.info("تم تطبيق أوزان متساوية بواقع 10% لكل بعد من الأبعاد الـ 10.")

  elif preset == "نموذج التركيز البحثي والأكاديمي":
    calculated_weights = [
        0.08,
        0.08,
        0.09,
        0.20,
        0.15,
        0.12,
        0.08,
        0.07,
        0.07,
        0.06,
    ]
    cr_val = 0.03
    st.info(
        "تم تطبيق نموذج يركز على الإنتاج العلمي (20%) والمحتوى الرقمي"
        " (15%) والجاذبية التعليمية (12%)."
    )

  else:
    st.markdown("##### 📝 تحرير مصفوفة المقارنات الزوجية $10 \\times 10$")
    short_dims = [f"د{i+1}" for i in range(10)]
    df_matrix = pd.DataFrame(
        st.session_state.ahp_matrix, index=short_dims, columns=short_dims
    )

    edited_df = st.data_editor(
        df_matrix, use_container_width=True, height=380
    )

    matrix_np = edited_df.to_numpy(dtype=float)
    for i in range(10):
      matrix_np[i, i] = 1.0
      for j in range(i + 1, 10):
        if matrix_np[i, j] <= 0:
          matrix_np[i, j] = 1.0
        matrix_np[j, i] = 1.0 / matrix_np[i, j]

    st.session_state.ahp_matrix = matrix_np
    calculated_weights, lambda_max, ci_val, cr_val = compute_ahp_weights(
        matrix_np
    )

    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
    c_col1.metric("القيمة الذاتية (λ_max)", f"{lambda_max}")
    c_col2.metric("مؤشر الاتساق (CI)", f"{ci_val}")
    c_col3.metric("نسبة الاتساق (CR)", f"{cr_val}")

    if cr_val <= 0.10:
      c_col4.success("✅ اتساق مقبول (CR ≤ 0.10)")
    else:
      c_col4.error("❌ اتساق غير مقبول (CR > 0.10)")

  st.markdown("---")
  st.subheader("📊 المتجه الأولي للأوزان المحسوبة ($w_i$)")

  df_w_chart = pd.DataFrame({
      "البعد": DIMENSIONS_LIST,
      "الوزن النسبي (%)": [w * 100 for w in calculated_weights],
  })

  fig_w = go.Figure(
      go.Bar(
          x=df_w_chart["البعد"],
          y=df_w_chart["الوزن النسبي (%)"],
          marker_color="#1f77b4",
          text=[f"{w:.1f}%" for w in df_w_chart["الوزن النسبي (%)"]],
          textposition="auto",
      )
  )
  fig_w.update_layout(
      yaxis=dict(
          title="الوزن النسبي (%)",
          range=[0, max(df_w_chart["الوزن النسبي (%)"]) + 5],
      ),
      height=400,
      margin=dict(l=40, r=40, t=30, b=80),
  )
  st.plotly_chart(fig_w, use_container_width=True)

  st.markdown("---")
  if st.button(
      "🚀 اعتماد وتطبيق هذه الأوزان على مؤشر CCPLI v4.0",
      type="primary",
      use_container_width=True,
  ):
    st.session_state.weights = calculated_weights
    save_persistent_data()
    st.success("تم تحديث وحفظ أوزان المؤشر بنجاح!")

# ------------------------------------------
# التبويب الرابع: دليل المصادر والمنهجية
# ------------------------------------------
with tab_docs:
  st.header("📖 التوثيق المنهجي ومصادر البيانات الدولية")
  st.caption(
      "اضغطي على أي بُعد لفتح النافذة المباشرة للمصادر والروابط الرسمية:"
  )

  cols = st.columns(2)
  for idx, (dim_name, info) in enumerate(SOURCES_DB.items()):
    col = cols[idx % 2]
    with col:
      with st.container(border=True):
        st.write(f"**{dim_name}**")
        st.caption(
            "**المؤسسات الرئيسة:**"
            f" {', '.join([s['name'] for s in info['المصادر الرسمية']])}"
        )
        if st.button(
            "🔍 فتح تفاصيل المصادر والروابط",
            key=f"btn_modal_docs_{idx}",
            use_container_width=True,
        ):
          show_sources_dialog(dim_name)
