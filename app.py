import io
import pandas as pd
import numpy as np
import streamlit as st
from enum import Enum

class DataStatus(str, Enum):
    VERIFIED = "Verified"
    DERIVED = "Derived"
    NA_PARTIAL = "NA / Partial Coverage"
    PENDING = "Pending Audit"
    REJECTED = "Rejected"

LANGUAGES = ["ar","en","fr","zh","de","es","ru","hi","pt","bn"]

def minmax_positive(s):
    s = pd.to_numeric(s, errors="coerce")
    v = s.dropna()
    if len(v) == 0: return pd.Series(index=s.index, dtype=float)
    lo, hi = v.min(), v.max()
    if hi == lo: return s.notna().astype(float) * 50
    return 100 * (s-lo)/(hi-lo)

def d131(df):
    req={"Language","Treebank","Tokens","SyntacticWords","UDVersion","Status"}
    miss=req-set(df.columns)
    if miss: raise ValueError(f"D1.3.1 missing: {sorted(miss)}")
    x=df[(df.Status=="Verified") & (df.UDVersion=="v2.15")]
    r=x.groupby("Language",as_index=False).agg(
        Tokens=("Tokens","sum"), SyntacticWords=("SyntacticWords","sum"))
    r["D1.3.1_Raw"]=r.SyntacticWords
    r["D1.3.1"]=minmax_positive(r["D1.3.1_Raw"])
    return r

def d132(df):
    req={"Language","Framework","Function","Status","Source","Version","OperationalModel"}
    miss=req-set(df.columns)
    if miss: raise ValueError(f"D1.3.2 missing: {sorted(miss)}")
    x=df.copy()
    x["ValidCell"]=(
        x.Status.eq("Verified") & x.Source.notna() & x.Version.notna()
        & x.OperationalModel.astype(str).str.lower().eq("true")
    )
    r=x.groupby("Language",as_index=False).agg(
        TotalCells=("ValidCell","size"), VerifiedCells=("ValidCell","sum"))
    r["D1.3.2_Raw"]=r.VerifiedCells/20
    r["D1.3.2"]=minmax_positive(r["D1.3.2_Raw"])
    return r

def d133(df):
    req={"Language","Tokens","Words","Bytes","EnglishTokens"}
    miss=req-set(df.columns)
    if miss: raise ValueError(f"D1.3.3 missing: {sorted(miss)}")
    x=df.copy()
    x["TFR"]=x.Tokens/x.Words.replace(0,np.nan)
    x["BTR"]=x.Bytes/x.Tokens.replace(0,np.nan)
    x["TPI"]=x.EnglishTokens/x.Tokens.replace(0,np.nan)
    x["TFR_Score"]=100-minmax_positive(x.TFR)
    x["BTR_Score"]=minmax_positive(x.BTR)
    x["TPI_Score"]=minmax_positive(x.TPI)
    x["D1.3.3"]=x[["TFR_Score","BTR_Score","TPI_Score"]].mean(axis=1)
    return x

def d134(df):
    req={"Language","Checkpoint","Direction","Split","Metric","Score","Status"}
    miss=req-set(df.columns)
    if miss: raise ValueError(f"D1.3.4 missing: {sorted(miss)}")
    x=df[
        df.Checkpoint.notna() & df.Direction.notna()
        & df.Split.eq("FLORES-200 devtest")
        & df.Metric.isin(["spBLEU","chrF++"])
        & pd.to_numeric(df.Score,errors="coerce").notna()
        & df.Status.eq("Verified")
    ]
    r=x.groupby("Language",as_index=False).agg(D1_3_4_Raw=("Score","mean"))
    if len(r): r["D1.3.4"]=minmax_positive(r.D1_3_4_Raw)
    else: r["D1.3.4"]=pd.Series(dtype=float)
    return r

def d135(df):
    req={"Language","BelebeleAccuracy","XNLIAccuracy","XNLIAvailable"}
    miss=req-set(df.columns)
    if miss: raise ValueError(f"D1.3.5 missing: {sorted(miss)}")
    x=df.copy()
    x["NLU_Raw"]=np.where(
        x.XNLIAvailable.astype(str).str.lower().eq("true") & x.XNLIAccuracy.notna(),
        .5*x.BelebeleAccuracy+.5*x.XNLIAccuracy,
        x.BelebeleAccuracy
    )
    x["D1.3.5"]=minmax_positive(x.NLU_Raw)
    return x

def merge_final(parts):
    out=pd.DataFrame({"Language":LANGUAGES})
    for p in parts:
        if "Language" in p: out=out.merge(p[["Language"]+[c for c in p if c.startswith("D1.3.") and c.count(".")>=2]],on="Language",how="left")
    comps=[c for c in ["D1.3.1","D1.3.2","D1.3.3","D1.3.4","D1.3.5"] if c in out]
    out["D1.3_Coverage"]=out[comps].notna().sum(axis=1)/5
    out["D1.3_Final"]=out[comps].mean(axis=1,skipna=True)
    return out

st.set_page_config(page_title="Language Power Index V4 — D1.3",layout="wide")
st.title("Language Power Index V4")
st.subheader("D1.3 — Digital / Computational Language Capability")
st.info("ارفعي ملفات CSV الخمسة ثم اضغطي «تشغيل التدقيق والحساب».")

files={}
labels={
"d131":"D1.3.1 — UD Treebanks",
"d132":"D1.3.2 — NLP Framework Matrix",
"d133":"D1.3.3 — Tokenization",
"d134":"D1.3.4 — Machine Translation",
"d135":"D1.3.5 — NLU"
}
for k,v in labels.items():
    files[k]=st.sidebar.file_uploader(v,type="csv",key=k)

st.markdown("**الأعمدة المطلوبة:**")
st.code("""D1.3.1: Language,Treebank,Tokens,SyntacticWords,UDVersion,Status
D1.3.2: Language,Framework,Function,Status,Source,Version,OperationalModel
D1.3.3: Language,Tokens,Words,Bytes,EnglishTokens
D1.3.4: Language,Checkpoint,Direction,Split,Metric,Score,Status
D1.3.5: Language,BelebeleAccuracy,XNLIAccuracy,XNLIAvailable""")

if st.button("تشغيل التدقيق والحساب",type="primary"):
    if not all(files.values()):
        st.error("ارفعي ملفات CSV الخمسة أولاً.")
        st.stop()
    try:
        parts=[
            d131(pd.read_csv(files["d131"])),
            d132(pd.read_csv(files["d132"])),
            d133(pd.read_csv(files["d133"])),
            d134(pd.read_csv(files["d134"])),
            d135(pd.read_csv(files["d135"]))
        ]
        final=merge_final(parts)
        st.success("تم تشغيل D1.3.")
        st.dataframe(final,use_container_width=True)
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="openpyxl") as w:
            for i,p in enumerate(parts,1): p.to_excel(w,f"D1.3.{i}",index=False)
            final.to_excel(w,"D1.3 Final",index=False)
        st.download_button("تحميل Master Audit Excel",buf.getvalue(),
            "D1.3_Master_Audit.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error("يوجد خطأ في ملف البيانات.")
        st.exception(e)
