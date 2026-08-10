# ============================================================
# LANGUAGE POWER INDEX V4
# D1.3 — DIGITAL / COMPUTATIONAL LANGUAGE CAPABILITY
# MASTER AUDIT ENGINE
# ============================================================

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, List, Dict
import pandas as pd
import numpy as np


# ============================================================
# 1. DATA STATUS
# ============================================================

class DataStatus(str, Enum):
    VERIFIED = "Verified"
    DERIVED = "Derived"
    NA_PARTIAL = "NA / Partial Coverage"
    PENDING = "Pending Audit"
    REJECTED = "Rejected"


VALID_CALCULATION_STATUS = {
    DataStatus.VERIFIED,
    DataStatus.DERIVED
}


# ============================================================
# 2. LANGUAGES
# ============================================================

LANGUAGES = {
    "ar": "Arabic",
    "en": "English",
    "fr": "French",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "hi": "Hindi",
    "pt": "Portuguese",
    "bn": "Bengali",
}


# ============================================================
# 3. AUDIT RECORD
# ============================================================

@dataclass
class AuditRecord:

    dimension: str
    indicator: str
    language: str

    raw_value: Optional[float] = None

    source: Optional[str] = None
    version: Optional[str] = None
    checkpoint: Optional[str] = None

    evidence: Optional[str] = None
    methodology: Optional[str] = None

    status: DataStatus = DataStatus.PENDING

    derived_value: Optional[float] = None
    normalized_score: Optional[float] = None

    note: Optional[str] = None


# ============================================================
# 4. AUDIT LOG
# ============================================================

class AuditLog:

    def __init__(self):
        self.records = []

    def add(
        self,
        action: str,
        dimension: str,
        indicator: str,
        language: str,
        old_value=None,
        new_value=None,
        reason=None
    ):

        self.records.append({
            "action": action,
            "dimension": dimension,
            "indicator": indicator,
            "language": language,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason
        })

    def dataframe(self):

        return pd.DataFrame(self.records)


# ============================================================
# 5. VALIDATE RAW DATA
# ============================================================

def validate_record(record: AuditRecord):

    # No source → cannot be verified
    if record.status == DataStatus.VERIFIED:

        if not record.source:
            record.status = DataStatus.PENDING
            record.note = "Verified status rejected: source missing."

        elif not record.version:
            record.status = DataStatus.PENDING
            record.note = "Verified status rejected: version missing."

        elif record.raw_value is None:
            record.status = DataStatus.PENDING
            record.note = "Verified status rejected: raw value missing."

    # Rejected values can never enter calculations
    if record.status == DataStatus.REJECTED:

        record.derived_value = None
        record.normalized_score = None

    return record


# ============================================================
# 6. CHECK WHETHER VALUE CAN ENTER CALCULATION
# ============================================================

def is_calculable(record: AuditRecord):

    return (
        record.status in VALID_CALCULATION_STATUS
        and record.raw_value is not None
    )


# ============================================================
# 7. POSITIVE MIN-MAX NORMALIZATION
# ============================================================

def minmax_positive(series: pd.Series):

    series = pd.to_numeric(series, errors="coerce")

    valid = series.dropna()

    if len(valid) == 0:
        return pd.Series(index=series.index, dtype=float)

    xmin = valid.min()
    xmax = valid.max()

    # Avoid division by zero
    if xmax == xmin:
        return pd.Series(
            [50.0 if not pd.isna(x) else np.nan for x in series],
            index=series.index
        )

    return 100 * (series - xmin) / (xmax - xmin)


# ============================================================
# 8. D1.3.1 — UD RESOURCE COVERAGE
# ============================================================

def calculate_d131(treebank_df):

    """
    Required columns:

    Language
    Treebank
    Tokens
    SyntacticWords
    UDVersion
    Status
    Source
    """

    df = treebank_df.copy()

    # Only verified v2.15 records
    valid = df[
        (df["Status"] == DataStatus.VERIFIED.value)
        & (df["UDVersion"] == "v2.15")
    ].copy()

    aggregate = (
        valid
        .groupby("Language", as_index=False)
        .agg(
            Tokens=("Tokens", "sum"),
            SyntacticWords=("SyntacticWords", "sum")
        )
    )

    aggregate["D1.3.1_Raw"] = aggregate["SyntacticWords"]

    aggregate["D1.3.1_Normalized"] = minmax_positive(
        aggregate["D1.3.1_Raw"]
    )

    return aggregate


# ============================================================
# 9. D1.3.2 — NLP FRAMEWORK SUPPORT
# ============================================================

FRAMEWORKS = [
    "Stanza",
    "spaCy",
    "NLTK",
    "Flair"
]

FUNCTIONS = [
    "Tokenizer",
    "POS",
    "Lemmatizer",
    "Parser",
    "NER"
]


def calculate_d132(matrix_df):

    """
    Required columns:

    Language
    Framework
    Function
    Status
    Source
    Version
    OperationalModel
    """

    df = matrix_df.copy()

    # Official evidence AND operational pretrained model
    df["ValidCell"] = (
        (df["Status"] == DataStatus.VERIFIED.value)
        & (df["OperationalModel"] == True)
        & (df["Source"].notna())
        & (df["Version"].notna())
    )

    df["CellScore"] = df["ValidCell"].astype(int)

    result = (
        df.groupby("Language", as_index=False)
        .agg(
            TotalCells=("CellScore", "count"),
            VerifiedCells=("CellScore", "sum")
        )
    )

    result["D1.3.2_Raw"] = (
        result["VerifiedCells"] / 20
    )

    result["D1.3.2_Normalized"] = minmax_positive(
        result["D1.3.2_Raw"]
    )

    return result


# ============================================================
# 10. D1.3.3 — TOKENIZATION
# ============================================================

def calculate_tfr(tokens, words):

    if words is None or words <= 0:
        return np.nan

    return tokens / words


def calculate_btr(bytes_count, tokens):

    if tokens is None or tokens <= 0:
        return np.nan

    return bytes_count / tokens


def calculate_tpi(english_tokens, language_tokens):

    if language_tokens is None or language_tokens <= 0:
        return np.nan

    return english_tokens / language_tokens


def calculate_d133(token_df):

    """
    Required:

    Language
    Tokens
    Words
    Bytes
    EnglishTokens
    """

    df = token_df.copy()

    df["TFR"] = df.apply(
        lambda r: calculate_tfr(
            r["Tokens"],
            r["Words"]
        ),
        axis=1
    )

    df["BTR"] = df.apply(
        lambda r: calculate_btr(
            r["Bytes"],
            r["Tokens"]
        ),
        axis=1
    )

    df["TPI"] = df.apply(
        lambda r: calculate_tpi(
            r["EnglishTokens"],
            r["Tokens"]
        ),
        axis=1
    )

    # Positive orientation:
    # Higher BTR / TPI = better efficiency
    #
    # TFR is reversed because lower fragmentation is better.

    df["TFR_Score"] = 100 - minmax_positive(df["TFR"])

    df["BTR_Score"] = minmax_positive(df["BTR"])

    df["TPI_Score"] = minmax_positive(df["TPI"])

    df["D1.3.3_Raw"] = (
        df["TFR_Score"]
        + df["BTR_Score"]
        + df["TPI_Score"]
    ) / 3

    return df


# ============================================================
# 11. D1.3.4 — MACHINE TRANSLATION
# ============================================================

def calculate_d134(mt_df):

    """
    Required:

    Language
    Checkpoint
    Direction
    Split
    Metric
    Score
    Status
    """

    df = mt_df.copy()

    # Strict metadata validation
    df["MetadataValid"] = (
        df["Checkpoint"].notna()
        & df["Direction"].notna()
        & df["Split"].eq("FLORES-200 devtest")
        & df["Metric"].isin(["spBLEU", "chrF++"])
        & df["Score"].notna()
        & df["Status"].eq(DataStatus.VERIFIED.value)
    )

    valid = df[df["MetadataValid"]].copy()

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "Language",
                "D1.3.4_Raw",
                "D1.3.4_Normalized"
            ]
        )

    # Equal metric weighting
    result = (
        valid
        .groupby("Language", as_index=False)
        .agg(
            D1_3_4_Raw=("Score", "mean")
        )
    )

    result["D1.3.4_Normalized"] = minmax_positive(
        result["D1_3_4_Raw"]
    )

    return result


# ============================================================
# 12. D1.3.5 — NLU
# ============================================================

def calculate_nlu_score(
    belebele,
    xnli,
    xnli_available=True
):

    if belebele is None:
        return np.nan

    # Full coverage
    if xnli_available and xnli is not None:

        return (
            0.50 * belebele
            + 0.50 * xnli
        )

    # Partial coverage
    return belebele


def calculate_d135(nlu_df):

    """
    Required:

    Language
    BelebeleAccuracy
    XNLIAccuracy
    XNLIAvailable
    """

    df = nlu_df.copy()

    df["NLU_Raw"] = df.apply(
        lambda r: calculate_nlu_score(
            r["BelebeleAccuracy"],
            r["XNLIAccuracy"],
            r["XNLIAvailable"]
        ),
        axis=1
    )

    df["D1.3.5_Normalized"] = minmax_positive(
        df["NLU_Raw"]
    )

    return df


# ============================================================
# 13. SPEARMAN RANK STABILITY
# ============================================================

def rank_stability(
    baseline: pd.Series,
    sensitivity: pd.Series
):

    combined = pd.concat(
        [baseline, sensitivity],
        axis=1
    ).dropna()

    if len(combined) < 2:
        return np.nan

    return combined.iloc[:, 0].corr(
        combined.iloc[:, 1],
        method="spearman"
    )


# ============================================================
# 14. RANK DIFFERENCE
# ============================================================

def rank_difference(
    baseline: pd.Series,
    sensitivity: pd.Series
):

    base_rank = baseline.rank(
        ascending=False,
        method="min"
    )

    sens_rank = sensitivity.rank(
        ascending=False,
        method="min"
    )

    return (base_rank - sens_rank).abs()


# ============================================================
# 15. SENSITIVITY ANALYSIS
# ============================================================

def nlu_sensitivity(
    belebele,
    xnli
):

    baseline = (
        0.50 * belebele
        + 0.50 * xnli
    )

    reading_bias = (
        0.70 * belebele
        + 0.30 * xnli
    )

    inference_bias = (
        0.30 * belebele
        + 0.70 * xnli
    )

    return {
        "Baseline_50_50": baseline,
        "S1a_70_30": reading_bias,
        "S1b_30_70": inference_bias
    }


# ============================================================
# 16. FINAL D1.3 COMPOSITE
# ============================================================

def calculate_d13_composite(df):

    components = [
        "D1.3.1",
        "D1.3.2",
        "D1.3.3",
        "D1.3.4",
        "D1.3.5"
    ]

    available = [
        c for c in components
        if c in df.columns
    ]

    if not available:
        raise ValueError(
            "No valid D1.3 components available."
        )

    # Equal weighting ONLY if this is the frozen Codebook rule.
    # Otherwise replace with the official weights.

    df["D1.3_Raw"] = df[available].mean(
        axis=1,
        skipna=True
    )

    df["D1.3_Coverage"] = (
        df[available].notna().sum(axis=1)
        / len(available)
    )

    return df


# ============================================================
# 17. MASTER AUDIT VALIDATION
# ============================================================

def audit_master_sheet(df):

    report = []

    for _, row in df.iterrows():

        status = row.get("Status")

        if status == DataStatus.PENDING.value:

            report.append({
                "Language": row.get("Language"),
                "Problem": "Pending Audit",
                "Action": "Exclude from derived calculations"
            })

        elif status == DataStatus.REJECTED.value:

            report.append({
                "Language": row.get("Language"),
                "Problem": "Rejected value",
                "Action": "Exclude permanently"
            })

        elif row.get("RawValue") is None:

            report.append({
                "Language": row.get("Language"),
                "Problem": "Missing raw value",
                "Action": "Do not impute"
            })

    return pd.DataFrame(report)


# ============================================================
# 18. EXPORT
# ============================================================

def export_master_audit(
    master_df,
    audit_log,
    filename="D1.3_Master_Audit.xlsx"
):

    with pd.ExcelWriter(
        filename,
        engine="openpyxl"
    ) as writer:

        master_df.to_excel(
            writer,
            sheet_name="Master Extraction",
            index=False
        )

        audit_log.dataframe().to_excel(
            writer,
            sheet_name="Audit Log",
            index=False
        )

    print(
        f"Master Audit exported to: {filename}"
    )


# ============================================================
# 19. MAIN PIPELINE
# ============================================================

def run_d13_pipeline(
    d131_df,
    d132_df,
    d133_df,
    d134_df,
    d135_df
):

    print("=" * 60)
    print("LANGUAGE POWER INDEX V4")
    print("D1.3 MASTER AUDIT PIPELINE")
    print("=" * 60)

    # -----------------------------------------
    # D1.3.1
    # -----------------------------------------

    d131 = calculate_d131(d131_df)

    # -----------------------------------------
    # D1.3.2
    # -----------------------------------------

    d132 = calculate_d132(d132_df)

    # -----------------------------------------
    # D1.3.3
    # -----------------------------------------

    d133 = calculate_d133(d133_df)

    # -----------------------------------------
    # D1.3.4
    # -----------------------------------------

    d134 = calculate_d134(d134_df)

    # -----------------------------------------
    # D1.3.5
    # -----------------------------------------

    d135 = calculate_d135(d135_df)

    # -----------------------------------------
    # Merge
    # -----------------------------------------

    final = pd.DataFrame({
        "Language": list(LANGUAGES.keys())
    })

    datasets = [
        d131,
        d132,
        d133,
        d134,
        d135
    ]

    for dataset in datasets:

        if "Language" not in dataset.columns:
            continue

        final = final.merge(
            dataset,
            on="Language",
            how="left"
        )

    # -----------------------------------------
    # Rename final component scores
    # -----------------------------------------

    rename_map = {}

    for col in final.columns:

        if col.endswith("_Normalized"):

            base = col.replace(
                "_Normalized",
                ""
            )

            rename_map[col] = base

    final = final.rename(
        columns=rename_map
    )

    # -----------------------------------------
    # Final D1.3
    # -----------------------------------------

    component_cols = [
        "D1.3.1",
        "D1.3.2",
        "D1.3.3",
        "D1.3.4",
        "D1.3.5"
    ]

    available = [
        c for c in component_cols
        if c in final.columns
    ]

    final["D1.3_Coverage"] = (
        final[available]
        .notna()
        .sum(axis=1)
        / len(available)
    )

    final["D1.3_Final"] = (
        final[available]
        .mean(
            axis=1,
            skipna=True
        )
    )

    return final


# ============================================================
# END
# ============================================================
