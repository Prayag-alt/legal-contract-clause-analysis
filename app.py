
import streamlit as st
import numpy as np
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Legal Contract Clause Analysis",
    page_icon="⚖️",
    layout="wide"
)


# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_resource
def load_models():

    svm_model = joblib.load(
        "models/svm_model.pkl"
    )

    tfidf = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    mlb = joblib.load(
        "models/mlb.pkl"
    )

    return svm_model, tfidf, mlb


svm_model, tfidf, mlb = load_models()


# ============================================================
# RISK EXTRACTION RULES
# ============================================================

RISK_RULES = {

    "High": {

        "Uncapped Liability": [
            "uncapped liability",
            "unlimited liability",
            "without limitation"
        ],

        "Irrevocable Or Perpetual License": [
            "irrevocable",
            "perpetual",
            "in perpetuity"
        ],

        "Non-Compete": [
            "non-compete",
            "non compete",
            "shall not compete"
        ],

        "Termination For Convenience": [
            "terminate",
            "termination",
            "without cause"
        ]
    },

    "Medium": {

        "Cap On Liability": [
            "liability cap",
            "aggregate liability",
            "maximum liability",
            "shall not exceed"
        ],

        "Audit Rights": [
            "audit",
            "audit rights",
            "inspect records"
        ],

        "Anti-Assignment": [
            "assign",
            "assignment",
            "transfer"
        ],

        "Exclusivity": [
            "exclusive",
            "exclusivity"
        ],

        "Minimum Commitment": [
            "minimum commitment",
            "minimum purchase",
            "minimum quantity"
        ],

        "Insurance": [
            "insurance",
            "insured",
            "coverage"
        ]
    },

    "Low": {

        "Governing Law": [
            "governing law",
            "laws of"
        ],

        "Renewal Term": [
            "renewal",
            "automatically renew"
        ],

        "Notice Period To Terminate Renewal": [
            "notice",
            "written notice"
        ]
    }
}


# ============================================================
# RISK EXTRACTION FUNCTION
# ============================================================

def extract_potential_risks(
    clause_text,
    predicted_labels
):

    text_lower = clause_text.lower()

    risks = []

    for risk_level, categories in RISK_RULES.items():

        for clause_type, keywords in categories.items():

            # Only apply rules for predicted categories
            if clause_type not in predicted_labels:
                continue

            matched_keywords = []

            for keyword in keywords:

                if keyword in text_lower:
                    matched_keywords.append(keyword)

            if matched_keywords:

                risks.append({

                    "Risk Level": risk_level,

                    "Clause Type": clause_type,

                    "Matched Keywords":
                        matched_keywords,

                    "Reason":
                        f"Potential {risk_level.lower()}-risk "
                        f"indicator associated with "
                        f"{clause_type}."
                })

    return risks


# ============================================================
# CLAUSE ANALYSIS FUNCTION
# ============================================================

def analyze_clause(clause_text):

    # Convert text to TF-IDF
    text_vector = tfidf.transform(
        [clause_text]
    )

    # SVM decision scores
    scores = svm_model.decision_function(
        text_vector
    )[0]

    # SVM threshold = 0.0
    predicted_indices = np.where(
        scores >= 0.0
    )[0]

    predicted_labels = [
        mlb.classes_[i]
        for i in predicted_indices
    ]

    # Confidence-like scores
    confidence_scores = {}

    for i in predicted_indices:

        # Sigmoid transformation
        score = np.clip(
            scores[i],
            -10,
            10
        )

        confidence = (
            1 / (1 + np.exp(-score))
        )

        confidence_scores[
            mlb.classes_[i]
        ] = round(
            float(confidence * 100),
            2
        )

    # Risk extraction
    risks = extract_potential_risks(
        clause_text,
        predicted_labels
    )

    return (
        predicted_labels,
        confidence_scores,
        risks
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚖️ Legal Contract Clause Analysis & Risk Extraction System"
)

st.markdown(
    """
    **NLP-based multi-label classification system for legal contract clauses**
    
    The system uses **TF-IDF + Linear SVM** for clause classification
    followed by a **rule-based risk extraction layer**.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System Information")

    st.write(
        "**Dataset:** CUAD"
    )

    st.write(
        "**Task:** Multi-label Clause Classification"
    )

    st.write(
        "**Number of Categories:** 39"
    )

    st.write(
        "**Selected Model:** TF-IDF + Linear SVM"
    )

    st.write(
        "**Micro F1:** 0.8201"
    )

    st.write(
        "**Macro F1:** 0.7358"
    )

    st.write(
        "**Hamming Loss:** 0.0105"
    )

    st.divider()

    st.warning(
        "This system identifies potential risk indicators "
        "and is not a substitute for professional legal advice."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.header("1. Enter Contract Clause")

input_method = st.radio(
    "Choose input method:",
    [
        "Enter Text",
        "Upload TXT File"
    ]
)


clause_text = ""


if input_method == "Enter Text":

    clause_text = st.text_area(
        "Paste legal contract clause here:",
        height=250,
        placeholder=(
            "Example: The Company may terminate "
            "this Agreement without cause upon "
            "thirty days written notice..."
        )
    )


else:

    uploaded_file = st.file_uploader(
        "Upload a TXT file",
        type=["txt"]
    )

    if uploaded_file is not None:

        clause_text = (
            uploaded_file
            .read()
            .decode("utf-8")
        )

        st.text_area(
            "Uploaded clause:",
            clause_text,
            height=250
        )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Clause",
    type="primary",
    use_container_width=True
):

    if not clause_text.strip():

        st.error(
            "Please enter or upload a legal clause."
        )

    else:

        with st.spinner(
            "Analyzing legal clause..."
        ):

            labels, confidence, risks = (
                analyze_clause(clause_text)
            )


        # ====================================================
        # CLAUSE CLASSIFICATION
        # ====================================================

        st.header(
            "2. Clause Classification"
        )

        if labels:

            classification_data = []

            for label in labels:

                classification_data.append({
                    "Clause Type": label,
                    "Confidence Score":
                        f"{confidence[label]:.2f}%"
                })

            classification_df = pd.DataFrame(
                classification_data
            )

            st.dataframe(
                classification_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No clause category exceeded the "
                "SVM decision threshold."
            )


        # ====================================================
        # RISK ANALYSIS
        # ====================================================

        st.header(
            "3. Potential Risk Analysis"
        )

        if risks:

            for risk in risks:

                level = risk["Risk Level"]

                if level == "High":

                    st.error(
                        f"🔴 HIGH RISK — "
                        f"{risk['Clause Type']}"
                    )

                elif level == "Medium":

                    st.warning(
                        f"🟡 MEDIUM RISK — "
                        f"{risk['Clause Type']}"
                    )

                else:

                    st.success(
                        f"🟢 LOW RISK — "
                        f"{risk['Clause Type']}"
                    )

                st.write(
                    f"**Matched Keywords:** "
                    f"{', '.join(risk['Matched Keywords'])}"
                )

                st.write(
                    f"**Explanation:** "
                    f"{risk['Reason']}"
                )

                st.divider()

        else:

            st.success(
                "No predefined potential risk indicators "
                "were detected."
            )


        # ====================================================
        # MODEL COMPARISON
        # ====================================================

        st.header(
            "4. Model Comparison"
        )

        comparison_df = pd.DataFrame({

            "Model": [
                "TF-IDF + Logistic Regression",
                "TF-IDF + Linear SVM",
                "BiLSTM",
                "BERT",
                "Legal-BERT"
            ],

            "Micro F1": [
                0.7900,
                0.8201,
                0.7395,
                0.7388,
                0.7975
            ],

            "Macro F1": [
                0.7182,
                0.7358,
                0.4783,
                0.4017,
                0.5080
            ],

            "Hamming Loss": [
                0.0129,
                0.0105,
                0.0148,
                0.0143,
                0.0114
            ],

            "Subset Accuracy": [
                0.6061,
                0.6782,
                0.6153,
                0.5923,
                0.6897
            ]
        })

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Legal Contract Clause Analysis and Risk Extraction System | "
    "NLP Project"
)
