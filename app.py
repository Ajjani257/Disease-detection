import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from typing import List, Optional


# --- Page config ---
st.set_page_config(page_title="Disease Detection", page_icon="🏥", layout="centered")


# --- Try to load model/vectorizer (graceful if missing) ---
model = None
vectorizer = None
model_load_error: Optional[str] = None
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as exc:
    model_load_error = (
        "Model/vectorizer could not be loaded. Keyword rules will be used as fallback only.\n"
        f"Details: {exc}"
    )


st.title("Disease Detection 🏥")
st.write(
    "Upload a text file or enter symptoms to predict likely diseases, view structured insights, and download results."
)

# Medical Disclaimer
st.warning(
    "⚠️ **Medical Disclaimer**: This application is for educational and informational purposes only. "
    "It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns."
)

if model_load_error:
    st.info(model_load_error)


# --- Human disease labels (reference set) ---
HUMAN_DISEASES: List[str] = [
    "Influenza",
    "Common Cold",
    "Eczema",
    "Asthma",
    "Allergic Rhinitis",
    "Anxiety Disorders",
    "Diabetes",
    "Gastroenteritis",
    "Pancreatitis",
    "Rheumatoid Arthritis",
    "Depression",
    "Liver Cancer",
    # Additional common human conditions supported by rules/dataset
    "COVID-19",
    "Migraine",
    "Tension headache",
    "Sinusitis",
    "GERD",
    "Gastritis",
    "Urinary tract infection",
    "Kidney stones",
    "Appendicitis",
    "Pneumonia",
    "Anemia",
    "Hypertension",
    "Hypoglycemia",
    "Hyperthyroidism",
    "Hypothyroidism",
    "Bronchitis",
    "Strep throat",
    "Tonsillitis",
    "Otitis media",
    "Conjunctivitis",
    "Cellulitis",
    "Sciatica",
    "Low back strain",
    "Dermatitis",
]


# Lightweight keyword map for best-guess fallback scoring
DISEASE_KEYWORDS = {
    "Influenza": ["fever", "chills", "body aches", "fatigue", "dry cough"],
    "Common Cold": ["runny nose", "stuffy nose", "congestion", "sneezing", "sore throat"],
    "COVID-19": ["loss of smell", "loss of taste", "fever", "dry cough", "shortness of breath"],
    "Asthma": ["wheezing", "chest tightness", "night", "exercise"],
    "Allergic Rhinitis": ["itchy eyes", "sneezing", "pollen", "dust", "outdoors"],
    "Migraine": ["aura", "photophobia", "phonophobia", "pulsating", "nausea"],
    "Tension headache": ["band-like", "pressure", "stress", "neck tightness"],
    "Sinusitis": ["facial pressure", "tooth pain", "thick", "yellow", "green", "bending forward"],
    "Gastroenteritis": ["vomiting", "diarrhea", "cramps", "stomach cramps", "low fever"],
    "GERD": ["heartburn", "burning chest", "lying down", "sour taste", "regurgitation"],
    "Gastritis": ["epigastric", "upper abdominal", "nausea", "nsaid", "early satiety"],
    "Pancreatitis": ["epigastric", "radiating to back", "severe", "vomiting"],
    "Urinary tract infection": ["dysuria", "burning", "frequency", "urgency"],
    "Kidney stones": ["flank pain", "groin", "hematuria", "blood in urine"],
    "Appendicitis": ["right lower", "migrated", "fever", "rebound"],
    "Pneumonia": ["productive cough", "rusty sputum", "pleuritic", "fever"],
    "Anemia": ["fatigue", "pale", "dizziness", "shortness of breath on exertion"],
    "Hypertension": ["very high blood pressure", "blurred vision", "pounding", "nosebleeds"],
    "Hypoglycemia": ["shakiness", "sweating", "confusion", "relief after eating"],
    "Hyperthyroidism": ["weight loss", "heat intolerance", "palpitations", "tremor"],
    "Hypothyroidism": ["weight gain", "cold intolerance", "dry skin", "constipation"],
    "Diabetes": ["frequent urination", "excessive thirst", "polydipsia", "polyuria", "blurry vision"],
    "Sciatica": ["shooting pain", "radiating down the leg", "tingling"],
    "Low back strain": ["low back pain", "after lifting", "muscle spasm"],
    "Dermatitis": ["itchy rash", "redness", "scaly", "blister", "contact"],
    "Eczema": ["itchy", "dry skin", "patches", "flexural"],
    "Strep throat": ["severe sore throat", "no cough", "swollen tender neck glands"],
    "Tonsillitis": ["sore throat", "difficulty swallowing", "enlarged tonsils", "white patches"],
    "Otitis media": ["ear pain", "fever", "tugging at ear", "trouble hearing"],
    "Conjunctivitis": ["red itchy eyes", "discharge", "crusting", "watery"],
    "Cellulitis": ["warm", "red", "tender", "spreading", "swelling"],
    "Rheumatoid Arthritis": ["joint pain", "swelling", "stiffness", "morning", "bilateral"],
    "Depression": ["sadness", "loss of interest", "anhedonia", "hopeless"],
    "Liver Cancer": ["abdominal pain", "weight loss", "jaundice", "liver", "mass"],
    "Anxiety Disorders": ["panic", "sense of doom", "racing heart", "trembling", "sweating"],
    "Bronchitis": ["productive cough", "mucus", "after a bad cold", "wheeze"],
}


def best_guess_from_keywords(symptom_text: str) -> str:
    text = symptom_text.lower()
    best_label = "Common Cold"
    best_score = -1
    for disease, keywords in DISEASE_KEYWORDS.items():
        score = 1
        for kw in keywords:
            if kw in text:
                score += 1
        if score > best_score:
            best_score = score
            best_label = disease
    return best_label


# --- Keyword-based rules aligned to the above diseases ---
def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _contains_all(text: str, keywords: List[str]) -> bool:
    return all(keyword in text for keyword in keywords)


def classify_by_keywords(symptom_text: str) -> Optional[str]:
    t = symptom_text.lower()

    # Prioritize urgent/emergent patterns
    if _contains_any(t, ["hives", "swelling of lips", "throat tightness", "anaphylaxis"]) and _contains_any(t, ["wheezing", "dizziness", "faint"]):
        return "Anaphylaxis"  # Not listed above, but acceptable as a rule-based urgent label

    # Respiratory infections
    if _contains_all(t, ["fever", "chills"]) and _contains_any(t, ["body aches", "severe fatigue"]) and _contains_any(t, ["dry cough", "cough"]):
        return "Influenza"
    if _contains_any(t, ["runny nose", "stuffy nose", "congestion"]) and _contains_any(t, ["sneezing", "sore throat"]) and not _contains_any(t, ["high fever", "severe"]):
        return "Common Cold"
    if _contains_any(t, ["loss of smell", "loss of taste"]) or (_contains_any(t, ["fever", "dry cough"]) and _contains_any(t, ["shortness of breath", "breathlessness"])):
        return "COVID-19"

    # Airways/Allergy
    if _contains_any(t, ["wheezing", "chest tightness"]) and _contains_any(t, ["exercise", "night", "cold air", "trigger"]):
        return "Asthma"
    if _contains_any(t, ["sneezing", "itchy eyes", "watery rhinorrhea", "clear runny nose"]) and _contains_any(t, ["spring", "pollen", "dust", "cats", "outdoors"]):
        return "Allergic Rhinitis"

    # Headache disorders
    if _contains_any(t, ["aura", "photophobia", "phonophobia", "pulsating", "throbbing"]) and _contains_any(t, ["nausea", "vomiting", "dark room"]):
        return "Migraine"
    if _contains_any(t, ["band-like", "pressure"]) and _contains_any(t, ["forehead", "back of head", "neck tightness", "stress"]):
        return "Tension headache"
    if _contains_any(t, ["facial pressure", "tooth pain"]) and _contains_any(t, ["thick", "yellow", "green"]) and _contains_any(t, ["bending forward", "worse bending"]):
        return "Sinusitis"

    # GI infections/inflammation
    if _contains_any(t, ["vomiting", "watery diarrhea"]) and _contains_any(t, ["stomach cramps", "cramps", "low fever"]):
        return "Gastroenteritis"
    if _contains_any(t, ["after eating", "restaurant", "leftovers", "food poisoning"]) and _contains_any(t, ["vomiting", "diarrhea"]):
        return "Gastroenteritis"
    if _contains_any(t, ["burning chest", "heartburn"]) and _contains_any(t, ["after meals", "lying down", "sour taste", "regurgitation"]):
        return "GERD"
    if _contains_any(t, ["epigastric", "upper abdominal"]) and _contains_any(t, ["nausea"]) and _contains_any(t, ["nsaid", "nsaids", "painkiller", "analgesic", "early satiety"]):
        return "Gastritis"
    if _contains_any(t, ["severe upper abdominal", "epigastric"]) and _contains_any(t, ["radiating to back"]) and _contains_any(t, ["nausea", "vomiting"]):
        return "Pancreatitis"

    # GU
    if _contains_any(t, ["burning with urination", "painful urination", "dysuria"]) and _contains_any(t, ["frequency", "urgency"]):
        return "Urinary tract infection"
    if _contains_any(t, ["flank pain"]) and _contains_any(t, ["radiating to groin", "groin"]) and _contains_any(t, ["blood in urine", "pink-tinged urine", "hematuria"]):
        return "Kidney stones"

    # Surgical abdomen / LRTI
    if _contains_any(t, ["migrated", "moving"]) and _contains_any(t, ["right lower"]) and _contains_any(t, ["fever", "rebound"]):
        return "Appendicitis"
    if _contains_any(t, ["productive cough", "rusty sputum", "pleuritic", "chest pain when breathing"]) and _contains_any(t, ["fever", "shortness of breath"]):
        return "Pneumonia"

    # Hematologic / endocrine
    if _contains_any(t, ["fatigue", "pale skin"]) and _contains_any(t, ["dizziness", "shortness of breath on exertion", "brittle nails", "craving ice"]):
        return "Anemia"
    if _contains_any(t, ["very high blood pressure", "pounding in head", "blurred vision", "nosebleeds"]) or _contains_any(t, ["hypertension"]):
        return "Hypertension"
    if _contains_any(t, ["shakiness", "sweating", "confusion"]) and _contains_any(t, ["after skipping", "improves with juice", "relief after eating"]):
        return "Hypoglycemia"
    if _contains_any(t, ["weight loss", "heat intolerance", "palpitations", "tremor", "trouble sleeping", "anxiety", "sweating"]) and not _contains_any(t, ["cold intolerance", "weight gain"]):
        return "Hyperthyroidism"
    if _contains_any(t, ["fatigue", "weight gain", "cold intolerance", "dry skin", "constipation", "hair thinning", "puffy face"]):
        return "Hypothyroidism"
    if _contains_any(t, ["frequent urination", "excessive thirst", "polydipsia", "polyuria", "blurry vision"]) and not _contains_any(t, ["infection", "uti"]):
        return "Diabetes"

    # MSK / neuro / skin
    if _contains_any(t, ["shooting pain", "radiating down the leg"]) and _contains_any(t, ["worse sitting", "tingling in foot", "sciatica"]):
        return "Sciatica"
    if _contains_any(t, ["low back pain"]) and _contains_any(t, ["after lifting", "muscle spasm"]) and _contains_any(t, ["better with rest", "improves with rest"]):
        return "Low back strain"
    if _contains_any(t, ["itchy rash", "redness", "scaly", "blister"]) and _contains_any(t, ["new soap", "metal", "watchband", "contact"]):
        return "Dermatitis"
    if _contains_any(t, ["itchy", "dry skin", "patches"]) and _contains_any(t, ["chronic", "recurrent", "flexural", "eczema"]):
        return "Eczema"

    # ENT / skin infections
    if _contains_any(t, ["severe sore throat"]) and _contains_any(t, ["no cough", "swollen tender neck glands"]):
        return "Strep throat"
    if _contains_any(t, ["sore throat", "difficulty swallowing"]) and _contains_any(t, ["enlarged tonsils", "white patches", "bad breath", "muffled voice"]):
        return "Tonsillitis"
    if _contains_any(t, ["ear pain"]) and _contains_any(t, ["fever", "tugging at ear", "trouble hearing"]):
        return "Otitis media"
    if _contains_any(t, ["red itchy eyes"]) and _contains_any(t, ["discharge", "crusting", "gritty", "watery"]):
        return "Conjunctivitis"
    if _contains_any(t, ["warm", "red", "tender"]) and _contains_any(t, ["spreading", "swelling"]):
        return "Cellulitis"

    # Rheum / psych / oncology
    if _contains_any(t, ["joint pain", "swelling", "stiffness"]) and _contains_any(t, ["morning", "bilateral", "hands", "wrists", "knees"]):
        return "Rheumatoid Arthritis"
    if _contains_any(t, ["sadness", "loss of interest", "anhedonia"]) or (_contains_any(t, ["fatigue", "sleep problems"]) and _contains_any(t, ["hopeless", "worthless", "depressed"])):
        return "Depression"
    if _contains_any(t, ["abdominal pain", "weight loss"]) and _contains_any(t, ["jaundice", "liver", "hepat", "mass", "swelling"]):
        return "Liver Cancer"
    if _contains_any(t, ["panic", "sense of doom", "racing heart"]) and _contains_any(t, ["trembling", "sweating", "tingling in hands"]):
        return "Anxiety Disorders"

    # Bronchitis
    if _contains_any(t, ["cough with mucus", "productive cough"]) and _contains_any(t, ["after a bad cold", "wheeze", "bronchitis"]):
        return "Bronchitis"

    return None


def predict_diseases(text_list: List[str]) -> List[str]:
    predictions: List[str] = []
    for text in text_list:
        rule_label = classify_by_keywords(text)
        if rule_label is not None:
            predictions.append(rule_label)
            continue

        model_label: Optional[str] = None
        if model is not None and vectorizer is not None:
            try:
                model_label = model.predict(vectorizer.transform([text]))[0]
            except Exception:
                model_label = None

        if model_label in HUMAN_DISEASES:
            predictions.append(model_label)  # type: ignore[arg-type]
        else:
            predictions.append(best_guess_from_keywords(text))

    return predictions


# --- Function to extract insights ---
def generate_insights(df: pd.DataFrame) -> pd.DataFrame:
    insights = df["Predicted Disease"].value_counts().reset_index()
    insights.columns = ["Disease", "Count"]
    return insights


# --- Input Options ---
input_mode = st.radio("Choose Input Type:", ("Enter Text", "Upload File"))

if input_mode == "Enter Text":
    st.write("**Examples of symptom descriptions:**")
    st.write("- Headache, fever, and fatigue")
    st.write("- Chest pain, shortness of breath")
    st.write("- Nausea, vomiting, abdominal pain")
    st.write("- Joint pain, swelling, stiffness")

    user_text = st.text_area("Enter human symptoms here:")
    if st.button("Predict"):
        if user_text.strip():
            pred = predict_diseases([user_text])[0]
            st.success(f"Predicted Disease: **{pred}**")
            result_df = pd.DataFrame([{"Input Text": user_text, "Predicted Disease": pred}])
            st.dataframe(result_df)

            st.subheader("📊 Actionable Insights")
            st.write("Since it's only one entry, no chart is generated.")
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="prediction_result.csv", mime="text/csv")
        else:
            st.warning("Please enter some text.")
else:
    uploaded_file = st.file_uploader("Upload a .txt file with human symptom descriptions", type=["txt"])
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode("utf-8")
        except Exception:
            file_content = ""

        text_lines = [line.strip() for line in file_content.strip().split("\n") if line.strip()]

        if len(text_lines) == 0:
            st.warning("The uploaded file is empty or invalid.")
        else:
            predictions = predict_diseases(text_lines)
            result_df = pd.DataFrame({"Input Text": text_lines, "Predicted Disease": predictions})
            st.subheader("🗂️ Structured Table")
            st.dataframe(result_df)

            st.subheader("📊 Actionable Insights")
            insight_df = generate_insights(result_df)
            st.dataframe(insight_df)

            # Plot
            fig, ax = plt.subplots()
            ax.bar(insight_df["Disease"], insight_df["Count"], color="skyblue")
            plt.xticks(rotation=45, ha="right")
            plt.title("Disease Frequency")
            plt.tight_layout()
            st.pyplot(fig)

            # Download
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Result CSV", data=csv, file_name="predicted_diseases.csv", mime="text/csv")



