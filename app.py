# import streamlit as st
# import joblib

# # Load model and vectorizer
# modelimport streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from typing import List, Optional

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Disease Detection 🏥")
st.write("Upload a text file or enter symptoms to predict likely diseases, view structured insights, and download results.")

# Medical Disclaimer
st.warning("⚠️ **Medical Disclaimer**: This application is for educational and informational purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.")

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

# --- Keyword-based rules aligned to the above diseases ---
def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _contains_all(text: str, keywords: List[str]) -> bool:
    return all(keyword in text for keyword in keywords)


def classify_by_keywords(symptom_text: str) -> Optional[str]:
    t = symptom_text.lower()

    # Prioritize urgent/emergent patterns
    if _contains_any(t, ["hives", "swelling of lips", "throat tightness", "anaphylaxis"]) and _contains_any(t, ["wheezing", "dizziness", "faint"]):
        return "Anaphylaxis"  # not in header list but supported in dataset

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
    if _contains_any(t, ["burning with urination", "painful urination", "dysuria"]) and _contains_any(t, ["frequency", "urgency"]) :
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
        # First, try rules
        rule_label = classify_by_keywords(text)
        if rule_label is not None:
            predictions.append(rule_label)
            continue
        # Fallback to existing model, but map to known human diseases only
        try:
            model_label = model.predict(vectorizer.transform([text]))[0]
            predictions.append(model_label if model_label in HUMAN_DISEASES else "Unknown")
        except Exception:
            predictions.append("Unknown")
    return predictions

# --- Function to extract insights ---
def generate_insights(df: pd.DataFrame) -> pd.DataFrame:
    insights = df['Predicted Disease'].value_counts().reset_index()
    insights.columns = ['Disease', 'Count']
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
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="prediction_result.csv", mime="text/csv")
        else:
            st.warning("Please enter some text.")

else:
    uploaded_file = st.file_uploader("Upload a .txt file with human symptom descriptions", type=["txt"])
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        text_lines = [line.strip() for line in file_content.strip().split('\n') if line.strip()]

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
            ax.bar(insight_df['Disease'], insight_df['Count'], color='skyblue')
            plt.xticks(rotation=45)
            plt.title("Disease Frequency")
            st.pyplot(fig)

            # Download
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Result CSV", data=csv, file_name="predicted_diseases.csv", mime="text/csv")
 = joblib.load("model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# st.title("Animal Health Condition Classifier")

# user_input = st.text_area("Enter clinical notes or animal symptoms:")

# if st.button("Predict"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         vect_input = vectorizer.transform([user_input])
#         prediction = model.predict(vect_input)[0]
#         st.success(f"Predicted Condition: **{prediction}**")


# import pandas as pd
# from datetime import datetime

# # Initialize session state
# if "record_list" not in st.session_state:
#     st.session_state["record_list"] = []

# # After prediction:
# record = {
#     "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M"),
#     "Notes": user_input,
#     "Condition": prediction,
# }
# st.session_state["record_list"].append(record)

# # Display structured data
# df = pd.DataFrame(st.session_state["record_list"])
# st.write("### Structured Records", df)

# import matplotlib.pyplot as plt

# # Most common conditions
# if not df.empty:
#     condition_counts = df['Condition'].value_counts()
#     st.write("### Condition Frequency")
#     st.bar_chart(condition_counts)

# csv = df.to_csv(index=False).encode('utf-8')
# st.download_button("Download Structured Data as CSV", csv, "structured_data.csv", "text/csv")

# if condition_counts.max() > 5:
#     st.warning(f"⚠️ High frequency of '{condition_counts.idxmax()}' — investigate product usage or quality in affected areas.")


# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# from datetime import datetime

# # Load the model and vectorizer
# model = joblib.load("model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# # Create a list to store structured data
# if "records" not in st.session_state:
#     st.session_state.records = []

# st.title("🐾 Veterinary Health Assistant")
# st.markdown("**AI for Manufacturing - Veesure Animal Health**")
# st.write("Enter clinical notes or animal symptoms:")

# # User input
# user_input = st.text_area("Clinical Notes / Animal Symptoms", height=100)

# if st.button("Predict"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         # Predict condition
#         vectorized = vectorizer.transform([user_input])
#         prediction = model.predict(vectorized)[0]

#         # Store structured record
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         st.session_state.records.append({
#             "Timestamp": timestamp,
#             "Symptoms": user_input,
#             "Predicted Condition": prediction
#         })

#         # Show prediction
#         st.success(f"🧪 Predicted Condition: **{prediction}**")

# # Show structured table
# if st.session_state.records:
#     st.subheader("📋 Structured Data Log")
#     df = pd.DataFrame(st.session_state.records)
#     st.dataframe(df)

#     # Show actionable insights: Condition frequency
#     st.subheader("📊 Most Predicted Conditions")
#     condition_counts = df["Predicted Condition"].value_counts()
    
#     fig, ax = plt.subplots()
#     condition_counts.plot(kind='bar', ax=ax, color='skyblue')
#     ax.set_xlabel("Condition")
#     ax.set_ylabel("Frequency")
#     ax.set_title("Frequency of Predicted Conditions")
#     st.pyplot(fig)


# import streamlit as st
# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load model and vectorizer
# model = pickle.load(open("model.pkl", "rb"))
# vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# st.set_page_config(page_title="Animal Symptom Classifier", layout="centered")
# st.title("🐾 AI for Animal Symptom Analysis")

# st.markdown("Enter animal symptoms or clinical notes to get predictions, download structured data, and view analytics.")

# # File upload for multiple inputs
# st.write("### Upload a Text File (.txt) with Multiple Records")
# uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

# input_texts = []

# if uploaded_file is not None:
#     input_texts = uploaded_file.read().decode("utf-8").splitlines()

# # Manual text input
# st.write("### Or Enter a Single Symptom / Note")
# user_input = st.text_input("Enter symptom/clinical note here:")

# if user_input:
#     input_texts.append(user_input)

# records = []

# if st.button("Predict Conditions"):
#     if not input_texts:
#         st.warning("Please upload a file or enter text manually.")
#     else:
#         for text in input_texts:
#             if text.strip():
#                 X = vectorizer.transform([text])
#                 prediction = model.predict(X)[0]
#                 record = {
#                     "Text": text,
#                     "Predicted Condition": prediction,
#                     "Record Type": "User Input" if text == user_input else "File Upload"
#                 }
#                 records.append(record)

#         if records:
#             # Convert to DataFrame
#             df = pd.DataFrame(records)

#             st.success("✅ Prediction Complete!")
#             st.write("### Structured Data Table")
#             st.dataframe(df)

#             # Download button
#             csv = df.to_csv(index=False)
#             st.download_button(
#                 label="📥 Download Structured Data as CSV",
#                 data=csv,
#                 file_name="structured_data.csv",
#                 mime="text/csv"
#             )

#             # Bar chart for prediction count
#             st.write("### 📊 Prediction Overview")
#             chart_data = df["Predicted Condition"].value_counts()
#             fig, ax = plt.subplots()
#             chart_data.plot(kind="bar", color="skyblue", ax=ax)
#             plt.xlabel("Predicted Condition")
#             plt.ylabel("Count")
#             plt.title("Distribution of Predicted Conditions")
#             st.pyplot(fig)


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Disease Detection 🏥")
st.write("Upload a text file or enter symptoms to predict likely diseases, view structured insights, and download results.")

# Medical Disclaimer
st.warning("⚠️ **Medical Disclaimer**: This application is for educational and informational purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.")

# --- Function to preprocess and predict ---
def predict_condition(text_list):
    vectors = vectorizer.transform(text_list)
    predictions = model.predict(vectors)
    return predictions

# --- Function to extract insights ---
def generate_insights(df):
    insights = df['Predicted Condition'].value_counts().reset_index()
    insights.columns = ['Condition', 'Count']
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
            pred = predict_condition([user_text])[0]
            st.success(f"Predicted Disease: **{pred}**")
            result_df = pd.DataFrame([{"Input Text": user_text, "Predicted Disease": pred}])
            st.dataframe(result_df)

            st.subheader("📊 Actionable Insights")
            st.write("Since it's only one entry, no chart is generated.")
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="prediction_result.csv", mime="text/csv")
        else:
            st.warning("Please enter some text.")

else:
    uploaded_file = st.file_uploader("Upload a .txt file with human symptom descriptions", type=["txt"])
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        text_lines = [line.strip() for line in file_content.strip().split('\n') if line.strip()]

        if len(text_lines) == 0:
            st.warning("The uploaded file is empty or invalid.")
        else:
            predictions = predict_condition(text_lines)
            result_df = pd.DataFrame({"Input Text": text_lines, "Predicted Disease": predictions})
            st.subheader("🗂️ Structured Table")
            st.dataframe(result_df)

            st.subheader("📊 Actionable Insights")
            insight_df = result_df['Predicted Disease'].value_counts().reset_index()
            insight_df.columns = ['Disease', 'Count']
            st.dataframe(insight_df)

            # Plot
            fig, ax = plt.subplots()
            ax.bar(insight_df['Disease'], insight_df['Count'], color='skyblue')
            plt.xticks(rotation=45)
            plt.title("Disease Frequency")
            st.pyplot(fig)

            # Download
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Result CSV", data=csv, file_name="predicted_diseases.csv", mime="text/csv")

