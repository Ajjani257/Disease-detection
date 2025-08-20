# Disease Detection

An AI-powered application for analyzing human symptom descriptions and predicting likely diseases.

## Features

- **Symptom Analysis**: Enter human symptoms to get predicted diseases
- **Batch Processing**: Upload text files with multiple symptom descriptions
- **Structured Data**: Get organized results in table format
- **Analytics**: View disease frequency charts and insights
- **Data Export**: Download results as CSV files

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Single Symptom Analysis**: Enter individual symptom descriptions in the text area
2. **Batch Analysis**: Upload a .txt file with multiple symptom descriptions (one per line)
3. **View Results**: See predicted diseases in a structured table
4. **Download Data**: Export results as CSV for further analysis

## Requirements

- Python 3.7+
- Streamlit
- Scikit-learn
- Joblib
- Matplotlib
- Pandas

## Disclaimer

This application is for educational and informational purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.