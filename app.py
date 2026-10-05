
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# -------------------------------------------------
# Disease Dataset
# -------------------------------------------------

data = {
    "fever": [
        1, 1, 1, 1, 0, 0, 0, 1, 1, 0,
        0, 1, 0, 1, 0, 0, 1, 1, 0, 0
    ],
    "cough": [
        1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
        1, 1, 0, 1, 0, 0, 1, 0, 1, 0
    ],
    "headache": [
        1, 0, 1, 1, 0, 1, 0, 1, 1, 0,
        0, 1, 1, 0, 0, 1, 0, 1, 0, 1
    ],
    "body_pain": [
        1, 1, 1, 0, 0, 1, 0, 1, 1, 0,
        0, 1, 1, 0, 0, 1, 0, 1, 0, 1
    ],
    "fatigue": [
        1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
        1, 1, 0, 1, 0, 1, 1, 0, 1, 0
    ],
    "sore_throat": [
        1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
        1, 1, 0, 1, 1, 0, 1, 0, 1, 0
    ],
    "runny_nose": [
        0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
        1, 0, 0, 1, 1, 0, 1, 0, 1, 0
    ],
    "nausea": [
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
        1, 0, 1, 0, 1, 1, 0, 1, 0, 1
    ],
    "vomiting": [
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
        1, 0, 1, 0, 1, 1, 0, 1, 0, 1
    ],
    "disease": [
        "Flu",
        "Common Cold",
        "Food Poisoning",
        "Flu",
        "Common Cold",
        "Migraine",
        "Allergy",
        "Flu",
        "Food Poisoning",
        "Allergy",
        "Food Poisoning",
        "Flu",
        "Migraine",
        "Common Cold",
        "Food Poisoning",
        "Migraine",
        "Allergy",
        "Migraine",
        "Common Cold",
        "Migraine"
    ]
}

df = pd.DataFrame(data)

# -------------------------------------------------
# Prepare Machine Learning Data
# -------------------------------------------------

X = df.drop("disease", axis=1)
y = df["disease"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Create Decision Tree model
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

model.fit(X, y_encoded)

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Disease Prediction Using Machine Learning")

st.write(
    "Select the symptoms you are experiencing and the machine learning "
    "model will provide an estimated prediction."
)

st.warning(
    "⚠️ This application is for educational purposes only and is not "
    "a substitute for professional medical advice."
)

st.divider()

# -------------------------------------------------
# Symptoms
# -------------------------------------------------

st.subheader("Select Your Symptoms")

col1, col2 = st.columns(2)

with col1:
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")
    headache = st.checkbox("Headache")
    body_pain = st.checkbox("Body Pain")
    fatigue = st.checkbox("Fatigue")

with col2:
    sore_throat = st.checkbox("Sore Throat")
    runny_nose = st.checkbox("Runny Nose")
    nausea = st.checkbox("Nausea")
    vomiting = st.checkbox("Vomiting")

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("🔍 Predict Disease"):

    symptoms = [[
        int(fever),
        int(cough),
        int(headache),
        int(body_pain),
        int(fatigue),
        int(sore_throat),
        int(runny_nose),
        int(nausea),
        int(vomiting)
    ]]

    prediction = model.predict(symptoms)

    predicted_disease = encoder.inverse_transform(prediction)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(symptoms)[0]

    confidence = max(probabilities) * 100

    st.divider()

    st.subheader("Prediction Result")

    st.success(
        f"Possible condition: **{predicted_disease}**"
    )

    st.info(
        f"Model confidence: **{confidence:.2f}%**"
    )

    st.caption(
        "The confidence shown is the model's statistical output, "
        "not a medical certainty."
    )

    # -------------------------------------------------
    # Show Selected Symptoms
    # -------------------------------------------------

    selected_symptoms = []

    symptom_values = {
        "Fever": fever,
        "Cough": cough,
        "Headache": headache,
        "Body Pain": body_pain,
        "Fatigue": fatigue,
        "Sore Throat": sore_throat,
        "Runny Nose": runny_nose,
        "Nausea": nausea,
        "Vomiting": vomiting
    }

    for symptom, selected in symptom_values.items():

        if selected:
            selected_symptoms.append(symptom)

    st.subheader("Selected Symptoms")

    if selected_symptoms:

        for symptom in selected_symptoms:
            st.write(f"✓ {symptom}")

    else:

        st.write("No symptoms selected.")

    st.divider()

    st.subheader("⚕️ Important")

    st.write(
        "This prediction is only a demonstration of machine learning. "
        "Symptoms can have many different causes. If you are concerned "
        "about your health or symptoms are severe or persistent, consult "
        "a qualified healthcare professional."
    )

