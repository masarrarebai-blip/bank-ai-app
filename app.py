import streamlit as st
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Bank AI System", layout="wide")

DATA_PATH = "Classeur1.csv"
DB_PATH = "database_clients.xlsx"

# =========================
# 🎨 UI STYLE
# =========================
st.markdown("""
<style>
body {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);}
.card {
    background: rgba(255,255,255,0.05);
    padding:20px;border-radius:20px;
    backdrop-filter:blur(10px);
    box-shadow:0 8px 32px rgba(0,0,0,0.3);
    color:white;margin-bottom:20px;
}
.title {font-size:20px;font-weight:bold;}
.metric {font-size:28px;font-weight:bold;}
.green {color:#00ff9f;}
.red {color:#ff4b5c;}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="card">
<h2>🏦 Bank AI Dashboard</h2>
<p>Risk Analysis & Insurance Prediction</p>
</div>
""", unsafe_allow_html=True)

# =========================
# ⚡ CACHE MODEL
# =========================
@st.cache_resource
def load_models():
    df = pd.read_csv(DATA_PATH, encoding="latin1").dropna()

    features = [
        'CLT AGE','CLT CATEGORIE','CLT REV MENS NET','MMM',
        'VOLUME DES REVENUS','CREDIT CONSO','CREDIT IMMO',
        'ENC DEBIT','TOT DES CREDITS'
    ]

    X = df[features]
    y_class = df['CLASSE DE RISQUE']
    y_reg = df['PRIME_ASSURANCE']

    scaler = StandardScaler()

    X_train, _, y_train, _ = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train2, _, y_train2, _ = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)
    X_train2_scaled = scaler.fit_transform(X_train2)

    clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500)
    clf.fit(X_train_scaled, y_train)

    reg = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500)
    reg.fit(X_train2_scaled, y_train2)

    return scaler, clf, reg

scaler, clf, reg = load_models()

# =========================
# SESSION STATE
# =========================
if "risk" not in st.session_state:
    st.session_state.risk = None
if "prime" not in st.session_state:
    st.session_state.prime = None
if "status" not in st.session_state:
    st.session_state.status = None

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("📊 Client Info")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 18, 100)
categorie = st.sidebar.number_input("Categorie", 0.0, 10000.0)
income = st.sidebar.number_input("Income", 0.0, 1_000_000.0)
mmm = st.sidebar.number_input("MMM", 0.0, 1_000_000.0)
volume = st.sidebar.number_input("VOLUME DES REVENUS", 0.0, 1_000_000.0)
conso = st.sidebar.number_input("Credit Conso", 0.0, 1_000_000.0)
immo = st.sidebar.number_input("Credit Immo", 0.0, 1_000_000.0)
debit = st.sidebar.number_input("Enc Debit", -1_000_000.0, 1_000_000.0)
total = st.sidebar.number_input("Total Credits", 0.0, 1_000_000.0)

predict_btn = st.button("🚀 Predict")
save_btn = st.button("💾 Save")

# =========================
# DATABASE CLEAN LOAD
# =========================
cols = [
    "Name","Age","Categorie","Income","MMM",
    "VOLUME DES REVENUS","Credit Conso","Credit Immo",
    "Enc Debit","Total Credits","Risk","Prime_DT"
]

if os.path.exists(DB_PATH):
    db = pd.read_excel(DB_PATH)
else:
    db = pd.DataFrame(columns=cols)

# 🔥 CLEAN DUPLICATES
if "PRIME_ASSURANCE" in db.columns:
    db = db.drop(columns=["PRIME_ASSURANCE"])

if "Volume" in db.columns:
    db = db.rename(columns={"Volume":"VOLUME DES REVENUS"})

# =========================
# PREDICT
# =========================
if predict_btn:
    input_data = np.array([[age,categorie,income,mmm,volume,conso,immo,debit,total]])
    input_scaled = scaler.transform(input_data)

    pred = clf.predict(input_scaled)[0]
    st.session_state.risk = pred

    if pred == 1:
        st.session_state.status = "Risqué"
        st.session_state.prime = 0
    else:
        st.session_state.status = "Non Risqué"
        st.session_state.prime = reg.predict(input_scaled)[0]

# =========================
# RESULT
# =========================
if st.session_state.risk is not None:

    color = "red" if st.session_state.risk==1 else "green"

    col1,col2,col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='card'><b>👤 {name}</b></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='card {color}'>{st.session_state.status}</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<div class='card'>💰 {round(st.session_state.prime,2)} DT</div>", unsafe_allow_html=True)

# =========================
# PDF
# =========================
def generate_pdf(client):
    file = f"{client['Name']}.pdf"
    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Bank AI Report", styles["Title"]))
    content.append(Spacer(1,20))

    for k,v in client.items():
        content.append(Paragraph(f"{k}: {v}", styles["Normal"]))
        content.append(Spacer(1,10))

    doc.build(content)
    return file

if st.session_state.risk is not None:
    if st.button("📄 Export PDF"):
        client = {
            "Name":name,
            "Age":age,
            "Risk":st.session_state.status,
            "Prime":round(st.session_state.prime,2)
        }
        pdf = generate_pdf(client)
        with open(pdf,"rb") as f:
            st.download_button("Download PDF",f,file_name=pdf)

# =========================
# SAVE
# =========================
if save_btn:

    if name=="":
        st.warning("Enter name")

    elif st.session_state.risk is None:
        st.warning("Click Predict first")

    else:

        new = {
            "Name":name,
            "Age":age,
            "Categorie":categorie,
            "Income":income,
            "MMM":mmm,
            "VOLUME DES REVENUS":volume,
            "Credit Conso":conso,
            "Credit Immo":immo,
            "Enc Debit":debit,
            "Total Credits":total,
            "Risk":st.session_state.status,
            "Prime_DT":round(st.session_state.prime,2)
        }

        db = pd.concat([db,pd.DataFrame([new])],ignore_index=True)
        db.to_excel(DB_PATH,index=False)

        st.success("Saved ✔")

        st.session_state.clear()
        st.rerun()

# =========================
# DISPLAY
# =========================
st.markdown("### 🗃️ Database")
st.dataframe(db)

# =========================
# DOWNLOAD
# =========================
if os.path.exists(DB_PATH):
    with open(DB_PATH,"rb") as f:
        st.download_button("📥 Download DB",f)