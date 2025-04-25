import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="📊CustomerGuard360", layout="wide")
st.title("📶 CustomerGuard360")

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_list = joblib.load("feature_list.pkl")
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Sidebar 
page = st.sidebar.radio("Navigate to", ["Dashboard", "Single Prediction", "Batch Prediction"])

# This is Preprocessing Function 
def preprocess_df(raw_df):
    df = raw_df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)
    df_enc = pd.get_dummies(df, drop_first=True)
    df_enc = df_enc.reindex(columns=feature_list, fill_value=0)
    df_enc[num_cols] = scaler.transform(df_enc[num_cols])
    return df_enc
if page == "Dashboard":
    st.header("📊 CustomerGuard360 Analytics Dashboard")
    uploaded = st.file_uploader("Upload Dataset", type="csv")
    if uploaded:
        raw = pd.read_csv(uploaded)
        raw.columns = raw.columns.str.strip()
        raw['Churn'] = raw['Churn'].map({'No': 0, 'Yes': 1})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", raw.shape[0])
        c2.metric("Churn Rate", f"{raw['Churn'].mean():.2%}")
        c3.metric("Avg Tenure", f"{raw['tenure'].mean():.1f} months")
        c4.metric("Avg Monthly Charge", f"${raw['MonthlyCharges'].mean():.2f}")
        st.markdown("---")

        fig1 = px.pie(raw, names='Churn', hole=0.4, title='Churn vs Stay')
        st.plotly_chart(fig1, use_container_width=True)

        cr = raw.groupby('Contract')['Churn'].mean().reset_index()
        fig2 = px.bar(cr, x='Contract', y='Churn', text='Churn', title='Churn Rate by Contract')
        fig2.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")

        X = preprocess_df(raw)
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        shap.summary_plot(shap_values, X, plot_type='bar', show=False)
        st.pyplot(fig3)
    else:
        st.info("Upload dataset to view analytics.")

elif page == "Single Prediction":
    st.header("👤 Single Customer Churn Prediction")
    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)
        tenure = c1.slider("Tenure (months)", 0, 72, 12)
        MonthlyCharges = c2.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        TotalCharges = c3.number_input("Total Charges", 0.0, 10000.0, 800.0)
        Contract = st.selectbox("Contract Type", ['Month-to-month','One year','Two year'])
        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        inp_df = pd.DataFrame([{ 'tenure': tenure,
                                 'MonthlyCharges': MonthlyCharges,
                                 'TotalCharges': TotalCharges,
                                 'Contract': Contract }])
        X_inp = preprocess_df(inp_df)
        pred = int(model.predict(X_inp)[0])
        prob = float(model.predict_proba(X_inp)[0,1])

        st.subheader("🔮 Prediction Result")
        if pred:
            st.error("🚨 Customer likely to CHURN")
        else:
            st.success("✅ Customer likely to STAY")
        st.info(f"Probability of Churn: {prob:.2%}")
        st.progress(min(1.0, prob))

        expl = shap.Explainer(model)
        sv = expl(X_inp)
        fig4, ax4 = plt.subplots(figsize=(8,4))
        shap.plots.waterfall(sv[0], show=False, max_display=10)
        st.pyplot(fig4)

# --- Batch Prediction Page ---
elif page == "Batch Prediction":
    st.header("Batch Churn Prediction")
    file = st.file_uploader("Upload CSV for predictions", type="csv")
    if file:
        data = pd.read_csv(file)
        data.columns = data.columns.str.strip()
        X = preprocess_df(data)
        preds = model.predict(X)
        probs = model.predict_proba(X)[:,1]
        res = data.copy()
        res['Churn_Prediction'] = np.where(preds==1,'Yes','No')
        res['Churn_Probability'] = probs
        st.dataframe(res)
        csv = res.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
        sep = res['Churn_Prediction'].value_counts()
        fig5 = px.pie(sep, names=sep.index, values=sep.values, hole=0.4,
                      title='Batch Churn Distribution')
        st.plotly_chart(fig5, use_container_width=True)