# app/app.py — Olist ML Inference App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Olist ML Dashboard", layout="wide")
st.title("🛒 Olist E-Commerce — ML Predictions")
st.markdown("Predict **order value** and **late delivery risk** from order features.")

@st.cache_resource
def load_models():
    base = '/mount/src/ml-portfolio'
    reg = joblib.load(os.path.join(base, 'models', 'best_regressor.pkl'))
    clf = joblib.load(os.path.join(base, 'models', 'best_classifier.pkl'))
    return reg, clf

reg_model, clf_model = load_models()
st.success("✅ Models loaded successfully!")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Order Details")
    num_items         = st.slider("Number of items", 1, 20, 2)
    total_price       = st.number_input("Total price (BRL)", 10.0, 2000.0, 150.0)
    total_freight     = st.number_input("Freight value (BRL)", 5.0, 200.0, 20.0)
    max_installments  = st.slider("Max installments", 1, 24, 1)
    review_score      = st.slider("Expected review score", 1, 5, 4)
    main_category     = st.selectbox("Product category", ['bed_bath_table','health_beauty','sports_leisure','furniture_decor','computers_accessories'])
    primary_payment   = st.selectbox("Payment type", ['credit_card','boleto','debit_card','voucher'])
    customer_state    = st.selectbox("Customer state", ['SP','RJ','MG','RS','PR'])

with col2:
    st.subheader("Temporal / Logistics")
    purchase_hour     = st.slider("Purchase hour", 0, 23, 14)
    purchase_month    = st.slider("Purchase month", 1, 12, 6)
    delivery_days_est = st.slider("Estimated delivery days", 1, 60, 15)
    approval_hours    = st.slider("Approval time (hours)", 0.0, 48.0, 1.5)

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([dict(
        num_items=num_items, total_price=total_price, total_freight=total_freight,
        num_sellers=1, total_payment=total_price+total_freight,
        max_installments=max_installments, avg_weight_g=500,
        review_score=review_score, has_review_comment=0,
        delivery_days_estimated=delivery_days_est,
        purchase_hour=purchase_hour, purchase_dayofweek=2,
        purchase_month=purchase_month, purchase_quarter=(purchase_month-1)//3+1,
        is_weekend=0, is_business_hours=int(9<=purchase_hour<=18),
        freight_ratio=total_freight/(total_price+1),
        avg_item_price=total_price/max(num_items,1),
        payment_diff=0, review_is_positive=int(review_score>=4),
        review_is_negative=int(review_score<=2),
        approval_hours=approval_hours, recency=30, frequency=1, monetary=total_price,
        rfm_score=9, recency_score=3, frequency_score=3, monetary_score=3,
        customer_state=customer_state, main_category=main_category,
        primary_payment=primary_payment
    )])
    pred_value = np.expm1(reg_model.predict(input_df)[0])
    pred_late  = clf_model.predict(input_df)[0]
    pred_prob  = clf_model.predict_proba(input_df)[0][1]

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Order Value", f"R$ {pred_value:.2f}")
    m2.metric("Late Delivery Risk",    f"{pred_prob*100:.1f}%")
    m3.metric("Delivery Prediction",  "⚠️ LATE" if pred_late else "✅ ON TIME")