import streamlit as st
import pandas as pd
import joblib
import os
import sys
import plotly.express as px

# Add python directory to path so we can import the models
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

from models.rule_based_model import RuleBasedModel
from models.isolation_forest_model import IsolationForestModel
from models.xgboost_model import XGBoostModel
from preprocessing import engineer_timestamp

st.set_page_config(page_title="IDS Pipeline Prototype", layout="wide", page_icon="🛡️")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .metric-card {
        background-color: #1e2129;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border-left: 5px solid #4CAF50;
    }
    .metric-card.attack { border-left-color: #f44336; }
    .metric-card.anomaly { border-left-color: #ff9800; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    artifact_dir = 'artifacts'
    try:
        encoder = joblib.load(os.path.join(artifact_dir, 'label_encoder.joblib'))
        scaler = joblib.load(os.path.join(artifact_dir, 'scaler_xgb.joblib'))
        
        iso_model = IsolationForestModel()
        iso_model.load_model(os.path.join(artifact_dir, 'isolation_forest_model.joblib'))
        
        xgb_model = XGBoostModel()
        xgb_model.load_model(os.path.join(artifact_dir, 'xgboost_model.joblib'))
        
        return encoder, scaler, iso_model, xgb_model
    except Exception as e:
        st.error(f"Error loading models. Did you run the training pipeline first? {e}")
        return None, None, None, None

def align_features(df, expected_features):
    """Ensure dataframe has the exact features expected by the model."""
    aligned_df = pd.DataFrame(index=df.index)
    for col in expected_features:
        if col in df.columns:
            aligned_df[col] = df[col]
        else:
            aligned_df[col] = 0  # Fill missing dummy columns with 0
    return aligned_df

def run_pipeline(df, encoder, scaler, iso_model, xgb_model):
    results = df.copy()
    results['Pipeline_Stage'] = 'Unprocessed'
    results['Final_Prediction'] = 'Unknown'
    
    progress = st.progress(0)
    status = st.empty()
    
    # Ensure timestamp is processed if present
    if 'timestamp' in df.columns and 'hour_of_day' not in df.columns:
        df = engineer_timestamp(df)
    
    # ---------------------------------------------------------
    # STAGE 1: RULE-BASED ENGINE
    # ---------------------------------------------------------
    status.markdown("🛡️ **Stage 1: Rule-Based Engine (Known Signatures)**")
    rule_model = RuleBasedModel(class_labels=encoder.classes_.tolist())
    
    # Run predictions
    rb_preds = rule_model.predict(df)
    # Get string labels from indices
    idx_to_label = {i: label for i, label in enumerate(encoder.classes_)}
    rb_labels = [idx_to_label[p] for p in rb_preds]
    
    results['Rule_Prediction'] = rb_labels
    
    # Filter traffic that bypassed the rules (i.e. classified as BENIGN by rules)
    bypassed_idx = [i for i, label in enumerate(rb_labels) if label == 'BENIGN']
    caught_idx = [i for i, label in enumerate(rb_labels) if label != 'BENIGN']
    
    for i in caught_idx:
        results.loc[i, 'Final_Prediction'] = rb_labels[i]
        results.loc[i, 'Pipeline_Stage'] = 'Rule-Based'
    
    progress.progress(33)
    
    if not bypassed_idx:
        status.success("All traffic analyzed!")
        progress.progress(100)
        return results
        
    df_bypassed = df.iloc[bypassed_idx].copy()
    
    # ---------------------------------------------------------
    # STAGE 2: ISOLATION FOREST
    # ---------------------------------------------------------
    status.markdown("🔍 **Stage 2: Isolation Forest (Anomaly Detection)**")
    
    # Prepare features for IF (scaled, no dummy columns)
    iso_expected_cols = iso_model.model.feature_names_in_
    df_iso = align_features(df_bypassed, iso_expected_cols)
    
    # For prototype, we might need to apply scaling. But IF training scaled it. 
    # We will use the xgb scaler, but only on the columns IF expects.
    # Actually, pipeline.py fit scaler on ALL continuous columns. 
    cols_to_scale_iso = [c for c in iso_expected_cols if c in scaler.feature_names_in_]
    # We create a temporary DataFrame for scaling, mapping columns properly
    df_iso_to_scale = align_features(df_bypassed, scaler.feature_names_in_)
    df_iso_scaled_full = pd.DataFrame(scaler.transform(df_iso_to_scale), columns=scaler.feature_names_in_, index=df_bypassed.index)
    df_iso_scaled = df_iso_scaled_full[iso_expected_cols]
    
    # Predict Anomalies
    iso_preds = iso_model.predict_labels(df_iso_scaled, benign_label="BENIGN", anomaly_label="ATTACK")
    
    anomaly_idx_in_bypassed = [i for i, label in enumerate(iso_preds) if label == 'ATTACK']
    benign_idx_in_bypassed = [i for i, label in enumerate(iso_preds) if label == 'BENIGN']
    
    # Assign BENIGN definitively
    for i in benign_idx_in_bypassed:
        orig_idx = bypassed_idx[i]
        results.loc[orig_idx, 'Final_Prediction'] = 'BENIGN'
        results.loc[orig_idx, 'Pipeline_Stage'] = 'Isolation Forest'
        
    progress.progress(66)
    
    if not anomaly_idx_in_bypassed:
        status.success("All traffic analyzed!")
        progress.progress(100)
        return results
        
    anomaly_orig_idx = [bypassed_idx[i] for i in anomaly_idx_in_bypassed]
    df_anomalies = df.iloc[anomaly_orig_idx].copy()
    
    # ---------------------------------------------------------
    # STAGE 3: XGBOOST IDENTIFICATION
    # ---------------------------------------------------------
    status.markdown("🎯 **Stage 3: XGBoost (Attack Identification)**")
    
    xgb_expected_cols = xgb_model.model.feature_names_in_
    
    # One-hot encode categoricals if they exist in the raw df
    cat_cols = ["protocol", "port_group", "source_day"]
    df_anomalies_enc = pd.get_dummies(df_anomalies, columns=[c for c in cat_cols if c in df_anomalies.columns])
    
    df_xgb = align_features(df_anomalies_enc, xgb_expected_cols)
    
    # Scale
    df_xgb_scaled_full = pd.DataFrame(scaler.transform(align_features(df_anomalies_enc, scaler.feature_names_in_)), 
                                      columns=scaler.feature_names_in_, index=df_anomalies.index)
    
    # Replace the continuous columns in df_xgb with scaled ones
    for col in scaler.feature_names_in_:
        if col in df_xgb.columns:
            df_xgb[col] = df_xgb_scaled_full[col]
            
    # Fill NaN silently (RobustScaler preserves NaNs, XGBoost handles them but let's be safe)
    df_xgb = df_xgb.fillna(0)
            
    xgb_pred_probs = xgb_model.model.predict(df_xgb)
    # Get max probability index
    xgb_preds = xgb_pred_probs.argmax(axis=1)
    
    xgb_labels = [idx_to_label[p] for p in xgb_preds]
    
    for i, label in enumerate(xgb_labels):
        orig_idx = anomaly_orig_idx[i]
        results.loc[orig_idx, 'Final_Prediction'] = label
        results.loc[orig_idx, 'Pipeline_Stage'] = 'XGBoost'
        
    progress.progress(100)
    status.success("Analysis Complete!")
    return results


def main():
    st.title("🛡️ Advanced Network Intrusion Detection System")
    st.markdown("### Functional Prototype: Multi-Stage Pipeline")
    
    st.markdown("""
    This prototype simulates the pipeline process for analyzing network traffic PCAPs (provided as CSV flow features).
    - **Stage 1:** Rule-Based Engine catches obvious/known signatures.
    - **Stage 2:** Isolation Forest screens remaining traffic for novel anomalies.
    - **Stage 3:** XGBoost classifies the specific attack type of flagged anomalies.
    """)
    
    encoder, scaler, iso_model, xgb_model = load_artifacts()
    
    if encoder is None:
        return
        
    uploaded_file = st.file_uploader("Upload Network Traffic Data (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df):,} flows.")
            
            if st.button("Run Full Pipeline Analysis", type="primary"):
                results = run_pipeline(df, encoder, scaler, iso_model, xgb_model)
                
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Metrics Summary
                col1, col2, col3, col4 = st.columns(4)
                
                total_flows = len(results)
                benign = len(results[results['Final_Prediction'] == 'BENIGN'])
                attacks = total_flows - benign
                
                caught_rule = len(results[results['Pipeline_Stage'] == 'Rule-Based'])
                caught_xgb = len(results[results['Pipeline_Stage'] == 'XGBoost'])
                
                with col1:
                    st.markdown(f"<div class='metric-card'><h3>{total_flows:,}</h3><p>Total Flows</p></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-card attack'><h3>{attacks:,}</h3><p>Attacks Detected</p></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='metric-card anomaly'><h3>{caught_rule:,}</h3><p>Blocked by Rules</p></div>", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"<div class='metric-card attack'><h3>{caught_xgb:,}</h3><p>Identified by AI</p></div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.subheader("Traffic Composition")
                    fig = px.pie(results, names='Final_Prediction', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
                    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                with chart_col2:
                    st.subheader("Detection Pipeline Stages")
                    stage_counts = results['Pipeline_Stage'].value_counts().reset_index()
                    stage_counts.columns = ['Stage', 'Count']
                    fig2 = px.bar(stage_counts, x='Stage', y='Count', color='Stage', color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.subheader("Detailed Flow Logs")
                # Show only important columns to user
                display_cols = ['Pipeline_Stage', 'Final_Prediction'] + [c for c in df.columns if c not in ['Pipeline_Stage', 'Final_Prediction']]
                st.dataframe(results[display_cols].head(1000), use_container_width=True)
                
                # Download button
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Full Report", data=csv, file_name="ids_analysis_report.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
