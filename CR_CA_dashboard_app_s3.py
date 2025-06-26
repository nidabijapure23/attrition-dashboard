import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import time
import base64
import boto3
from io import BytesIO
from s3_utils import save_excel_to_s3
import numpy as np

# Set page config
st.set_page_config(
    page_title="Attrition Tracking Dashboard",
    page_icon="📊",
    layout="wide"
)

# AWS S3 configuration (use Streamlit secrets for credentials and bucket info)
AWS_ACCESS_KEY_ID = st.secrets["aws_access_key_id"]
AWS_SECRET_ACCESS_KEY = st.secrets["aws_secret_access_key"]
AWS_REGION = st.secrets["aws_region"]
S3_BUCKET = st.secrets["s3_bucket"]
CR_EXCEL_KEY = st.secrets["cr_excel_key"]  # e.g., 'data/costa_rica_attrition.xlsx'
CA_EXCEL_KEY = st.secrets["ca_excel_key"]  # e.g., 'data/canada_attrition.xlsx'

# Helper function to load Excel from S3
@st.cache_data(ttl=30)
def load_excel_from_s3(bucket, key):
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_excel(BytesIO(obj['Body'].read()))
        if 'Regrettable Y/N' in df.columns:
            df['Regrettable Y/N'] = df['Regrettable Y/N'].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading Excel from S3: {e}")
        return None

# Sidebar
with st.sidebar:
    st.title("📊 Attrition Tracking Dashboard")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard provides insights into employee attrition predictions:
    - Total inactive employee count
    - Risk level distribution
    - Detailed employee list
    - Historical trends
    """)

# --- Tab Layout ---
tabs = st.tabs([
    "Overview",
    "HR Framework",
    "Costa Rica Predictions",
    "Canada Predictions"
])

# --- Tab 1: PPT Slide ---
with tabs[0]:
    st.image("static/Slide5.jpg", use_container_width=True)

# --- Tab 2: HR Framework ---
with tabs[1]:
    try:
        pdf_url = "https://raw.githubusercontent.com/nidabijapure23/attrition-dashboard/main/static/hr_framework.pdf"
        st.markdown(
            f'<iframe src="https://docs.google.com/gview?url={pdf_url}&embedded=true" width="100%" height="800px"></iframe>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Could not display PDF: {e}")

# --- Tab 3: Costa Rica Excel Sheet ---
with tabs[2]:
    st.header("Costa Rica Attrition Predictions")
    df = load_excel_from_s3(S3_BUCKET, CR_EXCEL_KEY)
    if df is not None:
        df['Report Date'] = pd.to_datetime(df['Report Date'])
        df['HR Comments (Only In Case Of Non-Regrettable Exits)'] = df['HR Comments (Only In Case Of Non-Regrettable Exits)'].astype(str)
        df['Ops Comments'] = df['Ops Comments'].astype(str)
        df['HR Comments (Only In Case Of Non-Regrettable Exits)'] = df['HR Comments (Only In Case Of Non-Regrettable Exits)'].replace('nan', '')
        df['Ops Comments'] = df['Ops Comments'].replace('nan', '')
        df['Attrition Probability'] = df['Attrition Probability'].astype(str).str.replace('%', '')
        df['Attrition Probability'] = pd.to_numeric(df['Attrition Probability'], errors='coerce')
        df['Attrition Probability'] = df['Attrition Probability'].apply(lambda x: x/100 if pd.notnull(x) and x > 1 else x)
        df['Attrition Probability'] = df['Attrition Probability'].fillna(0)
        if 'Regrettable Y/N' in df.columns:
            df['Regrettable Y/N'] = df['Regrettable Y/N'].astype(str)
        for col in ['Regrettable Y/N']:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(['nan', 'None', 'NaT'], '')
                df[col] = df[col].replace({np.nan: ''})  # Just in case

        cost_centers = ['All'] + sorted(df['Cost Center'].dropna().unique().tolist())
        tenure_buckets = ['All'] + sorted(df['Tenure Bucket (Today Based)'].dropna().unique().tolist())
        col1, col2, col3, col4= st.columns([ 1, 1, 1, 1])
        with col1:
            min_date = df['Report Date'].min().date()
            max_date = df['Report Date'].max().date()
            start_date = st.date_input("Start Date", min_date, key="cr_start_date")
        with col2:
            end_date = st.date_input("End Date", max_date, key="cr_end_date")
        with col3:
            selected_cost_center = st.selectbox("Cost Center", cost_centers, key="cr_cost_center")
        with col4:
            selected_tenure = st.selectbox("Tenure Bucket", tenure_buckets, key="cr_tenure_bucket")
        mask = (df['Report Date'].dt.date >= start_date) & (df['Report Date'].dt.date <= end_date)
        if selected_cost_center != 'All':
            mask &= (df['Cost Center'] == selected_cost_center)
        if selected_tenure != 'All':
            mask &= (df['Tenure Bucket (Today Based)'] == selected_tenure)
        filtered_df = df[mask]
        df['Attrition Probability'] = pd.to_numeric(df['Attrition Probability'].astype(str).str.replace('%', ''), errors='coerce')
        df['Attrition Probability'] = df['Attrition Probability'].apply(lambda x: x/100 if pd.notnull(x) and x > 1 else x)
        df['Attrition Probability'] = df['Attrition Probability'].fillna(0)
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        metric_colors = {
            'inactive': '#cc0000',
            'severe': '#ff4d4d',
            'more_likely': '#ff944d',
            'intermediate': '#ffc04d',
            'mild': '#ffe04d',
            'regrettable': '#008000'
        }
        with col1:
            total_inactive = (filtered_df['Attrition Prediction'] == 'Inactive').sum()
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {metric_colors['inactive']}; border-radius: 5px;'>
                <p style='color: white; margin: 0;'>Total Inactive Employees</p>
                <p style='color: white; margin: 0;'>{total_inactive}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            severe_count = filtered_df[filtered_df['Risk Level'] == 'Severe'].shape[0]
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {metric_colors['severe']}; border-radius: 5px;'>
                <p style='color: white; margin: 0;'>Severe Risk</p>
                <p style='color: white; margin: 0;'>{severe_count}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            more_likely_count = filtered_df[filtered_df['Risk Level'] == 'More Likely'].shape[0]
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {metric_colors['more_likely']}; border-radius: 5px;'>
                <p style='color: white; margin: 0;'>More Likely</p>
                <p style='color: white; margin: 0;'>{more_likely_count}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            intermediate_count = filtered_df[filtered_df['Risk Level'] == 'Intermediate Risk'].shape[0]
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {metric_colors['intermediate']}; border-radius: 5px;'>
                <p style='color: white; margin: 0;'>Intermediate Risk</p>
                <p style='color: white; margin: 0;'>{intermediate_count}</p>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            mild_count = filtered_df[filtered_df['Risk Level'] == 'Mild Risk'].shape[0]
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {metric_colors['mild']}; border-radius: 5px;'>
                <p style='color: white; margin: 0;'>Mild Risk</p>
                <p style='color: white; margin: 0;'>{mild_count}</p>
            </div>
            """, unsafe_allow_html=True)
        with col6:
            regrettable_count = filtered_df[filtered_df['Regrettable Y/N'] == 'Y'].shape[0]
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {metric_colors['regrettable']}; border-radius: 5px;'>
                <p style='color: white; margin: 0;'>Regrettable</p>
                <p style='color: white; margin: 0;'>{regrettable_count}</p>
            </div>
            """, unsafe_allow_html=True)
        st.subheader("⚠️ Risk Level Distribution")
        col1, col2 = st.columns(2)
        with col1:
            risk_order = ['Severe', 'More Likely', 'Intermediate Risk', 'Mild Risk']
            risk_counts = filtered_df['Risk Level'].value_counts().reindex(risk_order)
            fig_risk = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                orientation='v',
                title="Distribution of Risk Levels",
                color=risk_counts.index,
                color_discrete_map={
                    'Severe': '#ff4d4d',
                    'More Likely': '#ff944d',
                    'Intermediate Risk': '#ffc04d',
                    'Mild Risk': '#ffe04d'
                }
            )
            fig_risk.update_layout(
                xaxis_title="Risk Level",
                yaxis_title="Number of Employees",
                showlegend=False,
                bargap=0.6,
                height=400,
                width=200
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        with col2:
            try:
                if 'Cost Center' in filtered_df.columns:
                    inactive_df = filtered_df[filtered_df['Attrition Prediction'].str.lower() == 'inactive']
                    cost_center_counts = inactive_df['Cost Center'].value_counts()
                    if not cost_center_counts.empty:
                        top_5_centers = cost_center_counts.head(5)
                        fig_cost = px.pie(
                            values=top_5_centers.values,
                            names=top_5_centers.index,
                            title="Distribution by Cost Center (Top 5)",
                            hole=0.5
                        )
                        fig_cost.update_layout(
                            height=400,
                            showlegend=True,
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=-0.2,
                                xanchor="center",
                                x=0.5,
                                font=dict(size=10)
                            ),
                            margin=dict(l=10, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_cost, use_container_width=True)
                    else:
                        st.info("No inactive employees found in the selected date range")
                else:
                    st.warning("Cost Center information not available in the data")
            except Exception as e:
                st.error(f"Error creating cost center chart: {str(e)}")
                st.write("Debug - DataFrame columns:", filtered_df.columns.tolist())
                st.write("Debug - Sample data:", filtered_df[['Attrition Prediction', 'Cost Center']].head())
        st.subheader("👥 Employee List")
        table_df = filtered_df.copy()
        if len(table_df) == 0:
            st.warning("No data matches the current filters. Please adjust the filters.")
            st.stop()
        table_df = table_df.sort_values('Attrition Probability', ascending=False)
        table_df['Delete'] = False
        if 'SR.No.' in table_df.columns:
            table_df = table_df.drop('SR.No.', axis=1)
        table_df.insert(0, 'SR.No.', range(1, len(table_df) + 1))
        display_cols = ['SR.No.', 'Report Date','Employee ID','Employee Name','Cost Center','Attrition Prediction','Attrition Probability', 
                    'Risk Level', 'Tenure Bucket (Today Based)','Triggers', 'HR Comments (Only In Case Of Non-Regrettable Exits)', 'Ops Comments','Regrettable Y/N']
        table_df['Report Date'] = pd.to_datetime(table_df['Report Date']).dt.strftime('%Y-%m-%d')
        table_df = table_df[display_cols + ['Delete']]
        edited_df = st.data_editor(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "SR.No.": st.column_config.NumberColumn(
                    "SR.No.",
                    help="Serial Number",
                    width="small"
                ),
                "Attrition Probability": st.column_config.NumberColumn(
                    "Attrition Probability",
                    help="Probability of attrition",
                    width="medium",
                    format="%.2f",
                    step=0.01,
                    min_value=0.0,
                    max_value=1.0
                ),
                "Cost Center": st.column_config.TextColumn(
                    "Cost Center",
                    help="Employee's Cost Center",
                    width="medium"
                ),
                "HR Comments (Only In Case Of Non-Regrettable Exits)": st.column_config.TextColumn(
                    "HR Comments",
                    help="Add HR comments here",
                    width="large"
                ),
                "Ops Comments": st.column_config.TextColumn(
                    "OPS Comments",
                    help="Add OPS comments here",
                    width="large"
                ),
                "Regrettable Y/N": st.column_config.TextColumn(
                    "Regrettable",
                    help="Is this attrition regrettable?",
                    width="small"
                ),
                "Delete": st.column_config.CheckboxColumn(
                    "Delete",
                    help="Check to mark this row for deletion"
                )
            },
            key="cr_data_editor"
        )
        col_save, _, col_delete = st.columns([1,7.7,1])
        with col_save:
            if st.button("Save Comments"):
                try:
                    # Reload the latest data from S3
                    current_df = load_excel_from_s3(S3_BUCKET, CR_EXCEL_KEY)
                    current_df['Report Date'] = pd.to_datetime(current_df['Report Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    edited_df['Report Date'] = pd.to_datetime(edited_df['Report Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    for idx, row in edited_df.iterrows():
                        emp_id = str(row['Employee ID'])
                        date_val = str(row['Report Date'])
                        current_df['Employee ID'] = current_df['Employee ID'].astype(str)
                        mask = (current_df['Employee ID'] == emp_id) & (current_df['Report Date'] == date_val)
                        if mask.sum() > 0:
                            current_df.loc[mask, 'HR Comments (Only In Case Of Non-Regrettable Exits)'] = row['HR Comments (Only In Case Of Non-Regrettable Exits)']
                            current_df.loc[mask, 'Ops Comments'] = row['Ops Comments']
                            current_df.loc[mask, 'Regrettable Y/N'] = row['Regrettable Y/N']
                    save_cols = [col for col in display_cols if col not in ['SR.No.', 'Delete']]
                    current_df = current_df[save_cols]
                    save_excel_to_s3(
                        current_df, S3_BUCKET, CR_EXCEL_KEY,
                        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
                    )
                    st.session_state.action_message = "Comments saved successfully!"
                    st.session_state.action_message_type = "success"
                    time.sleep(2)
                except Exception as e:
                    st.session_state.action_message = f"Error saving comments: {str(e)}"
                    st.session_state.action_message_type = "error"
        with col_delete:
            if st.button("Delete Selected"):
                try:
                    current_df = load_excel_from_s3(S3_BUCKET, CR_EXCEL_KEY)
                    current_df['Report Date'] = pd.to_datetime(current_df['Report Date'])
                    edited_df['Report Date'] = pd.to_datetime(edited_df['Report Date'])
                    to_delete = edited_df[edited_df['Delete']]
                    if not to_delete.empty:
                        mask = pd.Series(True, index=current_df.index)
                        for _, row in to_delete.iterrows():
                            delete_mask = (
                                (current_df['Employee ID'] == row['Employee ID']) &
                                (current_df['Report Date'] == row['Report Date'])
                            )
                            mask = mask & ~delete_mask
                        current_df = current_df[mask]
                        current_df['Report Date'] = current_df['Report Date'].dt.strftime('%Y-%m-%d')
                        save_excel_to_s3(
                            current_df, S3_BUCKET, CR_EXCEL_KEY,
                            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
                        )
                        st.session_state.action_message = "Selected rows deleted successfully!"
                        st.session_state.action_message_type = "success"
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.warning("No rows selected for deletion")
                except Exception as e:
                    st.session_state.action_message = f"Error deleting rows: {str(e)}"
                    st.session_state.action_message_type = "error"
                    st.rerun()
        # Download button for filtered data
        csv = edited_df[display_cols].to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name=f"attrition_tracking_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        st.info("Note: Editing, saving, or deleting rows is disabled for S3 Excel data. Download the filtered data for further processing.")
    else:
        st.warning("No data available. Please ensure the Excel file exists in S3 and is accessible.")

# --- Tab 4: Canada Predictions ---
with tabs[3]:
    st.header("Canada Attrition Predictions")
    df1 = load_excel_from_s3(S3_BUCKET, CA_EXCEL_KEY)
    if df1 is not None:
        df1.columns = [col.strip() for col in df1.columns]
        st.write("Columns loaded:", df1.columns.tolist())
        if 'Regrettable Y/N' in df1.columns:
            df1['Regrettable Y/N'] = df1['Regrettable Y/N'].astype(str)
        for col in ['Employment Status', 'Regrettable Y/N']:
            if col in df1.columns:
                df1[col] = df1[col].astype(str).replace(['nan', 'None', 'NaT'], '').replace({pd.NA: '', None: '', float('nan'): ''}).str.strip()
        st.write("Sample data:", df1[['Employment Status', 'Regrettable Y/N']].head())
        df1['Report Date'] = pd.to_datetime(df1['Report Date'])
        df1['HR Comments (Only In Case Of Non-Regrettable Exits)'] = df1['HR Comments (Only In Case Of Non-Regrettable Exits)'].astype(str)
        df1['Ops Comments'] = df1['Ops Comments'].astype(str)
        df1['HR Comments (Only In Case Of Non-Regrettable Exits)'] = df1['HR Comments (Only In Case Of Non-Regrettable Exits)'].replace('nan', '')
        df1['Ops Comments'] = df1['Ops Comments'].replace('nan', '')
        df1['Attrition Probability'] = df1['Attrition Probability'].astype(str).str.replace('%', '')
        df1['Attrition Probability'] = pd.to_numeric(df1['Attrition Probability'], errors='coerce')
        df1['Attrition Probability'] = df1['Attrition Probability'].apply(lambda x: x/100 if pd.notnull(x) and x > 1 else x)
        df1['Attrition Probability'] = df1['Attrition Probability'].fillna(0)
    
        cost_centers = ['All'] + sorted(df1['Cost Center'].dropna().unique().tolist())
        tenure_buckets = ['All'] + sorted(df1['Tenure Bucket (Today Based)'].dropna().unique().tolist())
        col1, col2, col3, col4= st.columns([ 1, 1, 1, 1])
        with col1:
            min_date = df1['Report Date'].min().date()
            max_date = df1['Report Date'].max().date()
            start_date = st.date_input("Start Date", min_date, key="canada_start_date")
        with col2:
            end_date = st.date_input("End Date", max_date, key="canada_end_date")
        with col3:
            selected_cost_center = st.selectbox("Cost Center", cost_centers, key="canada_cost_center")
        with col4:
            selected_tenure = st.selectbox("Tenure Bucket", tenure_buckets, key="canada_tenure_bucket")
        mask = (df1['Report Date'].dt.date >= start_date) & (df1['Report Date'].dt.date <= end_date)
        if selected_cost_center != 'All':
            mask &= (df1['Cost Center'] == selected_cost_center)
        if selected_tenure != 'All':
            mask &= (df1['Tenure Bucket (Today Based)'] == selected_tenure)
        filtered_df1 = df1[mask]
        df1['Attrition Probability'] = pd.to_numeric(df1['Attrition Probability'].astype(str).str.replace('%', ''), errors='coerce')
        df1['Attrition Probability'] = df1['Attrition Probability'].apply(lambda x: x/100 if pd.notnull(x) and x > 1 else x)
        df1['Attrition Probability'] = df1['Attrition Probability'].fillna(0)
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        metric_colors = {
            'inactive': '#cc0000',
            'severe': '#ff4d4d',
            'more_likely': '#ff944d',
            'intermediate': '#ffc04d',
            'mild': '#ffe04d',
            'regrettable': '#008000'
        }
        with col1:
            total_inactive = (filtered_df1['Attrition Prediction'] == 'Inactive').sum()
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {metric_colors['inactive']}; border-radius: 5px;'>
                <p style='color: white; margin: 0;'>Total Inactive Employees</p>
                <p style='color: white; margin: 0;'>{total_inactive}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            severe_count = filtered_df1[filtered_df1['Risk Level'] == 'Severe'].shape[0]
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {metric_colors['severe']}; border-radius: 5px;'>
                <p style='color: white; margin: 0;'>Severe Risk</p>
                <p style='color: white; margin: 0;'>{severe_count}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            more_likely_count = filtered_df1[filtered_df1['Risk Level'] == 'More Likely'].shape[0]
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {metric_colors['more_likely']}; border-radius: 5px;'>
                <p style='color: white; margin: 0;'>More Likely</p>
                <p style='color: white; margin: 0;'>{more_likely_count}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            intermediate_count = filtered_df1[filtered_df1['Risk Level'] == 'Intermediate Risk'].shape[0]
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {metric_colors['intermediate']}; border-radius: 5px;'>
                <p style='color: white; margin: 0;'>Intermediate Risk</p>
                <p style='color: white; margin: 0;'>{intermediate_count}</p>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            mild_count = filtered_df1[filtered_df1['Risk Level'] == 'Mild Risk'].shape[0]
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {metric_colors['mild']}; border-radius: 5px;'>
                <p style='color: white; margin: 0;'>Mild Risk</p>
                <p style='color: white; margin: 0;'>{mild_count}</p>
            </div>
            """, unsafe_allow_html=True)
        with col6:
            regrettable_count = filtered_df1[filtered_df1['Regrettable Y/N'] == 'Y'].shape[0]
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {metric_colors['regrettable']}; border-radius: 5px;'>
                <p style='color: white; margin: 0;'>Regrettable</p>
                <p style='color: white; margin: 0;'>{regrettable_count}</p>
            </div>
            """, unsafe_allow_html=True)
        st.subheader("⚠️ Risk Level Distribution")
        col1, col2 = st.columns(2)
        with col1:
            risk_order = ['Severe', 'More Likely', 'Intermediate Risk', 'Mild Risk']
            risk_counts = filtered_df1['Risk Level'].value_counts().reindex(risk_order)
            fig_risk = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                orientation='v',
                title="Distribution of Risk Levels",
                color=risk_counts.index,
                color_discrete_map={
                    'Severe': '#ff4d4d',
                    'More Likely': '#ff944d',
                    'Intermediate Risk': '#ffc04d',
                    'Mild Risk': '#ffe04d'
                }
            )
            fig_risk.update_layout(
                xaxis_title="Risk Level",
                yaxis_title="Number of Employees",
                showlegend=False,
                bargap=0.6,
                height=400,
                width=200
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        with col2:
            try:
                if 'Cost Center' in filtered_df1.columns:
                    inactive_df1 = filtered_df1[filtered_df1['Attrition Prediction'].str.lower() == 'inactive']
                    cost_center_counts = inactive_df1['Cost Center'].value_counts()
                    if not cost_center_counts.empty:
                        top_5_centers = cost_center_counts.head(5)
                        fig_cost = px.pie(
                            values=top_5_centers.values,
                            names=top_5_centers.index,
                            title="Distribution by Cost Center (Top 5)",
                            hole=0.5
                        )
                        fig_cost.update_layout(
                            height=400,
                            showlegend=True,
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=-0.2,
                                xanchor="center",
                                x=0.5,
                                font=dict(size=10)
                            ),
                            margin=dict(l=10, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_cost, use_container_width=True)
                    else:
                        st.info("No inactive employees found in the selected date range")
                else:
                    st.warning("Cost Center information not available in the data")
            except Exception as e:
                st.error(f"Error creating cost center chart: {str(e)}")
                st.write("Debug - DataFrame columns:", filtered_df1.columns.tolist())
                st.write("Debug - Sample data:", filtered_df1[['Attrition Prediction', 'Cost Center']].head())
        st.subheader("👥 Employee List")
        table_df1 = filtered_df1.copy()
        if len(table_df1) == 0:
            st.warning("No data matches the current filters. Please adjust the filters.")
            st.stop()
        table_df1 = table_df1.sort_values('Attrition Probability', ascending=False)
        table_df1['Delete'] = False
        if 'SR.No.' in table_df1.columns:
            table_df1 = table_df1.drop('SR.No.', axis=1)
        table_df1.insert(0, 'SR.No.', range(1, len(table_df1) + 1))
        display_cols = ['SR.No.', 'Report Date','Employee ID','Employee Name','Cost Center','Attrition Prediction','Attrition Probability', 
                    'Risk Level', 'Tenure Bucket (Today Based)','TL Name','Employment Status', 'Triggers','Ops Comments','Regrettable Y/N','HR Comments (Only In Case Of Non-Regrettable Exits)']
        table_df1['Report Date'] = pd.to_datetime(table_df1['Report Date']).dt.strftime('%Y-%m-%d')
        table_df1 = table_df1[display_cols + ['Delete']]
        edited_df1 = st.data_editor(
            table_df1,
            use_container_width=True,
            hide_index=True,
            column_config={
                "SR.No.": st.column_config.NumberColumn(
                    "SR.No.",
                    help="Serial Number",
                    width="small"
                ),
                "Attrition Probability": st.column_config.NumberColumn(
                    "Attrition Probability",
                    help="Probability of attrition",
                    width="medium",
                    format="%.2f",
                    step=0.01,
                    min_value=0.0,
                    max_value=1.0
                ),
                "Cost Center": st.column_config.TextColumn(
                    "Cost Center",
                    help="Employee's Cost Center",
                    width="medium"
                ),
                "HR Comments (Only In Case Of Non-Regrettable Exits)": st.column_config.TextColumn(
                    "HR Comments",
                    help="Add HR comments here",
                    width="large"
                ),
                "Ops Comments": st.column_config.TextColumn(
                    "OPS Comments",
                    help="Add OPS comments here",
                    width="large"
                ),
                "Regrettable Y/N": st.column_config.TextColumn(
                    "Regrettable",
                    help="Is this attrition regrettable?",
                    width="small"
                ),
                "Delete": st.column_config.CheckboxColumn(
                    "Delete",
                    help="Check to mark this row for deletion"
                )
            },
            key="ca_data_editor"
        )
        col_save, _, col_delete = st.columns([1,7.7,1])
        with col_save:
            if st.button("Save Comments", key='ca_save_button'):
                try:
                    current_df1 = load_excel_from_s3(S3_BUCKET, CA_EXCEL_KEY)
                    current_df1['Report Date'] = pd.to_datetime(current_df1['Report Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    edited_df1['Report Date'] = pd.to_datetime(edited_df1['Report Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    for idx, row in edited_df1.iterrows():
                        emp_id = str(row['Employee ID'])
                        date_val = str(row['Report Date'])
                        current_df1['Employee ID'] = current_df1['Employee ID'].astype(str)
                        mask = (current_df1['Employee ID'] == emp_id) & (current_df1['Report Date'] == date_val)
                        if mask.sum() > 0:
                            current_df1.loc[mask, 'HR Comments (Only In Case Of Non-Regrettable Exits)'] = row['HR Comments (Only In Case Of Non-Regrettable Exits)']
                            current_df1.loc[mask, 'Ops Comments'] = row['Ops Comments']
                            current_df1.loc[mask, 'Regrettable Y/N'] = row['Regrettable Y/N']
                    save_cols = [col for col in display_cols if col not in ['SR.No.', 'Delete']]
                    current_df1 = current_df1[save_cols]
                    save_excel_to_s3(
                        current_df1, S3_BUCKET, CA_EXCEL_KEY,
                        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
                    )
                    st.session_state.action_message = "Comments saved successfully!"
                    st.session_state.action_message_type = "success"
                    time.sleep(2)
                except Exception as e:
                    st.session_state.action_message = f"Error saving comments: {str(e)}"
                    st.session_state.action_message_type = "error"
        with col_delete:
            if st.button("Delete Selected", key="ca_dlt_btn"):
                try:
                    current_df1 = load_excel_from_s3(S3_BUCKET, CA_EXCEL_KEY)
                    current_df1['Report Date'] = pd.to_datetime(current_df1['Report Date'])
                    edited_df1['Report Date'] = pd.to_datetime(edited_df1['Report Date'])
                    to_delete = edited_df1[edited_df1['Delete']]
                    if not to_delete.empty:
                        mask = pd.Series(True, index=current_df1.index)
                        for _, row in to_delete.iterrows():
                            delete_mask = (
                                (current_df1['Employee ID'] == row['Employee ID']) &
                                (current_df1['Report Date'] == row['Report Date'])
                            )
                            mask = mask & ~delete_mask
                        current_df1 = current_df1[mask]
                        current_df1['Report Date'] = current_df1['Report Date'].dt.strftime('%Y-%m-%d')
                        save_excel_to_s3(
                            current_df1, S3_BUCKET, CA_EXCEL_KEY,
                            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
                        )
                        st.session_state.action_message = "Selected rows deleted successfully!"
                        st.session_state.action_message_type = "success"
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.warning("No rows selected for deletion")
                except Exception as e:
                    st.session_state.action_message = f"Error deleting rows: {str(e)}"
                    st.session_state.action_message_type = "error"
                    st.rerun()
        csv = edited_df1[display_cols].to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name=f"attrition_tracking_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key = "ca_download_button"
        )
        st.info("Note: Editing, saving, or deleting rows is disabled for S3 Excel data. Download the filtered data for further processing.")
    else:
        st.warning("No data available. Please ensure the Excel file exists in S3 and is accessible.") 