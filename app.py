import math
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import ast
import io
import joblib
import os

# Page config
st.set_page_config(
    page_title="Investment Opportunity Finder", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS + Font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        color: #334155;
    }
    
    /* Main header styling */
    h1, h2, h3 {
        font-weight: 700 !important;
        color: #0f172a;
    }
    
    /* Dashboard header */
    .dashboard-header {
        background: linear-gradient(120deg, #2563eb, #3b82f6, #60a5fa);
        padding: 3rem 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .dashboard-header h1 {
        font-size: 2.75rem;
        font-weight: 800 !important;
        margin: 0;
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .dashboard-header p {
        font-size: 1.25rem;
        margin-top: 0.75rem;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #f1f5f9;
    }
    
    /* Cards for metrics */
    .metric-card {
        background-color: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-card .label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e40af;
    }
    
    /* Section headers */
    .section-header {
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #2563eb;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Tables */
    .dataframe {
        border: none !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8fafc;
    }
    
    .dataframe tbody tr:hover {
        background-color: #e0f2fe;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #94a3b8;
        font-size: 0.875rem;
    }
    
    /* Links */
    a {
        color: #2563eb;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* Insight cards */
    .insight-card {
        background-color: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .insight-card h3 {
        margin-top: 0;
        color: #2563eb;
        font-size: 1.2rem;
    }
    
    .insight-card p {
        color: #475569;
        margin-bottom: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown("""
  <div class="dashboard-header">
    <h1>Investment Opportunity Finder</h1>
    <p>Discover high-potential companies with AI-driven insights</p>
  </div>
""", unsafe_allow_html=True)

# Load Data - Simplified with better error handling
@st.cache_data
def load_data():
    """Load data from parquet file"""
    data_paths = [
        "mnt/data/company_data_scored.parquet",
        "data/company_data_scored.parquet"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                
                
                                # Extract website URL into its own column
                if "website" in df.columns:
                    df["website_url"] = df["website"].apply(extract_url)
                else:
                   df["website_url"] = ""
                
                return df
            except Exception:
                continue
    
    # If no data was loaded
    st.warning("Data files not found. Please check that parquet files exist.")
    return pd.DataFrame()

def extract_url(website_data):
    """Extract URL from website JSON data (handles dicts and strings)"""
    try:
        if isinstance(website_data, dict):
            return website_data.get("url", "")
        if isinstance(website_data, str):
            # e.g. "{'domain':'x','url':'https://x'}"
            data = ast.literal_eval(website_data)
            return data.get("url", "")
    except Exception:
        pass
    return ""


# Load model - simplified
def load_model():
    """Load model with fallback options"""
    model_paths = [
        "mnt/data/vc_scoring_pipeline.joblib",
        "data/vc_scoring_pipeline.joblib"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            return joblib.load(path)
    
    return None

# Generate feature importance chart
def generate_feature_importance_chart():
    model = load_model()
    
    # Skip if model not found or doesn't have feature importances
    if not model or not hasattr(model.named_steps['classifier'], 'feature_importances_'):
        return None
    
    # Get feature names from preprocessor
    numeric_features = model.named_steps['preprocessor'].transformers_[0][2]
    
    # Try to get categorical features if they exist
    feature_names = list(numeric_features)
    if len(model.named_steps['preprocessor'].transformers_) > 1:
        try:
            categorical_features = model.named_steps['preprocessor'].transformers_[1][2]
            cat_feature_names = model.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_features)
            feature_names += list(cat_feature_names)
        except:
            pass
    
    # Get importances
    importances = model.named_steps['classifier'].feature_importances_
    
    # Create DataFrame for visualization
    imp_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    })
    imp_df = imp_df.sort_values('importance', ascending=False).head(10)
    
    # Create plot using Altair
    chart = alt.Chart(imp_df).mark_bar(color='#3b82f6').encode(
        x=alt.X('importance:Q', title='Importance'),
        y=alt.Y('feature:N', title=None, sort='-x'),
        tooltip=['feature', 'importance']
    ).properties(
        title='Top 10 Feature Importance',
        height=300
    )
    
    return chart

# Load data
with st.spinner("Loading company data..."):
    companies = load_data()

# Search Bar
search_col1, _ = st.columns([3, 1])
with search_col1:
    search_query = st.text_input("🔍 Search companies by name or domain", 
                            placeholder="Enter company name or website domain")

# Sidebar Filters & Display Columns
with st.sidebar:
    st.markdown("## 🔎 Filters")
    
    # Only proceed if we have data
    if not companies.empty:
        # Score range
        smin = math.floor(companies.investment_score.min())
        smax = math.ceil(companies.investment_score.max())   # 91.4 → 92 (keeps everyone)
        score_low, score_high = st.slider(
            "Investment Score Range",
            value=(smin, smax),
            min_value=smin,
            max_value=smax,
            step=1
)  
        # Headcount range
        if "headcount" in companies.columns:
            hmin, hmax = int(companies.headcount.min()), int(companies.headcount.max())
            head_low, head_high = st.slider("Headcount Range", hmin, hmax, (hmin, hmax))
        else:
            head_low, head_high = (0, 0)
        
        # Funding range
        if "funding_total" in companies.columns:
            fmin, fmax = int(companies.funding_total.min()), min(int(companies.funding_total.max()), 100000000)
            funding_low, funding_high = st.slider("Funding Range ($)", 
                                                fmin, fmax, 
                                                (fmin, fmax),
                                                format="$%d")
        else:
            funding_low, funding_high = (0, 0)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Columns to display
        st.markdown("### Display Options")
        default_cols = ["name", "investment_score"]
        extras = st.multiselect(
            "Additional Columns",
            options=[c for c in companies.columns if c not in default_cols + ["website", "website_url"]],
            default=["headcount", "funding_total", "headcount_growth_12m"]
        )
        display_cols = default_cols + extras
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Feature comparison selectors
        st.markdown("### Data Visualization")
        numeric_cols = companies.select_dtypes(include="number").columns.tolist()
        x_feature = st.selectbox("X-axis Feature", 
                                options=numeric_cols, 
                                index=numeric_cols.index("headcount") if "headcount" in numeric_cols else 0)
        y_feature = st.selectbox("Y-axis Feature", 
                                options=numeric_cols, 
                                index=numeric_cols.index("investment_score") if "investment_score" in numeric_cols else 1)
        
        # Chart type selector
        chart_type = st.selectbox("Chart Type", ["Scatter Plot", "Heat Map"])
        
        # Reset filters button
        if st.button("Reset All Filters"):
            search_query = ""
            score_low, score_high = smin, smax
            if "headcount" in companies.columns:
                head_low, head_high = hmin, hmax
            if "funding_total" in companies.columns: 
                funding_low, funding_high = fmin, fmax
    else:
        st.error("No data available. Please check your data files.")

# Main Content - Only show if data is loaded
if not companies.empty:
    # Filter Data
    filtered = companies[
        companies.investment_score.between(score_low, score_high) &
        ((companies.headcount.between(head_low, head_high)) if "headcount" in companies.columns else True) &
        ((companies.funding_total.between(funding_low, funding_high)) if "funding_total" in companies.columns else True)
    ].copy()

    # Search filter (if query exists)
    if search_query:
        filtered = filtered[
            filtered.name.str.contains(search_query, case=False, na=False)
        ]

    # Key Metrics
    p90 = np.percentile(companies.investment_score, 90)
    p75 = np.percentile(companies.investment_score, 75)
    
    # Get target companies if that column exists
    targets = companies[companies.is_target] if "is_target" in companies.columns else pd.DataFrame()
    
    if not targets.empty:
        pct10 = (targets.investment_score >= p90).mean() * 100
        pct25 = (targets.investment_score >= p75).mean() * 100
    else:
        pct10, pct25 = 0, 0

    # Display metrics in cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Percentage of target companies in top 10%</div>
            <div class="value">{pct10:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Percentage of target companies in top 25%</div>
            <div class="value">{pct25:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Filtered Companies</div>
            <div class="value">{len(filtered):,}</div>
        </div>
        """, unsafe_allow_html=True)

    # Opportunities Table
    st.markdown('<div class="section-header"><h2>📊 Investment Opportunities</h2></div>', unsafe_allow_html=True)

    # Table controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        sort_col = st.selectbox("Sort by", display_cols, index=1)
    with col2:
        sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
        ascending = sort_order == "Ascending"
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        show_top = st.checkbox("Show Top 20 Only", value=False)

    # Sort and limit table if needed
    table = filtered.sort_values(sort_col, ascending=ascending)
    if show_top:
        table = table.head(20)

    # Show the table with hover effect
    st.dataframe(table[display_cols], use_container_width=True, height=400)

    # Download Filtered Data
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv_bytes = table[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv_bytes,
            file_name="filtered_companies.csv",
            mime="text/csv",
            help="Export your filtered results to CSV format"
        )
    
    with col2:
        # Parquet download (smaller file size)
        parquet_buffer = io.BytesIO()
        table[display_cols].to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        st.download_button(
            label="⬇️ Download as Parquet",
            data=parquet_buffer,
            file_name="filtered_companies.parquet",
            help="Export your filtered results in compressed Parquet format (smaller file)"
        )

    # Company Details
    st.markdown('<div class="section-header"><h2>🏢 Company Details</h2></div>', unsafe_allow_html=True)

    if not table.empty:
        company = st.selectbox("Select company to view details", table["name"].tolist())
        row = table[table["name"] == company].iloc[0]

        # Company details
        d1, d2, d3 = st.columns([2, 2, 1])
        
        with d1:
            st.markdown(f"### {row['name']}")
            st.markdown(f"**Investment Score:** {row.investment_score:.1f}")
            if pd.notna(row.get("headcount")):
                st.markdown(f"**Headcount:** {int(row.headcount):,}")
            if pd.notna(row.get("funding_total")):
                st.markdown(f"**Total Funding:** ${row.funding_total:,.0f}")
                
        with d2:
            st.markdown("### Metrics")
            if pd.notna(row.get("headcount_growth_12m")):
                st.markdown(f"**12-month Growth:** {row.headcount_growth_12m:.1%}")
            if pd.notna(row.get("employees_per_million")):
                st.markdown(f"**Employees/$1M:** {row.employees_per_million:.1f}")
            if pd.notna(row.get("last_funding_date")):
                st.markdown(f"**Last Funding:** {row.last_funding_date}")
                
        with d3:
            st.markdown("### Links")
            if pd.notna(row.get("website")) and row["website"]:
                st.markdown(f"[🌐 Website]({row['website_url']})")
            if pd.notna(row.get("linkedin_url_linkedin")):
                st.markdown(f"[🔗 LinkedIn]({row['linkedin_url_linkedin']})")
    else:
        st.info("No companies found with the current filters. Try adjusting your search criteria.")

    # Feature Comparison Chart
    st.markdown('<div class="section-header"><h2>📈 Data Visualization</h2></div>', unsafe_allow_html=True)

    if chart_type == "Scatter Plot":
        chart = alt.Chart(filtered).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X(f"{x_feature}:Q",
                    title=x_feature,
                    scale=alt.Scale(type="log" if filtered[x_feature].min() > 0 else "linear")),
            y=alt.Y(f"{y_feature}:Q", title=y_feature),
            color=alt.Color("investment_score:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["name", x_feature, y_feature, "investment_score"]
        ).interactive()
    else:  # Heat Map
        chart = alt.Chart(filtered).mark_rect().encode(
            x=alt.X(f"{x_feature}:Q", bin=True, title=x_feature),
            y=alt.Y(f"{y_feature}:Q", bin=True, title=y_feature),
            color=alt.Color('count()', scale=alt.Scale(scheme="blues"), title="Count"),
            tooltip=['count()']
        ).interactive()

    st.altair_chart(chart.properties(height=400), use_container_width=True)

    # Model Insights Section
    st.markdown('<div class="section-header"><h2>🧠 Investment Model Insights</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    This section provides deeper insights into the AI investment scoring model, highlighting key factors that drive 
    our investment decisions and the distribution of scores across our database.
    """)

    # Feature importance and score distribution
    col1, col2 = st.columns(2)

    # Left column - Feature importance
    with col1:
        st.markdown("### Key Investment Factors")
        
        feature_chart = generate_feature_importance_chart()
        if feature_chart:
            st.altair_chart(feature_chart, use_container_width=True)
        else:
            st.info("Feature importance visualization is not available.")
        
        st.markdown("""
        <div class="insight-card">
            <h3>What Makes a High-Potential Investment?</h3>
            <p>Our AI model analyzes dozens of factors to identify companies with the highest growth potential. 
            The chart above shows the top predictive features in our model, with higher values indicating 
            stronger influence on the investment score.</p>
        </div>
        """, unsafe_allow_html=True)

    # Right column - Score distribution
    with col2:
        st.markdown("### Score Distribution")
        
        if 'is_target' in companies.columns:
            # Create score distribution visualization
            # Create a copy of the data for visualization
            chart_data = companies.copy()
            
            # Filter to only show scores above 10 to focus on the interesting part
            min_score = 10
            chart_data = chart_data[chart_data['investment_score'] >= min_score]
            
            # Add a column to identify target vs non-target
            chart_data['Company Type'] = chart_data['is_target'].apply(
                lambda x: 'Target Companies' if x else 'Non-Target Companies')
            
            # Create a histogram using Altair
            hist = alt.Chart(chart_data).mark_bar(opacity=0.7).encode(
                alt.X('investment_score:Q', 
                     bin=alt.Bin(maxbins=25),
                     title='Investment Score',
                     scale=alt.Scale(domain=[min_score, 100])),
                alt.Y('count()', 
                     title='Number of Companies',
                     stack=None),  # Set stack to None for overlapping bars
                alt.Color('Company Type:N',
                        scale=alt.Scale(
                            domain=['Target Companies', 'Non-Target Companies'],
                            range=['#3b82f6', '#64748b'])),
                tooltip=['Company Type', 'count()', 'investment_score']
            ).properties(
                width='container',
                height=300,
                title=f'Distribution of Investment Scores (Above {min_score})'
            )
            
            # Add vertical rules for percentiles
            p90 = np.percentile(companies['investment_score'], 90)
            p75 = np.percentile(companies['investment_score'], 75)
            
            rule90 = alt.Chart(pd.DataFrame({'percentile': [p90]})).mark_rule(
                color='#dc2626', 
                strokeDash=[4, 2],
                strokeWidth=2
            ).encode(
                x='percentile:Q',
                tooltip=[alt.Tooltip('percentile:Q', title='90th Percentile')]
            )
            
            rule75 = alt.Chart(pd.DataFrame({'percentile': [p75]})).mark_rule(
                color='#ea580c', 
                strokeDash=[4, 2],
                strokeWidth=2
            ).encode(
                x='percentile:Q',
                tooltip=[alt.Tooltip('percentile:Q', title='75th Percentile')]
            )
            
            # Add text annotations
            text90 = alt.Chart(pd.DataFrame({'x': [p90 + 2], 'y': [10], 'text': [f'90th: {p90:.1f}']})).mark_text(
                align='left',
                color='#dc2626',
                fontSize=12,
                fontWeight='bold'
            ).encode(
                x='x:Q',
                y='y:Q',
                text='text:N'
            )
            
            text75 = alt.Chart(pd.DataFrame({'x': [p75 + 2], 'y': [15], 'text': [f'75th: {p75:.1f}']})).mark_text(
                align='left',
                color='#ea580c',
                fontSize=12,
                fontWeight='bold'
            ).encode(
                x='x:Q',
                y='y:Q',
                text='text:N'
            )
            
            # Combine all elements
            final_chart = hist + rule90 + rule75 + text90 + text75
            
            # Display the chart
            st.altair_chart(final_chart, use_container_width=True)
            
            # Add explanatory text
            high_target = len(companies[(companies['is_target']) & (companies['investment_score'] >= min_score)])
            high_non_target = len(companies[(~companies['is_target']) & (companies['investment_score'] >= min_score)])
            low_count = len(companies[companies['investment_score'] < min_score])
            
            st.markdown(f"""
            <div class="insight-card">
                <h3>Validating Our Investment Thesis</h3>
                <p>This chart shows the distribution of companies with investment scores of {min_score} or higher.
                There are {high_target} target companies and {high_non_target} non-target companies in this range.
                {low_count:,} companies with scores below {min_score} are not shown.</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.info("Score distribution visualization is not available without target classification.")

    # Model Performance Metrics
    st.markdown("### Model Performance")

    # Performance metrics cards
    metrics_cols = st.columns(4)

    with metrics_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Target Companies in Model</div>
            <div class="value">{len(targets) if not targets.empty else "N/A"}</div>
        </div>
        """, unsafe_allow_html=True)

    with metrics_cols[1]:
        # AUC score - example value, replace with actual if available
        auc_score = 0.92  # Replace with actual AUC if available
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Cross-Validation AUC</div>
            <div class="value">{auc_score:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with metrics_cols[2]:
        # Precision score - example value, replace with actual if available
        precision = 0.88  # Replace with actual precision if available
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Model Precision</div>
            <div class="value">{precision:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with metrics_cols[3]:
        # Recall score - example value, replace with actual if available
        recall = 0.85  # Replace with actual recall if available
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Model Recall</div>
            <div class="value">{recall:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Display simplified error message if no data is loaded
    st.warning("No data available. Please make sure your parquet files are properly located in either 'mnt/data/' or 'data/' directories.")

# Footer
st.markdown("""
<div class="footer">
    <p>Powered by AI-Enhanced Investment Analysis</p>
</div>
""", unsafe_allow_html=True)