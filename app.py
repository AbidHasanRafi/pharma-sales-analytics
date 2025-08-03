import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from dateutil.parser import parse
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Pharma Sales Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #e8f4f8;
    }
    .st-bb {
        background-color: white;
    }
    .st-at {
        background-color: #0068c9;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        background-image: none;
    }
    .metric-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to parse date
def parse_date(date_str):
    try:
        return parse(date_str)
    except:
        return None

# Function to load and preprocess data
@st.cache_data
def load_data(uploaded_file):
    try:
        # Try reading with default encoding
        df = pd.read_csv(uploaded_file)
        
        # Convert YearMonth to datetime if it exists
        if 'YearMonth' in df.columns:
            df['YearMonth'] = df['YearMonth'].apply(parse_date)
            # Drop rows where date parsing failed
            df = df[df['YearMonth'].notna()]
        
        # Data cleaning
        if 'SoldQnty' in df.columns:
            df['SoldQnty'] = pd.to_numeric(df['SoldQnty'], errors='coerce')
        if 'SoldTP' in df.columns:
            df['SoldTP'] = pd.to_numeric(df['SoldTP'], errors='coerce')
        if 'TargetQnty' in df.columns:
            df['TargetQnty'] = pd.to_numeric(df['TargetQnty'], errors='coerce')
        if 'TargetValue' in df.columns:
            df['TargetValue'] = pd.to_numeric(df['TargetValue'], errors='coerce')
        
        # Calculate achievement rates
        if 'TargetQnty' in df.columns and 'SoldQnty' in df.columns:
            df['QtyAchievementRate'] = (df['SoldQnty'] / df['TargetQnty']).replace([np.inf, -np.inf], np.nan)
        if 'TargetValue' in df.columns and 'SoldTP' in df.columns:
            df['ValueAchievementRate'] = (df['SoldTP'] / df['TargetValue']).replace([np.inf, -np.inf], np.nan)
        
        return df
    except UnicodeDecodeError:
        # If UTF-8 fails, try other encodings
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin1')
            
            if 'YearMonth' in df.columns:
                df['YearMonth'] = df['YearMonth'].apply(parse_date)
                df = df[df['YearMonth'].notna()]
            
            # Data cleaning
            if 'SoldQnty' in df.columns:
                df['SoldQnty'] = pd.to_numeric(df['SoldQnty'], errors='coerce')
            if 'SoldTP' in df.columns:
                df['SoldTP'] = pd.to_numeric(df['SoldTP'], errors='coerce')
            if 'TargetQnty' in df.columns:
                df['TargetQnty'] = pd.to_numeric(df['TargetQnty'], errors='coerce')
            if 'TargetValue' in df.columns:
                df['TargetValue'] = pd.to_numeric(df['TargetValue'], errors='coerce')
            
            # Calculate achievement rates
            if 'TargetQnty' in df.columns and 'SoldQnty' in df.columns:
                df['QtyAchievementRate'] = (df['SoldQnty'] / df['TargetQnty']).replace([np.inf, -np.inf], np.nan)
            if 'TargetValue' in df.columns and 'SoldTP' in df.columns:
                df['ValueAchievementRate'] = (df['SoldTP'] / df['TargetValue']).replace([np.inf, -np.inf], np.nan)
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

# Main App
def main():
    st.sidebar.title("Pharma Sales Analytics")
    
    # File upload
    st.sidebar.title("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your sales data CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.sidebar.success("✅ Data loaded successfully!")
            
            # Display basic info
            st.sidebar.markdown(f"""
            **Data Summary:**
            - Rows: {df.shape[0]:,}
            - Columns: {df.shape[1]}
            - Period: {df['YearMonth'].min().strftime('%b %Y') if 'YearMonth' in df.columns else 'N/A'} to {df['YearMonth'].max().strftime('%b %Y') if 'YearMonth' in df.columns else 'N/A'}
            """)
            
            # Sidebar navigation
            st.sidebar.title("Navigation")
            app_mode = st.sidebar.radio(
                "Go to",
                ["📊 Dashboard", "🔍 Data Explorer", "📈 Performance Analysis", "🤖 Predictive Analytics"],
                key="nav_radio"
            )
            
            # Dashboard Page
            if app_mode == "📊 Dashboard":
                st.title("Pharmaceutical Sales Dashboard")
                st.markdown("Comprehensive overview of sales performance across teams, products, and territories.")
                
                # KPI Cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_sales = df['SoldTP'].sum() / 1e6  # in millions
                    st.metric("Total Sales (TP)", f"${total_sales:.2f}M")
                with col2:
                    total_quantity = df['SoldQnty'].sum() / 1e3  # in thousands
                    st.metric("Total Quantity Sold", f"{total_quantity:.1f}K")
                with col3:
                    if 'QtyAchievementRate' in df.columns:
                        avg_achievement = df['QtyAchievementRate'].mean() * 100
                        st.metric("Avg. Achievement Rate", f"{avg_achievement:.1f}%")
                    else:
                        st.metric("Avg. Achievement Rate", "N/A")
                with col4:
                    if 'YearMonth' in df.columns:
                        months = df['YearMonth'].nunique()
                        st.metric("Months of Data", months)
                    else:
                        st.metric("Months of Data", "N/A")
                
                # First row of charts
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Sales by Therapeutic Group")
                    if 'Thgroup' in df.columns and 'SoldTP' in df.columns:
                        tg_sales = df.groupby('Thgroup')['SoldTP'].sum().sort_values(ascending=False)
                        fig = px.bar(
                            tg_sales,
                            x=tg_sales.index,
                            y=tg_sales.values,
                            color=tg_sales.values,
                            color_continuous_scale='Blues',
                            labels={'x': 'Therapeutic Group', 'y': 'Total Sales'}
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Therapeutic Group or Sales data not available")
                
                with col2:
                    st.subheader("Monthly Sales Trend")
                    if 'YearMonth' in df.columns and 'SoldTP' in df.columns:
                        monthly_sales = df.groupby('YearMonth')['SoldTP'].sum().reset_index()
                        fig = px.line(
                            monthly_sales,
                            x='YearMonth',
                            y='SoldTP',
                            markers=True,
                            labels={'YearMonth': 'Month', 'SoldTP': 'Total Sales'}
                        )
                        fig.update_traces(line_color='#0068c9')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Date or Sales data not available")
                
                # Second row of charts
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Top Performing Products")
                    if 'Pname' in df.columns and 'SoldTP' in df.columns:
                        top_products = df.groupby('Pname')['SoldTP'].sum().nlargest(10).reset_index()
                        fig = px.bar(
                            top_products,
                            x='SoldTP',
                            y='Pname',
                            orientation='h',
                            color='SoldTP',
                            color_continuous_scale='Blues',
                            labels={'Pname': 'Product', 'SoldTP': 'Total Sales'}
                        )
                        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Product or Sales data not available")
                
                with col2:
                    st.subheader("Team Performance")
                    if 'Team' in df.columns and 'SoldTP' in df.columns:
                        team_perf = df.groupby('Team')['SoldTP'].sum().sort_values(ascending=False).reset_index()
                        fig = px.pie(
                            team_perf,
                            names='Team',
                            values='SoldTP',
                            hole=0.3,
                            labels={'Team': 'Team', 'SoldTP': 'Total Sales'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Team or Sales data not available")
                
                # Third row - Achievement analysis
                st.subheader("Target Achievement Analysis")
                if 'TargetQnty' in df.columns and 'SoldQnty' in df.columns:
                    achievement_df = df.groupby('Team').agg({
                        'TargetQnty': 'sum',
                        'SoldQnty': 'sum'
                    }).reset_index()
                    achievement_df['AchievementRate'] = (achievement_df['SoldQnty'] / achievement_df['TargetQnty']).replace([np.inf, -np.inf], np.nan)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(
                            achievement_df,
                            x='Team',
                            y=['TargetQnty', 'SoldQnty'],
                            barmode='group',
                            labels={'value': 'Quantity', 'variable': 'Metric'},
                            title="Target vs Actual Quantity"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            achievement_df,
                            x='Team',
                            y='AchievementRate',
                            labels={'AchievementRate': 'Achievement Rate'},
                            title="Achievement Rate by Team"
                        )
                        fig.update_traces(marker_color='#4CAF50')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Target or Sales quantity data not available")
            
            # Data Explorer Page
            elif app_mode == "🔍 Data Explorer":
                st.title("Data Explorer")
                st.markdown("Explore and analyze the pharmaceutical sales data in detail.")
                
                # Filters
                with st.expander("🔍 Filters", expanded=True):
                    cols = st.columns(3)
                    
                    # Dynamic filter creation based on available columns
                    filters = {}
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    for i, col in enumerate(categorical_cols):
                        with cols[i % 3]:
                            if df[col].nunique() < 50:  # Only show filter for columns with reasonable unique values
                                selected = st.multiselect(
                                    f"Filter by {col}",
                                    df[col].unique(),
                                    default=df[col].unique()[:5] if df[col].nunique() > 5 else df[col].unique(),
                                    key=f"filter_{col}"
                                )
                                filters[col] = selected
                    
                    # Numeric filters
                    with st.expander("🔢 Numeric Filters"):
                        num_cols = st.columns(3)
                        for i, col in enumerate(df.select_dtypes(include=['number']).columns):
                            with num_cols[i % 3]:
                                min_val = float(df[col].min())
                                max_val = float(df[col].max())
                                step = (max_val - min_val) / 100
                                val_range = st.slider(
                                    f"Range for {col}",
                                    min_val,
                                    max_val,
                                    (min_val, max_val),
                                    step=step,
                                    key=f"range_{col}"
                                )
                                filters[col] = val_range
                    
                    # Date filter if available
                    if 'YearMonth' in df.columns:
                        with st.expander("📅 Date Filter"):
                            date_range = st.date_input(
                                "Select Date Range",
                                [df['YearMonth'].min(), df['YearMonth'].max()],
                                key="date_range"
                            )
                
                # Apply filters
                filtered_df = df.copy()
                
                # Apply categorical filters
                for col, values in filters.items():
                    if col in df.select_dtypes(include=['object', 'category']).columns:
                        filtered_df = filtered_df[filtered_df[col].isin(values)]
                    elif col in df.select_dtypes(include=['number']).columns:
                        min_val, max_val = values
                        filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
                
                # Apply date filter
                if 'YearMonth' in df.columns and 'date_range' in locals():
                    filtered_df = filtered_df[
                        (filtered_df['YearMonth'] >= pd.to_datetime(date_range[0])) &
                        (filtered_df['YearMonth'] <= pd.to_datetime(date_range[1]))
                    ]
                
                # Show filtered data
                st.subheader("Filtered Data")
                st.dataframe(filtered_df.head(100))
                
                # Data statistics
                with st.expander("📊 Filtered Data Statistics"):
                    st.write(filtered_df.describe())
                
                # Visualizations
                st.subheader("📈 Interactive Visualizations")
                
                chart_type = st.selectbox(
                    "Select Chart Type",
                    ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"],
                    key="chart_type_selector"
                )
                
                if chart_type == "Bar Chart":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_axis = st.selectbox(
                            "X-axis",
                            filtered_df.select_dtypes(include=['object', 'category']).columns,
                            key="bar_x_axis"
                        )
                    with col2:
                        y_axis = st.selectbox(
                            "Y-axis",
                            filtered_df.select_dtypes(include=['number']).columns,
                            key="bar_y_axis"
                        )
                    
                    agg_func = st.selectbox(
                        "Aggregation",
                        ["sum", "mean", "count"],
                        key="bar_agg"
                    )
                    
                    grouped_data = filtered_df.groupby(x_axis)[y_axis].agg(agg_func).reset_index()
                    
                    fig = px.bar(
                        grouped_data,
                        x=x_axis,
                        y=y_axis,
                        title=f"{agg_func.title()} of {y_axis} by {x_axis}",
                        color=y_axis,
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Line Chart":
                    x_options = ['YearMonth'] if 'YearMonth' in filtered_df.columns else []
                    x_options += list(filtered_df.select_dtypes(include=['datetime']).columns)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        x_axis = st.selectbox(
                            "X-axis",
                            x_options,
                            key="line_x_axis"
                        )
                    with col2:
                        y_axis = st.selectbox(
                            "Y-axis",
                            filtered_df.select_dtypes(include=['number']).columns,
                            key="line_y_axis"
                        )
                    
                    group_by = st.selectbox(
                        "Group by",
                        [None] + list(filtered_df.select_dtypes(include=['object', 'category']).columns),
                        key="line_group"
                    )
                    
                    if group_by:
                        fig = px.line(
                            filtered_df,
                            x=x_axis,
                            y=y_axis,
                            color=group_by,
                            title=f"{y_axis} Trend by {group_by}"
                        )
                    else:
                        fig = px.line(
                            filtered_df.groupby(x_axis)[y_axis].sum().reset_index(),
                            x=x_axis,
                            y=y_axis,
                            title=f"{y_axis} Trend"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Scatter Plot":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_axis = st.selectbox(
                            "X-axis",
                            filtered_df.select_dtypes(include=['number']).columns,
                            key="scatter_x_axis"
                        )
                    with col2:
                        y_axis = st.selectbox(
                            "Y-axis",
                            filtered_df.select_dtypes(include=['number']).columns,
                            key="scatter_y_axis"
                        )
                    with col3:
                        color_by = st.selectbox(
                            "Color by",
                            [None] + list(filtered_df.select_dtypes(include=['object', 'category']).columns),
                            key="scatter_color"
                        )
                    
                    size_by = st.selectbox(
                        "Size by",
                        [None] + list(filtered_df.select_dtypes(include=['number']).columns),
                        key="scatter_size"
                    )
                    
                    fig = px.scatter(
                        filtered_df,
                        x=x_axis,
                        y=y_axis,
                        color=color_by,
                        size=size_by,
                        title=f"{y_axis} vs {x_axis}",
                        hover_data=list(filtered_df.select_dtypes(include=['object', 'category']).columns)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Histogram":
                    col = st.selectbox(
                        "Column",
                        filtered_df.select_dtypes(include=['number']).columns,
                        key="hist_col"
                    )
                    
                    bins = st.slider(
                        "Number of bins",
                        5, 100, 20,
                        key="hist_bins"
                    )
                    
                    fig = px.histogram(
                        filtered_df,
                        x=col,
                        nbins=bins,
                        title=f"Distribution of {col}",
                        color_discrete_sequence=['#0068c9']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Box Plot":
                    col1, col2 = st.columns(2)
                    with col1:
                        col = st.selectbox(
                            "Column",
                            filtered_df.select_dtypes(include=['number']).columns,
                            key="box_col"
                        )
                    with col2:
                        group_by = st.selectbox(
                            "Group by",
                            [None] + list(filtered_df.select_dtypes(include=['object', 'category']).columns),
                            key="box_group"
                        )
                    
                    fig = px.box(
                        filtered_df,
                        x=group_by,
                        y=col,
                        title=f"Distribution of {col} by {group_by if group_by else 'Overall'}",
                        color=group_by
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Pie Chart":
                    col = st.selectbox(
                        "Column",
                        filtered_df.select_dtypes(include=['object', 'category']).columns,
                        key="pie_col"
                    )
                    
                    if filtered_df[col].nunique() <= 20:
                        fig = px.pie(
                            filtered_df,
                            names=col,
                            title=f"Distribution of {col}",
                            hole=0.3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Too many unique values in {col} for a pie chart. Please select another column.")
            
            # Performance Analysis Page
            elif app_mode == "📈 Performance Analysis":
                st.title("Performance Analysis")
                st.markdown("Deep dive into team, product, and territory performance metrics.")
                
                tab1, tab2, tab3 = st.tabs(["Team Performance", "Product Performance", "Target Achievement"])
                
                with tab1:
                    st.subheader("Team Performance Analysis")
                    
                    if 'Team' in df.columns:
                        # Team selection
                        selected_teams = st.multiselect(
                            "Select Teams to Compare",
                            df['Team'].unique(),
                            default=df['Team'].unique()[:3],
                            key="team_perf_selector"
                        )
                        
                        # Metric selection
                        metric_options = []
                        if 'SoldQnty' in df.columns:
                            metric_options.append("Quantity Sold")
                        if 'SoldTP' in df.columns:
                            metric_options.append("Sales Value")
                        if 'QtyAchievementRate' in df.columns:
                            metric_options.append("Achievement Rate")
                        
                        if metric_options:
                            metric = st.radio(
                                "Performance Metric",
                                metric_options,
                                key="perf_metric"
                            )
                            
                            # Map display names to actual columns
                            metric_map = {
                                "Quantity Sold": "SoldQnty",
                                "Sales Value": "SoldTP",
                                "Achievement Rate": "QtyAchievementRate"
                            }
                            metric_col = metric_map[metric]
                            
                            # Time period selection
                            time_period = st.selectbox(
                                "Time Period",
                                ["Overall", "Monthly", "Quarterly", "Yearly"],
                                key="time_period_selector"
                            )
                            
                            # Filter data
                            team_df = df[df['Team'].isin(selected_teams)]
                            
                            # Aggregate based on time period
                            if time_period == "Overall":
                                grouped = team_df.groupby('Team')[metric_col].mean().reset_index()
                                x_axis = 'Team'
                                title = f"Average {metric} by Team"
                            elif time_period == "Monthly" and 'YearMonth' in df.columns:
                                grouped = team_df.groupby(['Team', 'YearMonth'])[metric_col].mean().reset_index()
                                x_axis = 'YearMonth'
                                title = f"{metric} Trend by Month"
                            elif time_period == "Quarterly" and 'YearMonth' in df.columns:
                                team_df['Quarter'] = team_df['YearMonth'].dt.to_period('Q').astype(str)
                                grouped = team_df.groupby(['Team', 'Quarter'])[metric_col].mean().reset_index()
                                x_axis = 'Quarter'
                                title = f"{metric} Trend by Quarter"
                            elif time_period == "Yearly" and 'YearMonth' in df.columns:
                                team_df['Year'] = team_df['YearMonth'].dt.year
                                grouped = team_df.groupby(['Team', 'Year'])[metric_col].mean().reset_index()
                                x_axis = 'Year'
                                title = f"{metric} Trend by Year"
                            else:
                                st.warning("Selected time period not available with current data")
                                grouped = pd.DataFrame()
                            
                            if not grouped.empty:
                                # Plot
                                if time_period == "Overall":
                                    fig = px.bar(
                                        grouped,
                                        x=x_axis,
                                        y=metric_col,
                                        color='Team',
                                        title=title,
                                        text=metric_col
                                    )
                                    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
                                else:
                                    fig = px.line(
                                        grouped,
                                        x=x_axis,
                                        y=metric_col,
                                        color='Team',
                                        title=title,
                                        markers=True
                                    )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Performance table
                                st.subheader("Performance Summary")
                                if time_period == "Overall":
                                    st.dataframe(
                                        grouped.sort_values(metric_col, ascending=False).style
                                        .background_gradient(cmap='Blues', subset=[metric_col])
                                        .format({metric_col: "{:.2f}"})
                                    )
                                else:
                                    pivot_table = pd.pivot_table(
                                        grouped,
                                        values=metric_col,
                                        index='Team',
                                        columns=x_axis,
                                        aggfunc='mean'
                                    )
                                    st.dataframe(
                                        pivot_table.style
                                        .background_gradient(cmap='Blues')
                                        .format("{:.2f}")
                                    )
                            else:
                                st.warning("No data available for the selected filters")
                        else:
                            st.warning("No performance metrics available in the data")
                    else:
                        st.warning("Team information not found in the data")
                
                with tab2:
                    st.subheader("Product Performance Analysis")
                    
                    if 'Pname' in df.columns:
                        # Product selection
                        selected_products = st.multiselect(
                            "Select Products",
                            df['Pname'].unique(),
                            default=df['Pname'].unique()[:5],
                            key="product_selector"
                        )
                        
                        # Metric selection
                        metric_options = []
                        if 'SoldQnty' in df.columns:
                            metric_options.append("Quantity Sold")
                        if 'SoldTP' in df.columns:
                            metric_options.append("Sales Value")
                        if 'QtyAchievementRate' in df.columns:
                            metric_options.append("Achievement Rate")
                        
                        if metric_options:
                            product_metric = st.radio(
                                "Product Metric",
                                metric_options,
                                key="product_metric"
                            )
                            
                            # Map display names to actual columns
                            metric_map = {
                                "Quantity Sold": "SoldQnty",
                                "Sales Value": "SoldTP",
                                "Achievement Rate": "QtyAchievementRate"
                            }
                            metric_col = metric_map[product_metric]
                            
                            # Filter data
                            product_df = df[df['Pname'].isin(selected_products)]
                            
                            # Time period selection
                            time_period = st.selectbox(
                                "Time Period",
                                ["Overall", "Monthly", "Quarterly", "Yearly"],
                                key="product_time_period"
                            )
                            
                            # Aggregate based on time period
                            if time_period == "Overall":
                                grouped = product_df.groupby('Pname')[metric_col].mean().reset_index()
                                x_axis = 'Pname'
                                title = f"Average {product_metric} by Product"
                            elif time_period == "Monthly" and 'YearMonth' in df.columns:
                                grouped = product_df.groupby(['Pname', 'YearMonth'])[metric_col].mean().reset_index()
                                x_axis = 'YearMonth'
                                title = f"{product_metric} Trend by Month"
                            elif time_period == "Quarterly" and 'YearMonth' in df.columns:
                                product_df['Quarter'] = product_df['YearMonth'].dt.to_period('Q').astype(str)
                                grouped = product_df.groupby(['Pname', 'Quarter'])[metric_col].mean().reset_index()
                                x_axis = 'Quarter'
                                title = f"{product_metric} Trend by Quarter"
                            elif time_period == "Yearly" and 'YearMonth' in df.columns:
                                product_df['Year'] = product_df['YearMonth'].dt.year
                                grouped = product_df.groupby(['Pname', 'Year'])[metric_col].mean().reset_index()
                                x_axis = 'Year'
                                title = f"{product_metric} Trend by Year"
                            else:
                                st.warning("Selected time period not available with current data")
                                grouped = pd.DataFrame()
                            
                            if not grouped.empty:
                                # Plot
                                if time_period == "Overall":
                                    fig = px.bar(
                                        grouped,
                                        x=metric_col,
                                        y='Pname',
                                        orientation='h',
                                        color=metric_col,
                                        color_continuous_scale='Blues',
                                        title=title,
                                        labels={'Pname': 'Product', metric_col: product_metric}
                                    )
                                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                                else:
                                    fig = px.line(
                                        grouped,
                                        x=x_axis,
                                        y=metric_col,
                                        color='Pname',
                                        title=title,
                                        markers=True
                                    )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Performance table
                                st.subheader("Performance Summary")
                                if time_period == "Overall":
                                    st.dataframe(
                                        grouped.sort_values(metric_col, ascending=False).style
                                        .background_gradient(cmap='Blues', subset=[metric_col])
                                        .format({metric_col: "{:.2f}"})
                                    )
                                else:
                                    pivot_table = pd.pivot_table(
                                        grouped,
                                        values=metric_col,
                                        index='Pname',
                                        columns=x_axis,
                                        aggfunc='mean'
                                    )
                                    st.dataframe(
                                        pivot_table.style
                                        .background_gradient(cmap='Blues')
                                        .format("{:.2f}")
                                    )
                            else:
                                st.warning("No data available for the selected filters")
                        else:
                            st.warning("No product metrics available in the data")
                    else:
                        st.warning("Product information not found in the data")
                
                with tab3:
                    st.subheader("Target Achievement Analysis")
                    
                    # Hierarchy selection
                    level_options = []
                    if 'RSM' in df.columns:
                        level_options.append("Region (RSM)")
                    if 'FM' in df.columns:
                        level_options.append("Field Manager (FM)")
                    if 'MPO' in df.columns:
                        level_options.append("Medical Rep (MPO)")
                    if 'Team' in df.columns:
                        level_options.append("Team")
                    if 'Emp Code' in df.columns:
                        level_options.append("Employee")
                    if 'Pname' in df.columns:
                        level_options.append("Product")
                    
                    if level_options:
                        level = st.selectbox(
                            "Analysis Level",
                            level_options,
                            key="analysis_level"
                        )
                        
                        # Map display names to actual columns
                        level_map = {
                            "Region (RSM)": "RSM",
                            "Field Manager (FM)": "FM",
                            "Medical Rep (MPO)": "MPO",
                            "Team": "Team",
                            "Employee": "Emp Code",
                            "Product": "Pname"
                        }
                        group_col = level_map[level]
                        
                        # Time period selection
                        time_period = st.selectbox(
                            "Time Period",
                            ["Overall", "Monthly", "Quarterly", "Yearly"],
                            key="achievement_time_period"
                        )
                        
                        # Filter data
                        if time_period == "Overall":
                            target_df = df.copy()
                        elif time_period == "Monthly" and 'YearMonth' in df.columns:
                            months = st.multiselect(
                                "Select Months",
                                df['YearMonth'].dt.strftime('%b %Y').unique(),
                                default=df['YearMonth'].dt.strftime('%b %Y').unique()[:3],
                                key="achievement_months"
                            )
                            target_df = df[df['YearMonth'].dt.strftime('%b %Y').isin(months)]
                        elif time_period == "Quarterly" and 'YearMonth' in df.columns:
                            quarters = st.multiselect(
                                "Select Quarters",
                                df['YearMonth'].dt.to_period('Q').astype(str).unique(),
                                default=df['YearMonth'].dt.to_period('Q').astype(str).unique()[:2],
                                key="achievement_quarters"
                            )
                            target_df = df[df['YearMonth'].dt.to_period('Q').astype(str).isin(quarters)]
                        elif time_period == "Yearly" and 'YearMonth' in df.columns:
                            years = st.multiselect(
                                "Select Years",
                                df['YearMonth'].dt.year.unique(),
                                default=df['YearMonth'].dt.year.unique(),
                                key="achievement_years"
                            )
                            target_df = df[df['YearMonth'].dt.year.isin(years)]
                        else:
                            st.warning("Selected time period not available with current data")
                            target_df = pd.DataFrame()
                        
                        if not target_df.empty:
                            # Calculate achievement metrics
                            agg_dict = {}
                            if 'TargetQnty' in df.columns:
                                agg_dict['TargetQnty'] = 'sum'
                            if 'SoldQnty' in df.columns:
                                agg_dict['SoldQnty'] = 'sum'
                            if 'TargetValue' in df.columns:
                                agg_dict['TargetValue'] = 'sum'
                            if 'SoldTP' in df.columns:
                                agg_dict['SoldTP'] = 'sum'
                            
                            if agg_dict:
                                achievement_df = target_df.groupby(group_col).agg(agg_dict).reset_index()
                                
                                # Calculate achievement rates if possible
                                if 'TargetQnty' in agg_dict and 'SoldQnty' in agg_dict:
                                    achievement_df['QtyAchievementRate'] = (achievement_df['SoldQnty'] / achievement_df['TargetQnty']) * 100
                                
                                if 'TargetValue' in agg_dict and 'SoldTP' in agg_dict:
                                    achievement_df['ValueAchievementRate'] = (achievement_df['SoldTP'] / achievement_df['TargetValue']) * 100
                                
                                # Display metrics
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if 'QtyAchievementRate' in achievement_df.columns:
                                        avg_qty = achievement_df['QtyAchievementRate'].mean()
                                        st.metric(
                                            "Average Quantity Achievement",
                                            f"{avg_qty:.1f}%",
                                            delta=f"{(avg_qty - 100):.1f}% vs Target" if avg_qty else None
                                        )
                                
                                with col2:
                                    if 'ValueAchievementRate' in achievement_df.columns:
                                        avg_val = achievement_df['ValueAchievementRate'].mean()
                                        st.metric(
                                            "Average Value Achievement",
                                            f"{avg_val:.1f}%",
                                            delta=f"{(avg_val - 100):.1f}% vs Target" if avg_val else None
                                        )
                                
                                # Achievement visualization
                                if 'TargetQnty' in achievement_df.columns and 'SoldQnty' in achievement_df.columns:
                                    st.subheader("Target vs Actual Quantity")
                                    fig = px.bar(
                                        achievement_df.sort_values('QtyAchievementRate', ascending=False),
                                        x=group_col,
                                        y=['TargetQnty', 'SoldQnty'],
                                        barmode='group',
                                        title="Quantity Performance"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Achievement rate visualization
                                if 'QtyAchievementRate' in achievement_df.columns:
                                    st.subheader("Achievement Rate")
                                    fig = px.bar(
                                        achievement_df.sort_values('QtyAchievementRate', ascending=False),
                                        x=group_col,
                                        y='QtyAchievementRate',
                                        title="Achievement Rate",
                                        text='QtyAchievementRate',
                                        color='QtyAchievementRate',
                                        color_continuous_scale='RdYlGn',
                                        range_color=[0, 200]
                                    )
                                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                                    fig.add_hline(y=100, line_dash="dash", line_color="red")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Top/Bottom performers
                                if 'QtyAchievementRate' in achievement_df.columns:
                                    st.subheader("Top/Bottom Performers")
                                    top_n = st.slider(
                                        "Number of Performers to Show",
                                        3, 10, 5,
                                        key="performer_slider"
                                    )
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    top_performers = achievement_df.nlargest(top_n, 'QtyAchievementRate')
                                    fig1 = px.bar(
                                        top_performers,
                                        x=group_col,
                                        y='QtyAchievementRate',
                                        title=f"Top {top_n} Performers",
                                        text='QtyAchievementRate',
                                        color='QtyAchievementRate',
                                        color_continuous_scale='Greens',
                                        range_color=[100, 200]
                                    )
                                    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                                    fig1.update_layout(showlegend=False)
                                    st.plotly_chart(fig1, use_container_width=True)
                                
                                with col2:
                                    bottom_performers = achievement_df.nsmallest(top_n, 'QtyAchievementRate')
                                    fig2 = px.bar(
                                        bottom_performers,
                                        x=group_col,
                                        y='QtyAchievementRate',
                                        title=f"Bottom {top_n} Performers",
                                        text='QtyAchievementRate',
                                        color='QtyAchievementRate',
                                        color_continuous_scale='Reds',
                                        range_color=[0, 100]
                                    )
                                    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                                    fig2.update_layout(showlegend=False)
                                    st.plotly_chart(fig2, use_container_width=True)
                            
                            # Detailed performance table
                            st.subheader("Detailed Performance")
                            display_cols = [group_col]
                            if 'TargetQnty' in achievement_df.columns:
                                display_cols.append('TargetQnty')
                            if 'SoldQnty' in achievement_df.columns:
                                display_cols.append('SoldQnty')
                            if 'QtyAchievementRate' in achievement_df.columns:
                                display_cols.append('QtyAchievementRate')
                            if 'TargetValue' in achievement_df.columns:
                                display_cols.append('TargetValue')
                            if 'SoldTP' in achievement_df.columns:
                                display_cols.append('SoldTP')
                            if 'ValueAchievementRate' in achievement_df.columns:
                                display_cols.append('ValueAchievementRate')
                            
                            st.dataframe(
                                achievement_df[display_cols].sort_values(
                                    'QtyAchievementRate' if 'QtyAchievementRate' in achievement_df.columns else group_col,
                                    ascending=False
                                ).style
                                .background_gradient(cmap='RdYlGn', subset=['QtyAchievementRate'] if 'QtyAchievementRate' in achievement_df.columns else [])
                                .background_gradient(cmap='RdYlGn', subset=['ValueAchievementRate'] if 'ValueAchievementRate' in achievement_df.columns else [])
                                .format({
                                    'QtyAchievementRate': '{:.1f}%',
                                    'ValueAchievementRate': '{:.1f}%',
                                    'TargetQnty': '{:,.0f}',
                                    'SoldQnty': '{:,.0f}',
                                    'TargetValue': '${:,.0f}',
                                    'SoldTP': '${:,.0f}'
                                })
                            )
                        else:
                            st.warning("No target/achievement metrics found in the data")
                    else:
                        st.warning("No data available for the selected filters")
            else:
                st.warning("No hierarchy levels found in the data")
        
        # Predictive Analytics Page
        elif app_mode == "🤖 Predictive Analytics":
            st.title("Predictive Analytics")
            st.markdown("Leverage machine learning to forecast sales and predict performance.")
            
            tab1, tab2, tab3 = st.tabs(["Sales Forecasting", "Performance Prediction", "Recommendation Engine"])
            
            with tab1:
                st.subheader("Sales Forecasting")
                st.markdown("Build models to forecast future sales quantities or values.")
                
                # Model configuration
                with st.expander("Model Configuration", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Target variable selection
                        target_options = []
                        if 'SoldQnty' in df.columns:
                            target_options.append("Quantity Sold (SoldQnty)")
                        if 'SoldTP' in df.columns:
                            target_options.append("Sales Value (SoldTP)")
                        
                        if target_options:
                            target = st.selectbox(
                                "Target Variable",
                                target_options,
                                key="forecast_target"
                            )
                            target_col = target.split("(")[1].replace(")", "").strip()
                        else:
                            st.warning("No suitable target variables found")
                            target_col = None
                    
                    with col2:
                        # Model selection
                        model_type = st.selectbox(
                            "Model Algorithm",
                            ["Random Forest", "XGBoost", "Linear Regression", "Prophet (Time Series)"],
                            key="forecast_model"
                        )
                
                # Feature selection
                if target_col:
                    st.subheader("Feature Selection")
                    st.markdown("Select features to include in the forecasting model.")
                    
                    # Available features
                    feature_options = []
                    if 'RSM' in df.columns:
                        feature_options.append('RSM')
                    if 'FM' in df.columns:
                        feature_options.append('FM')
                    if 'MPO' in df.columns:
                        feature_options.append('MPO')
                    if 'Team' in df.columns:
                        feature_options.append('Team')
                    if 'Pcode' in df.columns:
                        feature_options.append('Pcode')
                    if 'Pname' in df.columns:
                        feature_options.append('Pname')
                    if 'Brand' in df.columns:
                        feature_options.append('Brand')
                    if 'Thgroup' in df.columns:
                        feature_options.append('Thgroup')
                    if 'TargetQnty' in df.columns:
                        feature_options.append('TargetQnty')
                    if 'TargetValue' in df.columns:
                        feature_options.append('TargetValue')
                    if 'YearMonth' in df.columns:
                        feature_options.append('YearMonth')
                    
                    features = st.multiselect(
                        "Select Features",
                        feature_options,
                        default=[f for f in ['Pcode', 'Brand', 'Thgroup', 'TargetQnty'] if f in feature_options],
                        key="forecast_features"
                    )
                    
                    # Time horizon for forecasting
                    forecast_period = st.selectbox(
                        "Forecast Period",
                        ["Next Month", "Next Quarter", "Next 6 Months", "Next Year"],
                        key="forecast_period"
                    )
                    
                    # Train/test split
                    test_size = st.slider(
                        "Test Set Size (%)",
                        10, 40, 20,
                        key="test_size"
                    ) / 100
                
                # Model training and evaluation
                if target_col and features and st.button("Train Model", key="train_model"):
                    st.subheader("Model Training")
                    
                    with st.spinner("Training model..."):
                        try:
                            # Prepare data
                            X = df[features]
                            y = df[target_col]
                            
                            # Handle categorical features
                            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                            if len(categorical_cols) > 0:
                                X_processed = pd.get_dummies(X, columns=categorical_cols)
                            else:
                                X_processed = X.copy()
                            
                            # Handle missing values
                            X_processed = X_processed.fillna(0)
                            y = y.fillna(0)
                            
                            # Train-test split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_processed, y, test_size=test_size, random_state=42
                            )
                            
                            # Initialize and train model
                            if model_type == "Random Forest":
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                            elif model_type == "XGBoost":
                                try:
                                    from xgboost import XGBRegressor
                                    model = XGBRegressor(random_state=42)
                                except ImportError:
                                    st.warning("xgboost is not installed. Please install it to use XGBRegressor.")
                                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                            elif model_type == "Linear Regression":
                                from sklearn.linear_model import LinearRegression
                                model = LinearRegression()
                            else:  # Prophet
                                st.warning("Prophet model not implemented in this example")
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                            
                            model.fit(X_train, y_train)
                            
                            # Evaluate model
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            mae = mean_absolute_error(y_test, y_pred)
                            from sklearn.metrics import mean_squared_error
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean Absolute Error", f"{mae:.2f}")
                            with col2:
                                st.metric("Mean Squared Error", f"{mse:.2f}")
                            with col3:
                                st.metric("R-squared Score", f"{r2:.2f}")
                            
                            # Feature importance
                            if hasattr(model, 'feature_importances_'):
                                st.subheader("Feature Importance")
                                importance_df = pd.DataFrame({
                                    'Feature': X_processed.columns,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(
                                    importance_df.head(10),
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title="Top 10 Most Important Features",
                                    color='Importance',
                                    color_continuous_scale='Blues'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Actual vs Predicted plot
                            st.subheader("Actual vs Predicted Values")
                            results_df = pd.DataFrame({
                                'Actual': y_test,
                                'Predicted': y_pred
                            }).head(100)
                            
                            fig = px.scatter(
                                results_df,
                                x='Actual',
                                y='Predicted',
                                trendline="lowess",
                                title="Actual vs Predicted Values",
                                labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'}
                            )
                            fig.add_shape(
                                type="line",
                                x0=min(y_test.min(), y_pred.min()),
                                y0=min(y_test.min(), y_pred.min()),
                                x1=max(y_test.max(), y_pred.max()),
                                y1=max(y_test.max(), y_pred.max()),
                                line=dict(color="Red", width=2, dash="dot")
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Forecast visualization (placeholder)
                            st.subheader("Forecast Results")
                            st.info("Note: This is a placeholder visualization. In a real implementation, you would generate actual forecasts.")
                            
                            # Sample forecast data
                            forecast_dates = pd.date_range(
                                start=df['YearMonth'].max(),
                                periods=6,
                                freq='M'
                            )
                            forecast_values = [y_pred.mean() * (0.9 + 0.1*i) for i in range(6)]
                            
                            fig = px.line(
                                x=forecast_dates,
                                y=forecast_values,
                                title=f"Forecasted {target} for Next 6 Months",
                                labels={'x': 'Month', 'y': target}
                            )
                            fig.update_traces(line_color='#FFA500', line_width=3)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error in model training: {str(e)}")
            
            with tab2:
                st.subheader("Performance Prediction")
                st.markdown("Predict future performance of teams, products, or employees.")
                
                # Prediction type
                pred_options = []
                if 'Team' in df.columns:
                    pred_options.append("Team")
                if 'Pname' in df.columns:
                    pred_options.append("Product")
                if 'Emp Code' in df.columns:
                    pred_options.append("Employee")
                
                if pred_options:
                    pred_type = st.radio(
                        "Predict for",
                        pred_options,
                        key="pred_type"
                    )
                    
                    # Item selection
                    if pred_type == "Team":
                        selected_item = st.selectbox(
                            "Select Team",
                            df['Team'].unique(),
                            key="team_pred"
                        )
                    elif pred_type == "Product":
                        selected_item = st.selectbox(
                            "Select Product",
                            df['Pname'].unique(),
                            key="product_pred"
                        )
                    else:
                        selected_item = st.selectbox(
                            "Select Employee",
                            df['Emp Code'].unique(),
                            key="emp_pred"
                        )
                    
                    # Prediction parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        pred_period = st.selectbox(
                            "Prediction Period",
                            ["Next Month", "Next Quarter", "Next 6 Months"],
                            key="pred_period"
                        )
                    
                    with col2:
                        confidence_level = st.slider(
                            "Confidence Level",
                            50, 95, 80,
                            key="confidence_level"
                        )
                    
                    if st.button("Generate Prediction", key="generate_prediction"):
                        st.subheader("Prediction Results")
                        st.info("Note: This is a simulated prediction. In a real implementation, you would use actual ML models.")
                        
                        # Placeholder results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Predicted Sales Quantity",
                                "1,250 units",
                                delta="+5% from last period"
                            )
                        
                        with col2:
                            st.metric(
                                "Predicted Sales Value",
                                "$25,000",
                                delta="+8% from last period"
                            )
                        
                        with col3:
                            st.metric(
                                "Confidence Interval",
                                f"±{100 - confidence_level}%",
                                delta=f"{confidence_level}% confidence"
                            )
                        
                        # Historical vs predicted chart
                        st.subheader("Historical vs Predicted Performance")
                        
                        # Sample data
                        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
                        historical = [1000, 1200, 950, 1100, 1050, 1250, None]
                        predicted = [None, None, None, None, None, 1250, 1350]
                        lower_bound = [None, None, None, None, None, 1150, 1250]
                        upper_bound = [None, None, None, None, None, 1350, 1450]
                        
                        fig = px.line(
                            x=months,
                            y=historical,
                            title="Sales Trend (Historical and Predicted)",
                            labels={'x': 'Month', 'y': 'Sales Quantity'}
                        )
                        
                        # Add predicted line
                        fig.add_scatter(
                            x=months,
                            y=predicted,
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='orange', width=3)
                        )
                        
                        # Add confidence interval
                        fig.add_scatter(
                            x=months,
                            y=upper_bound,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            name='Upper Bound'
                        )
                        
                        fig.add_scatter(
                            x=months,
                            y=lower_bound,
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(255, 165, 0, 0.2)',
                            name=f'{confidence_level}% Confidence'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Key drivers
                        st.subheader("Key Performance Drivers")
                        drivers = [
                            "Strong performance in key accounts",
                            "Seasonal demand increase",
                            "Recent marketing campaign",
                            "Competitor stock issues"
                        ]
                        
                        for driver in drivers:
                            st.markdown(f"- {driver}")
                else:
                    st.warning("No prediction types available in the data")
            
            with tab3:
                st.subheader("Recommendation Engine")
                st.markdown("Get personalized recommendations to improve sales performance.")
                
                # Recommendation type
                rec_options = []
                if 'Team' in df.columns:
                    rec_options.append("For Team")
                if 'Pname' in df.columns:
                    rec_options.append("For Product")
                if 'RSM' in df.columns:
                    rec_options.append("For Territory")
                
                if rec_options:
                    rec_type = st.radio(
                        "Recommendation Type",
                        rec_options,
                        key="rec_type"
                    )
                    
                    if rec_type == "For Team":
                        selected_team = st.selectbox(
                            "Select Team",
                            df['Team'].unique(),
                            key="rec_team"
                        )
                        
                        if st.button("Generate Recommendations", key="team_rec"):
                            st.subheader(f"Recommendations for {selected_team}")
                            
                            # Team performance analysis
                            team_df = df[df['Team'] == selected_team]
                            
                            # Product focus recommendations
                            if 'Pname' in df.columns and 'SoldQnty' in df.columns:
                                top_products = team_df.groupby('Pname')['SoldQnty'].sum().nlargest(3).index.tolist()
                                st.markdown("""
                                <div class="highlight">
                                    <h4>📌 Product Focus</h4>
                                    <p>Continue promoting your top performing products:</p>
                                    <ul>
                                """, unsafe_allow_html=True)
                                
                                for product in top_products:
                                    st.markdown(f"<li><b>{product}</b> - Top performer with strong market demand</li>", unsafe_allow_html=True)
                                
                                st.markdown("""
                                    </ul>
                                    <p>Consider increasing inventory levels for these products to avoid stockouts.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Territory recommendations
                            if 'RSM' in df.columns:
                                st.markdown("""
                                <div class="highlight">
                                    <h4>🌍 Territory Optimization</h4>
                                    <p>Based on territory performance analysis:</p>
                                    <ul>
                                        <li>Increase visits to <b>underperforming regions</b> by 20%</li>
                                        <li>Reallocate resources from saturated markets to high-growth areas</li>
                                        <li>Consider partnerships with local distributors in untapped territories</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Training recommendations
                            st.markdown("""
                            <div class="highlight">
                                <h4>🎓 Training Opportunities</h4>
                                <p>Recommended training programs for your team:</p>
                                <ul>
                                    <li><b>Advanced Product Knowledge</b> - Deep dive into mechanism of action</li>
                                    <li><b>Consultative Selling Techniques</b> - Focus on customer needs</li>
                                    <li><b>Time Management</b> - Optimizing field visits and admin work</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    elif rec_type == "For Product":
                        selected_product = st.selectbox(
                            "Select Product",
                            df['Pname'].unique(),
                            key="rec_product"
                        )
                        
                        if st.button("Generate Recommendations", key="product_rec"):
                            st.subheader(f"Recommendations for {selected_product}")
                            
                            # Product performance analysis
                            product_df = df[df['Pname'] == selected_product]
                            
                            # Regional performance
                            if 'RSM' in df.columns and 'SoldQnty' in df.columns:
                                top_regions = product_df.groupby('RSM')['SoldQnty'].sum().nlargest(3).index.tolist()
                                bottom_regions = product_df.groupby('RSM')['SoldQnty'].sum().nsmallest(3).index.tolist()
                                
                                st.markdown("""
                                <div class="highlight">
                                    <h4>📍 Regional Strategy</h4>
                                    <p>Your product performs well in these regions:</p>
                                    <ul>
                                """, unsafe_allow_html=True)
                                
                                for region in top_regions:
                                    st.markdown(f"<li><b>{region}</b> - Consider increasing marketing budget</li>", unsafe_allow_html=True)
                                
                                st.markdown("""
                                    </ul>
                                    <p>Opportunities in these underperforming regions:</p>
                                    <ul>
                                """, unsafe_allow_html=True)
                                
                                for region in bottom_regions:
                                    st.markdown(f"<li><b>{region}</b> - Investigate market barriers</li>", unsafe_allow_html=True)
                                
                                st.markdown("</ul></div>", unsafe_allow_html=True)
                            
                            # Competitive positioning
                            st.markdown("""
                            <div class="highlight">
                                <h4>🆚 Competitive Positioning</h4>
                                <ul>
                                    <li>Highlight <b>unique selling points</b> compared to alternatives</li>
                                    <li>Develop <b>comparative materials</b> for sales team</li>
                                    <li>Monitor competitor pricing and adjust strategy accordingly</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Promotional strategy
                            st.markdown("""
                            <div class="highlight">
                                <h4>📢 Promotional Strategy</h4>
                                <ul>
                                    <li>Plan <b>seasonal promotions</b> aligned with demand patterns</li>
                                    <li>Consider <b>bundling</b> with complementary products</li>
                                    <li>Develop <b>targeted campaigns</b> for key customer segments</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    else:  # For Territory
                        selected_territory = st.selectbox(
                            "Select Territory",
                            df['RSM'].unique(),
                            key="rec_territory"
                        )
                        
                        if st.button("Generate Recommendations", key="territory_rec"):
                            st.subheader(f"Recommendations for {selected_territory}")
                            
                            # Territory performance analysis
                            territory_df = df[df['RSM'] == selected_territory]
                            
                            # Market penetration
                            st.markdown("""
                            <div class="highlight">
                                <h4>📈 Market Penetration</h4>
                                <ul>
                                    <li>Focus on <b>key accounts</b> with highest potential</li>
                                    <li>Develop <b>account-specific strategies</b> for top 10 customers</li>
                                    <li>Identify and target <b>white space opportunities</b></li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Product mix optimization
                            if 'Pname' in df.columns and 'SoldQnty' in df.columns:
                                top_products = territory_df.groupby('Pname')['SoldQnty'].sum().nlargest(3).index.tolist()
                                growth_products = territory_df.groupby('Pname')['SoldQnty'].sum().sort_values(ascending=False).head(10).index.tolist()
                                
                                st.markdown("""
                                <div class="highlight">
                                    <h4>💊 Product Mix Optimization</h4>
                                    <p>Maximize performance of your top products:</p>
                                    <ul>
                                """, unsafe_allow_html=True)
                                
                                for product in top_products:
                                    st.markdown(f"<li><b>{product}</b> - Continue current successful strategies</li>", unsafe_allow_html=True)
                                
                                st.markdown("""
                                    </ul>
                                    <p>Focus on growth products with potential:</p>
                                    <ul>
                                """, unsafe_allow_html=True)
                                
                                for product in growth_products[3:6]:
                                    st.markdown(f"<li><b>{product}</b> - Increase promotion and detailing</li>", unsafe_allow_html=True)
                                
                                st.markdown("</ul></div>", unsafe_allow_html=True)
                            
                            # Resource allocation
                            st.markdown("""
                            <div class="highlight">
                                <h4>🔄 Resource Allocation</h4>
                                <ul>
                                    <li>Optimize <b>sales rep territories</b> based on potential</li>
                                    <li>Reallocate <b>marketing budget</b> to high-ROI activities</li>
                                    <li>Adjust <b>inventory levels</b> based on demand patterns</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No recommendation types available in the data")
        
        # Report Generator Page
        elif app_mode == "📑 Report Generator":
            st.title("Custom Report Generator")
            st.markdown("Create customized performance reports for teams, products, or time periods.")
            
            # Report configuration
            with st.expander("Report Configuration", expanded=True):
                report_type = st.selectbox(
                    "Report Type",
                    ["Monthly Performance", "Product Analysis", "Team Evaluation", "Custom"],
                    key="report_type"
                )
                
                if report_type == "Monthly Performance":
                    if 'YearMonth' in df.columns:
                        months = df['YearMonth'].dt.strftime('%b %Y').unique()
                        selected_month = st.selectbox(
                            "Select Month",
                            sorted(months),
                            key="report_month"
                        )
                    
                    level_options = []
                    if 'RSM' in df.columns:
                        level_options.append('Region (RSM)')
                    if 'FM' in df.columns:
                        level_options.append('Field Manager (FM)')
                    if 'Team' in df.columns:
                        level_options.append('Team')
                    
                    if level_options:
                        level = st.selectbox(
                            "Aggregation Level",
                            level_options,
                            key="report_level"
                        )
                
                elif report_type == "Product Analysis":
                    if 'Pname' in df.columns:
                        selected_products = st.multiselect(
                            "Select Products",
                            df['Pname'].unique(),
                            default=df['Pname'].unique()[:3],
                            key="report_products"
                        )
                    
                    metric_options = []
                    if 'SoldQnty' in df.columns:
                        metric_options.append('Quantity Sold')
                    if 'SoldTP' in df.columns:
                        metric_options.append('Sales Value')
                    if 'QtyAchievementRate' in df.columns:
                        metric_options.append('Achievement Rate')
                    
                    if metric_options:
                        primary_metric = st.selectbox(
                            "Primary Metric",
                            metric_options,
                            key="report_metric"
                        )
                
                elif report_type == "Team Evaluation":
                    if 'Team' in df.columns:
                        selected_teams = st.multiselect(
                            "Select Teams",
                            df['Team'].unique(),
                            default=df['Team'].unique()[:3],
                            key="report_teams"
                        )
                    
                    time_period = st.selectbox(
                        "Time Period",
                        ['Last Month', 'Last Quarter', 'Last 6 Months', 'Year-to-Date'],
                        key="report_time"
                    )
                
                else:  # Custom report
                    # Metric selection
                    metric_options = []
                    if 'SoldQnty' in df.columns:
                        metric_options.append('Quantity Sold')
                    if 'SoldTP' in df.columns:
                        metric_options.append('Sales Value')
                    if 'TargetQnty' in df.columns:
                        metric_options.append('Target Quantity')
                    if 'TargetValue' in df.columns:
                        metric_options.append('Target Value')
                    if 'QtyAchievementRate' in df.columns:
                        metric_options.append('Achievement Rate')
                    
                    selected_metrics = st.multiselect(
                        "Select Metrics",
                        metric_options,
                        default=metric_options[:3],
                        key="custom_metrics"
                    )
                    
                    # Filter options
                    filter_options = []
                    if 'RSM' in df.columns:
                        filter_options.append('Region')
                    if 'FM' in df.columns:
                        filter_options.append('Field Manager')
                    if 'Team' in df.columns:
                        filter_options.append('Team')
                    if 'Brand' in df.columns:
                        filter_options.append('Brand')
                    if 'Thgroup' in df.columns:
                        filter_options.append('Therapeutic Group')
                    
                    selected_filters = st.multiselect(
                        "Filter By",
                        filter_options,
                        key="custom_filters"
                    )
                    
                    # Date range
                    if 'YearMonth' in df.columns:
                        date_range = st.date_input(
                            "Date Range",
                            [df['YearMonth'].min(), df['YearMonth'].max()],
                            key="report_date_range"
                        )
            
            # Generate report
            if st.button("Generate Report", key="generate_report"):
                st.success("Report Generated Successfully!")
                
                # Report header
                st.markdown(f"""
                # Pharma Sales Performance Report
                **Report Type:** {report_type}  
                **Generated On:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
                """)
                
                # Executive summary
                st.markdown("""
                ## Executive Summary
                This report provides a comprehensive analysis of sales performance based on the selected parameters. 
                Key insights and recommendations are highlighted to support data-driven decision making.
                """)
                
                # Key metrics
                st.markdown("## Key Performance Indicators")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sales Value", "$1,250,000", "+12% vs previous period")
                with col2:
                    st.metric("Total Quantity Sold", "25,000 units", "+8% vs target")
                with col3:
                    st.metric("Average Achievement", "92%", "+5% vs previous period")
                
                # Performance analysis
                st.markdown("## Performance Analysis")
                
                # Sample charts (in a real app, these would be generated from actual data)
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.bar(
                        x=['Team A', 'Team B', 'Team C'],
                        y=[120000, 95000, 110000],
                        title="Sales by Team",
                        labels={'x': 'Team', 'y': 'Sales Value'},
                        color=['#0068c9', '#4CAF50', '#FFA500']
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = px.line(
                        x=['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                        y=[1000, 1200, 950, 1100, 1050],
                        title="Monthly Sales Trend",
                        labels={'x': 'Month', 'y': 'Sales Quantity'},
                        markers=True
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Detailed findings
                st.markdown("## Detailed Findings")
                st.markdown("""
                - **Top Performing Product:** Marvelon Tablets achieved 125% of target
                - **Growth Opportunity:** Adam 20 Tablets showing 15% month-over-month growth
                - **Area for Improvement:** Anaroxyl Injections underperforming in Western region
                """)
                
                # Recommendations
                st.markdown("## Recommendations")
                st.markdown("""
                1. **Increase promotion** of high-growth products in underpenetrated regions
                2. **Review inventory levels** for top performers to prevent stockouts
                3. **Conduct training** on consultative selling for underperforming teams
                4. **Adjust targets** based on market potential analysis
                """)
                
                # Download options
                st.markdown("---")
                st.markdown("### Download Report")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        "Download as PDF",
                        data="PDF content would be generated here",
                        file_name="sales_report.pdf",
                        mime="application/pdf"
                    )
                with col2:
                    st.download_button(
                        "Download as Excel",
                        data="Excel content would be generated here",
                        file_name="sales_report.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                with col3:
                    st.download_button(
                        "Download as PowerPoint",
                        data="PPT content would be generated here",
                        file_name="sales_report.pptx",
                        mime="application/vnd.ms-powerpoint"
                    )
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Pharma Sales Analytics** v1.0")
        st.sidebar.markdown(f"Data last updated: {datetime.now().strftime('%Y-%m-%d')}")


    else:
        st.error("Please load the uploaded file or check the file format and try again.")

# If no file is uploaded at all
if 'uploaded_file' not in locals() or uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis")
    st.markdown("""
    ### Expected Data Format:
    The CSV file should contain pharmaceutical sales data with columns similar to:
    - **Hierarchy:** RSM, FM, MPO, Team
    - **Employee:** Emp Code
    - **Product:** Pcode, Pname, Brand, Thgroup
    - **Performance Metrics:** TargetQnty, SoldQnty, TargetValue, SoldTP
    - **Date:** YearMonth (format like 'May-25' or '2025-05')
    
    Sample data:
    """)
    st.table(pd.DataFrame({
        "RSM": ["B1", "B1", "B2"],
        "FM": ["B10", "B10", "B20"],
        "Team": ["Nugenta", "Nugenta", "Alpha"],
        "Pname": ["MARVELON TAB", "OVOSTAT GOLD TAB", "ADAM 20 TAB"],
        "SoldQnty": [227, 418, 150],
        "SoldTP": [20606, 27821, 12000],
        "YearMonth": ["May-25", "May-25", "May-25"]
    }))

if __name__ == "__main__":
    main()