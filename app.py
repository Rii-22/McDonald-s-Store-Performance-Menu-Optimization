"""
McDonald's Store Performance & Menu Optimization Dashboard
Bridging Kitchen Efficiency with Customer Satisfaction
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import re
from datetime import datetime, timedelta

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="McDonald's Operations Dashboard",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GOLDEN ARCHES THEME
# ============================================================================

st.markdown("""
    <style>
    /* Main App Background - Dark Grey */
    .stApp {
        background-color: #292929;
        color: #FFFFFF;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 2px solid #FFC72C;
    }
    
    /* Headers - Golden Yellow */
    h1 {
        color: #FFC72C !important;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    h2, h3 {
        color: #FFC72C !important;
        font-weight: 700;
    }
    
    /* Metrics - Golden Yellow */
    [data-testid="stMetricValue"] {
        color: #FFC72C;
        font-size: 2.5rem;
        font-weight: 900;
    }
    
    [data-testid="stMetricLabel"] {
        color: #CCCCCC;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    [data-testid="stMetricDelta"] {
        color: #DA291C;
    }
    
    /* Buttons - McDonald's Red */
    .stButton>button {
        background-color: #DA291C;
        color: #FFFFFF;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 12px 32px;
        transition: all 0.3s;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        background-color: #FF3D3D;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(218, 41, 28, 0.4);
    }
    
    /* Sliders - Golden Yellow */
    .stSlider>div>div>div>div {
        background-color: #FFC72C;
    }
    
    /* Text */
    p, label, .stMarkdown {
        color: #CCCCCC;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a1a;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #3a3a3a;
        color: #CCCCCC;
        border-radius: 8px;
        padding: 10px 20px;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #DA291C;
        color: #FFFFFF;
        border: 2px solid #FFC72C;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #3a3a3a !important;
        color: #FFFFFF !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #3a3a3a;
        color: #FFC72C;
        border-radius: 8px;
        border-left: 4px solid #DA291C;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess {
        background-color: #3a3a3a;
        color: #FFFFFF;
        border-left: 4px solid #FFC72C;
    }
    
    .stInfo {
        background-color: #3a3a3a;
        color: #FFFFFF;
        border-left: 4px solid #DA291C;
    }
    
    .stWarning {
        background-color: #3a3a3a;
        color: #FFFFFF;
        border-left: 4px solid #FF8C00;
    }
    
    /* Divider */
    hr {
        border-color: #FFC72C;
        opacity: 0.3;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA GENERATION ENGINE
# ============================================================================

@st.cache_data
def generate_mcdonalds_orders(n_orders=8000, seed=42):
    """
    Generate synthetic McDonald's order data with realistic operational patterns.
    
    Returns:
        pd.DataFrame: Order dataset with menu items, timing, and satisfaction metrics
    """
    np.random.seed(seed)
    
    # === ORDER DETAILS ===
    order_ids = [f"ORD_{str(i).zfill(6)}" for i in range(1, n_orders + 1)]
    
    # Timestamps across a typical day (6 AM - 11 PM)
    start_time = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
    
    # Weighted distribution for realistic traffic patterns
    hour_weights = [
        0.02, 0.03, 0.08, 0.12, 0.10, 0.08,  # 6-11 AM (breakfast)
        0.15, 0.18, 0.08, 0.05, 0.03, 0.02,  # 12-5 PM (lunch peak)
        0.04, 0.02                            # 6-7 PM (dinner)
    ]
    
    timestamps = []
    for _ in range(n_orders):
        hour = np.random.choice(range(6, 6+len(hour_weights)), p=hour_weights)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        timestamp = start_time.replace(hour=hour, minute=minute, second=second)
        timestamps.append(timestamp)
    
    timestamps = sorted(timestamps)
    
    # Peak hour flag (12 PM - 2 PM)
    is_peak_hour = [(12 <= t.hour < 14) for t in timestamps]
    
    # === ORDER CHANNELS ===
    # Drive-Thru dominates (55%), followed by Kiosk (20%), Counter (15%), Delivery (10%)
    channels = np.random.choice(
        ['Drive-Thru', 'Kiosk', 'Delivery', 'Counter'],
        n_orders,
        p=[0.55, 0.20, 0.10, 0.15]
    )
    
    # === MENU ITEMS ===
    # Primary items with realistic distribution
    primary_items = [
        'Big Mac',
        'Quarter Pounder with Cheese',
        'McChicken',
        'Filet-O-Fish',
        'Cheeseburger',
        'Chicken McNuggets (10pc)',
        'McDouble',
        'Premium Crispy Chicken Deluxe',
        'Egg McMuffin',
        'Bacon Egg & Cheese Biscuit'
    ]
    
    primary_weights = [0.20, 0.15, 0.12, 0.08, 0.10, 0.13, 0.08, 0.06, 0.05, 0.03]
    primary_item = np.random.choice(primary_items, n_orders, p=primary_weights)
    
    # Add-on items (70% of orders have add-ons)
    addon_items = ['Fries (Medium)', 'Coca-Cola (Medium)', 'McFlurry (Oreo)', 'Apple Pie', 'None']
    addon_weights = [0.35, 0.25, 0.10, 0.05, 0.25]
    addon_item = np.random.choice(addon_items, n_orders, p=addon_weights)
    
    # === STORE CONTEXT ===
    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
    store_regions = np.random.choice(regions, n_orders, p=[0.22, 0.20, 0.18, 0.18, 0.22])
    
    # === ORDER VALUE ===
    # Base prices with variation
    base_prices = {
        'Big Mac': 5.99,
        'Quarter Pounder with Cheese': 6.49,
        'McChicken': 4.29,
        'Filet-O-Fish': 5.49,
        'Cheeseburger': 2.99,
        'Chicken McNuggets (10pc)': 6.99,
        'McDouble': 3.99,
        'Premium Crispy Chicken Deluxe': 6.99,
        'Egg McMuffin': 4.49,
        'Bacon Egg & Cheese Biscuit': 4.99
    }
    
    addon_prices = {
        'Fries (Medium)': 2.79,
        'Coca-Cola (Medium)': 1.99,
        'McFlurry (Oreo)': 3.99,
        'Apple Pie': 1.49,
        'None': 0.00
    }
    
    order_values = []
    for i in range(n_orders):
        base = base_prices[primary_item[i]]
        addon = addon_prices[addon_item[i]]
        # Add small random variation (+/- 10%)
        value = (base + addon) * np.random.uniform(0.95, 1.05)
        order_values.append(round(value, 2))
    
    # === PREP TIME ===
    # Base prep time depends on item complexity and channel
    base_prep_times = {
        'Big Mac': 180,
        'Quarter Pounder with Cheese': 200,
        'McChicken': 150,
        'Filet-O-Fish': 210,
        'Cheeseburger': 120,
        'Chicken McNuggets (10pc)': 140,
        'McDouble': 130,
        'Premium Crispy Chicken Deluxe': 220,
        'Egg McMuffin': 110,
        'Bacon Egg & Cheese Biscuit': 125
    }
    
    # Channel multipliers
    channel_multipliers = {
        'Drive-Thru': 1.2,   # Slower due to communication
        'Kiosk': 1.0,        # Baseline
        'Delivery': 1.4,     # Extra packaging
        'Counter': 0.95      # Slightly faster
    }
    
    prep_times = []
    for i in range(n_orders):
        base_time = base_prep_times[primary_item[i]]
        
        # Add addon time
        if addon_item[i] != 'None':
            base_time += 30
        
        # Apply channel multiplier
        base_time *= channel_multipliers[channels[i]]
        
        # Peak hour adds congestion (15-30% slower)
        if is_peak_hour[i]:
            base_time *= np.random.uniform(1.15, 1.30)
        
        # Add random variation
        final_time = base_time * np.random.uniform(0.85, 1.15)
        prep_times.append(int(final_time))
    
    # === CUSTOMER RATING ===
    # Rating inversely correlated with prep time
    # Breaking point: ~240 seconds (4 minutes)
    
    customer_ratings = []
    for prep_time in prep_times:
        # Base rating logic: faster = better
        if prep_time < 150:
            base_rating = 5.0
        elif prep_time < 210:
            base_rating = 4.5
        elif prep_time < 270:
            base_rating = 4.0
        elif prep_time < 330:
            base_rating = 3.5
        else:
            base_rating = 3.0
        
        # Add random variation (+/- 0.5 stars)
        rating = base_rating + np.random.uniform(-0.5, 0.5)
        rating = max(1.0, min(5.0, rating))  # Clamp to 1-5
        
        # Round to nearest 0.5
        rating = round(rating * 2) / 2
        customer_ratings.append(rating)
    
    # === CREATE DATAFRAME ===
    df = pd.DataFrame({
        'Order_ID': order_ids,
        'Timestamp': timestamps,
        'Channel': channels,
        'Primary_Item': primary_item,
        'Add_On_Item': addon_item,
        'Order_Value_USD': order_values,
        'Prep_Time_Secs': prep_times,
        'Customer_Rating': customer_ratings,
        'Store_Region': store_regions,
        'Is_Peak_Hour': is_peak_hour
    })
    
    return df

# ============================================================================
# ANALYTICAL FUNCTIONS
# ============================================================================

@st.cache_data
def analyze_throughput_friction(df):
    """
    THE THROUGHPUT FRICTION ANALYSIS
    
    Correlates Prep_Time_Secs with Customer_Rating to find the "Breaking Point"
    where ratings drop from 5-stars to 3-stars.
    """
    # Calculate correlation
    correlation, p_value = stats.pearsonr(df['Prep_Time_Secs'], df['Customer_Rating'])
    
    # Find rating thresholds
    rating_5_times = df[df['Customer_Rating'] == 5.0]['Prep_Time_Secs']
    rating_4_times = df[df['Customer_Rating'] == 4.0]['Prep_Time_Secs']
    rating_3_times = df[df['Customer_Rating'] == 3.0]['Prep_Time_Secs']
    
    # Breaking point = average prep time where ratings drop to 3.0
    breaking_point = rating_3_times.mean() if len(rating_3_times) > 0 else df['Prep_Time_Secs'].median()
    
    # Optimal zone = average prep time for 5.0 ratings
    optimal_time = rating_5_times.mean() if len(rating_5_times) > 0 else df['Prep_Time_Secs'].min()
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'breaking_point_secs': breaking_point,
        'optimal_time_secs': optimal_time,
        'rating_5_avg_time': rating_5_times.mean() if len(rating_5_times) > 0 else 0,
        'rating_4_avg_time': rating_4_times.mean() if len(rating_4_times) > 0 else 0,
        'rating_3_avg_time': rating_3_times.mean() if len(rating_3_times) > 0 else 0
    }

@st.cache_data
def categorize_menu_items(df):
    """
    REGEX MENU CATEGORIZATION
    
    Groups items into "Value Menu" vs "Premium Collection" and analyzes
    attachment rates (probability of add-on items).
    """
    def categorize_item(item_name):
        # Value Menu: Contains "Mc", "Cheese", "Chicken McNuggets", or specific budget items
        value_patterns = r'(Mc|Cheeseburger|McNuggets)'
        
        # Premium: Contains "Premium", "Quarter Pounder", "Deluxe", "Filet"
        premium_patterns = r'(Premium|Quarter Pounder|Deluxe|Filet)'
        
        if re.search(premium_patterns, item_name, re.IGNORECASE):
            return 'Premium Collection'
        elif re.search(value_patterns, item_name, re.IGNORECASE):
            return 'Value Menu'
        else:
            return 'Core Menu'
    
    df['Menu_Category'] = df['Primary_Item'].apply(categorize_item)
    
    # Calculate attachment rates (% of orders with add-ons)
    category_stats = df.groupby('Menu_Category').agg({
        'Order_ID': 'count',
        'Add_On_Item': lambda x: (x != 'None').sum(),
        'Order_Value_USD': 'mean'
    }).reset_index()
    
    category_stats.columns = ['Menu_Category', 'Total_Orders', 'Orders_With_Addons', 'Avg_Order_Value']
    category_stats['Attachment_Rate_%'] = (category_stats['Orders_With_Addons'] / category_stats['Total_Orders'] * 100).round(2)
    
    return df, category_stats

@st.cache_data
def detect_peak_hour_anomalies(df):
    """
    PEAK HOUR ANOMALY DETECTION
    
    Uses Z-score to identify prep-time outliers during lunch rush (12-2 PM)
    indicating kitchen bottlenecks.
    """
    peak_orders = df[df['Is_Peak_Hour'] == True].copy()
    
    if len(peak_orders) == 0:
        return peak_orders, []
    
    # Calculate Z-scores for prep times
    z_scores = np.abs(stats.zscore(peak_orders['Prep_Time_Secs']))
    peak_orders['Z_Score'] = z_scores
    
    # Anomalies: Z-score > 2.5
    anomalies = peak_orders[z_scores > 2.5].copy()
    
    return peak_orders, anomalies

@st.cache_data
def analyze_profitable_pairings(df):
    """
    BASKET ANALYSIS
    
    Identifies most profitable menu pairings (Primary + Add-On combinations).
    """
    # Filter out orders without add-ons
    paired_orders = df[df['Add_On_Item'] != 'None'].copy()
    
    # Create pairing column
    paired_orders['Pairing'] = paired_orders['Primary_Item'] + ' + ' + paired_orders['Add_On_Item']
    
    # Analyze pairings
    pairing_stats = paired_orders.groupby('Pairing').agg({
        'Order_ID': 'count',
        'Order_Value_USD': 'mean',
        'Customer_Rating': 'mean',
        'Prep_Time_Secs': 'mean'
    }).reset_index()
    
    pairing_stats.columns = ['Pairing', 'Frequency', 'Avg_Value', 'Avg_Rating', 'Avg_Prep_Time']
    
    # Calculate profitability score (Value * Frequency / Prep_Time)
    pairing_stats['Profitability_Score'] = (
        pairing_stats['Avg_Value'] * pairing_stats['Frequency'] / (pairing_stats['Avg_Prep_Time'] / 100)
    ).round(2)
    
    pairing_stats = pairing_stats.sort_values('Profitability_Score', ascending=False)
    
    return pairing_stats

@st.cache_data
def simulate_staffing_impact(df, staffing_increase_pct):
    """
    STAFFING SIMULATION
    
    Hypothetically reduces prep time based on staffing increase and predicts
    impact on customer ratings.
    """
    df_sim = df.copy()
    
    # Staffing increase reduces prep time (diminishing returns)
    # Formula: New_Time = Old_Time * (1 - (increase% * 0.7))
    reduction_factor = 1 - (staffing_increase_pct / 100 * 0.7)
    df_sim['Prep_Time_Secs_New'] = (df_sim['Prep_Time_Secs'] * reduction_factor).astype(int)
    
    # Predict new ratings based on new prep times
    def predict_rating(prep_time):
        if prep_time < 150:
            return 5.0
        elif prep_time < 210:
            return 4.5
        elif prep_time < 270:
            return 4.0
        elif prep_time < 330:
            return 3.5
        else:
            return 3.0
    
    df_sim['Customer_Rating_New'] = df_sim['Prep_Time_Secs_New'].apply(predict_rating)
    
    # Calculate improvements
    avg_rating_improvement = df_sim['Customer_Rating_New'].mean() - df_sim['Customer_Rating'].mean()
    avg_time_reduction = df_sim['Prep_Time_Secs'].mean() - df_sim['Prep_Time_Secs_New'].mean()
    
    return {
        'original_avg_rating': df['Customer_Rating'].mean(),
        'new_avg_rating': df_sim['Customer_Rating_New'].mean(),
        'rating_improvement': avg_rating_improvement,
        'original_avg_time': df['Prep_Time_Secs'].mean(),
        'new_avg_time': df_sim['Prep_Time_Secs_New'].mean(),
        'time_reduction_secs': avg_time_reduction
    }

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    
    # === HEADER ===
    st.markdown("""
        <h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0;'>
            🍔 McDONALD'S OPERATIONS COMMAND CENTER
        </h1>
        <p style='text-align: center; color: #DA291C; font-size: 1.4rem; margin-top: 5px; font-weight: bold;'>
            Kitchen Efficiency × Customer Satisfaction = Golden Arches Excellence
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === GENERATE DATA ===
    with st.spinner('🍟 Generating order data from stores nationwide...'):
        df = generate_mcdonalds_orders(n_orders=8000)
        df, menu_category_stats = categorize_menu_items(df)
        peak_orders, anomalies = detect_peak_hour_anomalies(df)
        pairing_stats = analyze_profitable_pairings(df)
        friction_analysis = analyze_throughput_friction(df)
    
    # === SIDEBAR CONTROLS ===
    st.sidebar.title("🎛️ Store Manager Controls")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("👥 Staffing Simulation")
    st.sidebar.caption("Adjust staffing levels to see predicted impact on operations")
    
    staffing_increase = st.sidebar.slider(
        "Increase Staffing Level (%)",
        min_value=0,
        max_value=50,
        value=0,
        step=5,
        help="Simulate adding more kitchen staff during peak hours"
    )
    
    if staffing_increase > 0:
        sim_results = simulate_staffing_impact(df, staffing_increase)
        
        st.sidebar.success(f"**Predicted Improvements:**")
        st.sidebar.metric(
            "Avg Customer Rating",
            f"{sim_results['new_avg_rating']:.2f} ⭐",
            delta=f"+{sim_results['rating_improvement']:.2f}"
        )
        st.sidebar.metric(
            "Avg Prep Time",
            f"{sim_results['new_avg_time']:.0f}s",
            delta=f"-{sim_results['time_reduction_secs']:.0f}s"
        )
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("🔍 Filters")
    
    selected_channels = st.sidebar.multiselect(
        "Order Channel",
        options=sorted(df['Channel'].unique()),
        default=sorted(df['Channel'].unique())
    )
    
    selected_regions = st.sidebar.multiselect(
        "Store Region",
        options=sorted(df['Store_Region'].unique()),
        default=sorted(df['Store_Region'].unique())
    )
    
    # Apply filters
    filtered_df = df[
        (df['Channel'].isin(selected_channels)) &
        (df['Store_Region'].isin(selected_regions))
    ]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 **Viewing:** {len(filtered_df):,} / {len(df):,} orders")
    
    # === STORE MANAGER KPI TILES ===
    st.subheader("📊 Store Manager KPI Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        drive_thru_orders = filtered_df[filtered_df['Channel'] == 'Drive-Thru']
        avg_drive_thru_time = drive_thru_orders['Prep_Time_Secs'].mean()
        
        st.metric(
            "Avg Drive-Thru Time",
            f"{avg_drive_thru_time:.0f}s",
            delta=f"{avg_drive_thru_time - 180:.0f}s vs target (180s)",
            delta_color="inverse"
        )
    
    with col2:
        peak_sales = filtered_df[filtered_df['Is_Peak_Hour'] == True]['Order_Value_USD'].sum()
        
        st.metric(
            "Peak Hour Sales",
            f"${peak_sales:,.0f}",
            delta=f"{len(filtered_df[filtered_df['Is_Peak_Hour'] == True]):,} orders"
        )
    
    with col3:
        total_orders = len(filtered_df)
        orders_with_addons = len(filtered_df[filtered_df['Add_On_Item'] != 'None'])
        upsell_rate = (orders_with_addons / total_orders * 100) if total_orders > 0 else 0
        
        st.metric(
            "Menu Upsell Rate",
            f"{upsell_rate:.1f}%",
            delta=f"{upsell_rate - 70:.1f}% vs target (70%)",
            delta_color="normal"
        )
    
    with col4:
        avg_rating = filtered_df['Customer_Rating'].mean()
        
        st.metric(
            "Avg Customer Rating",
            f"{avg_rating:.2f} ⭐",
            delta=f"{avg_rating - 4.5:.2f} vs target (4.5)",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    # === THROUGHPUT FRICTION ANALYSIS ===
    st.subheader("⚡ Throughput Friction Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        **The Breaking Point: Where Speed Meets Satisfaction**
        
        **Statistical Correlation:**
        - **Pearson Correlation:** {friction_analysis['correlation']:.4f}
        - **P-Value:** {friction_analysis['p_value']:.6f}
        - **Significance:** {"✅ Highly Significant" if friction_analysis['p_value'] < 0.001 else "⚠️ Significant" if friction_analysis['p_value'] < 0.05 else "❌ Not Significant"}
        
        **Critical Time Thresholds:**
        - **⭐⭐⭐⭐⭐ (5-Star Zone):** {friction_analysis['rating_5_avg_time']:.0f} seconds
        - **⭐⭐⭐⭐ (4-Star Zone):** {friction_analysis['rating_4_avg_time']:.0f} seconds
        - **⭐⭐⭐ (3-Star Breaking Point):** {friction_analysis['rating_3_avg_time']:.0f} seconds
        
        **⚠️ The Golden Rule: Keep prep time under {friction_analysis['breaking_point_secs']:.0f} seconds to maintain customer satisfaction!**
        """)
    
    with col2:
        # Rating distribution
        rating_dist = filtered_df['Customer_Rating'].value_counts().sort_index()
        st.bar_chart(rating_dist)
        st.caption("Customer Rating Distribution")
    
    st.markdown("---")
    
    # === CHANNEL ANALYSIS ===
    st.subheader("📈 Sales Performance by Channel")
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Volume Analysis",
        "⏱️ Prep Time Trends",
        "💰 Value vs Speed"
    ])
    
    with tab1:
        st.markdown("**Order Volume by Channel** (Proving Drive-Thru Dominance)")
        
        channel_volume = filtered_df['Channel'].value_counts().sort_values(ascending=False)
        st.bar_chart(channel_volume)
        
        # Table
        channel_stats = filtered_df.groupby('Channel').agg({
            'Order_ID': 'count',
            'Order_Value_USD': 'mean',
            'Customer_Rating': 'mean',
            'Prep_Time_Secs': 'mean'
        }).reset_index()
        
        channel_stats.columns = ['Channel', 'Total_Orders', 'Avg_Order_Value', 'Avg_Rating', 'Avg_Prep_Time']
        channel_stats = channel_stats.sort_values('Total_Orders', ascending=False)
        
        st.dataframe(channel_stats, hide_index=True, use_container_width=True)
        
        # Insight
        top_channel = channel_stats.iloc[0]
        st.success(f"🚗 **{top_channel['Channel']}** dominates with {top_channel['Total_Orders']:,} orders ({(top_channel['Total_Orders']/len(filtered_df)*100):.1f}% of total volume)")
    
    with tab2:
        st.markdown("**Average Prep Time Throughout the Day**")
        
        # Group by hour
        filtered_df['Hour'] = pd.to_datetime(filtered_df['Timestamp']).dt.hour
        hourly_prep = filtered_df.groupby('Hour')['Prep_Time_Secs'].mean().reset_index()
        hourly_prep.columns = ['Hour', 'Avg_Prep_Time']
        
        st.line_chart(hourly_prep.set_index('Hour'))
        
        # Peak analysis
        peak_hour_avg = filtered_df[filtered_df['Is_Peak_Hour'] == True]['Prep_Time_Secs'].mean()
        non_peak_avg = filtered_df[filtered_df['Is_Peak_Hour'] == False]['Prep_Time_Secs'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Peak Hour Avg (12-2 PM)", f"{peak_hour_avg:.0f}s")
        with col2:
            st.metric("Non-Peak Avg", f"{non_peak_avg:.0f}s", delta=f"{peak_hour_avg - non_peak_avg:.0f}s slower", delta_color="inverse")
    
    with tab3:
        st.markdown("**Order Value vs. Prep Time** (Colored by Channel)")
        
        # Create scatter data
        scatter_data = filtered_df[['Order_Value_USD', 'Prep_Time_Secs', 'Channel']].copy()
        
        # Pivot for multi-series scatter
        channels_for_scatter = scatter_data['Channel'].unique()
        
        # Simple scatter representation
        st.scatter_chart(
            scatter_data.set_index('Order_Value_USD')['Prep_Time_Secs']
        )
        
        st.caption("Higher order values correlate with longer prep times, but channel efficiency varies")
    
    st.markdown("---")
    
    # === MENU CATEGORY ANALYSIS ===
    st.subheader("🍔 Menu Category Performance")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Attachment Rate by Menu Category**")
        st.dataframe(menu_category_stats, hide_index=True, use_container_width=True)
        
        # Insight
        best_category = menu_category_stats.loc[menu_category_stats['Attachment_Rate_%'].idxmax()]
        st.success(f"🎯 **{best_category['Menu_Category']}** shows highest upsell potential at {best_category['Attachment_Rate_%']:.1f}% attachment rate")
    
    with col2:
        st.markdown("**Category Breakdown**")
        category_chart = menu_category_stats.set_index('Menu_Category')['Attachment_Rate_%']
        st.bar_chart(category_chart)
    
    st.markdown("---")
    
    # === PROFITABLE PAIRINGS ===
    st.subheader("💎 Most Profitable Menu Pairings")
    
    top_pairings = pairing_stats.head(10)
    
    st.markdown("**Top 10 High-Value Combinations**")
    st.dataframe(
        top_pairings[['Pairing', 'Frequency', 'Avg_Value', 'Avg_Rating', 'Profitability_Score']],
        hide_index=True,
        use_container_width=True
    )
    
    # Recommendation
    best_pairing = top_pairings.iloc[0]
    st.info(f"""
    🏆 **Recommended Bundle Promotion:** {best_pairing['Pairing']}
    - Ordered {best_pairing['Frequency']:,} times
    - Average value: ${best_pairing['Avg_Value']:.2f}
    - Customer rating: {best_pairing['Avg_Rating']:.2f} ⭐
    - Profitability score: {best_pairing['Profitability_Score']:.0f}
    """)
    
    st.markdown("---")
    
    # === PEAK HOUR ANOMALIES ===
    st.subheader("🚨 Peak Hour Kitchen Bottleneck Detection")
    
    if len(anomalies) > 0:
        st.warning(f"⚠️ **{len(anomalies)} anomalous orders detected** during peak hours (12 PM - 2 PM)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Slowest Peak Hour Orders** (Z-Score > 2.5)")
            anomaly_display = anomalies.nlargest(10, 'Prep_Time_Secs')[
                ['Order_ID', 'Timestamp', 'Primary_Item', 'Prep_Time_Secs', 'Z_Score', 'Customer_Rating']
            ]
            st.dataframe(anomaly_display, hide_index=True, use_container_width=True)
        
        with col2:
            st.metric("Avg Anomaly Prep Time", f"{anomalies['Prep_Time_Secs'].mean():.0f}s")
            st.metric("Avg Normal Peak Time", f"{peak_orders[peak_orders['Z_Score'] <= 2.5]['Prep_Time_Secs'].mean():.0f}s")
            st.metric("Difference", f"{anomalies['Prep_Time_Secs'].mean() - peak_orders['Prep_Time_Secs'].mean():.0f}s", delta_color="inverse")
        
        # Root cause analysis
        st.markdown("**Potential Bottleneck Causes:**")
        
        anomaly_items = anomalies['Primary_Item'].value_counts().head(3)
        anomaly_channels = anomalies['Channel'].value_counts().head(3)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Most Common Items in Anomalies:**")
            for item, count in anomaly_items.items():
                st.write(f"- {item}: {count} occurrences")
        
        with col2:
            st.write("**Channels Most Affected:**")
            for channel, count in anomaly_channels.items():
                st.write(f"- {channel}: {count} anomalies")
    else:
        st.success("✅ No significant bottlenecks detected during peak hours!")
    
    st.markdown("---")
    
    # === EXECUTIVE SUMMARY ===
    st.subheader("📋 Executive Summary & Action Plan")
    
    with st.expander("📊 **View Complete Analysis & Strategic Recommendations**", expanded=True):
        
        drive_thru_pct = len(df[df['Channel'] == 'Drive-Thru']) / len(df) * 100
        
        # Get simulation results for any staffing level (default to 0 if not set)
        if staffing_increase > 0:
            sim_results = simulate_staffing_impact(df, staffing_increase)
            staffing_text = f"{staffing_increase}% increase → {sim_results['time_reduction_secs']:.0f}s faster"
        else:
            staffing_text = "Configure sidebar slider to simulate staffing changes"
        
        st.markdown(f"""
        ### 🔍 Key Operational Findings
        
        #### 1. Throughput-Satisfaction Correlation
        - **Strong negative correlation** detected (r = {friction_analysis['correlation']:.4f}, p < 0.001)
        - **Breaking Point identified:** {friction_analysis['breaking_point_secs']:.0f} seconds
        - Orders exceeding this threshold experience **dramatic rating drops** (5.0 → 3.0 stars)
        - **Golden Rule:** Maintain prep times under {friction_analysis['optimal_time_secs']:.0f}s for optimal satisfaction
        
        #### 2. Channel Distribution Insights
        - **Drive-Thru dominates** with {drive_thru_pct:.1f}% of all orders
        - Drive-Thru average prep time: {drive_thru_orders['Prep_Time_Secs'].mean():.0f}s
        - Kiosk orders show **{((filtered_df[filtered_df['Channel']=='Kiosk']['Customer_Rating'].mean() / filtered_df['Customer_Rating'].mean() - 1) * 100):.1f}%** rating difference vs average
        - Delivery channel needs attention: longest prep times, packaging complexity
        
        #### 3. Menu Upsell Opportunities
        - Current upsell rate: **{upsell_rate:.1f}%**
        - {menu_category_stats.loc[menu_category_stats['Attachment_Rate_%'].idxmax(), 'Menu_Category']} category shows highest attachment potential
        - **Top pairing:** {best_pairing['Pairing']} (${best_pairing['Avg_Value']:.2f} avg value, {best_pairing['Frequency']:,} orders)
        - Premium items command {menu_category_stats[menu_category_stats['Menu_Category']=='Premium Collection']['Avg_Order_Value'].values[0] if len(menu_category_stats[menu_category_stats['Menu_Category']=='Premium Collection']) > 0 else 0:.1f}% price premium
        
        #### 4. Peak Hour Performance
        - **{len(anomalies)} bottleneck events** identified during lunch rush (12 PM - 2 PM)
        - Peak hour prep times **{((peak_hour_avg / non_peak_avg - 1) * 100):.1f}% slower** than off-peak
        - Kitchen congestion causes average {anomalies['Prep_Time_Secs'].mean() - peak_orders['Prep_Time_Secs'].mean() if len(anomalies) > 0 else 0:.0f}s delay in extreme cases
        
        ---
        
        ### 🎯 Strategic Action Plan
        
        #### **Immediate Actions (Week 1-2)**
        
        1. **Speed of Service Blitz**
           - Target: Reduce Drive-Thru time from {avg_drive_thru_time:.0f}s to 180s
           - Method: Staff retraining on order assembly efficiency
           - Expected impact: +0.3 star rating improvement
        
        2. **Peak Hour Staffing Optimization**
           - Add {len(anomalies) // 10 + 2} kitchen staff during 12-2 PM window
           - {staffing_text}
           - Expected: Eliminate {len(anomalies)} bottleneck events
        
        3. **Menu Upsell Campaign**
           - Promote {best_pairing['Pairing']} as "Manager's Special Combo"
           - Train Drive-Thru staff on suggestive selling
           - Target: Increase attachment rate from {upsell_rate:.1f}% to 75%
        
        #### **Short-Term Initiatives (Month 1-2)**
        
        1. **Channel-Specific Optimization**
           - **Drive-Thru:** Implement dual-lane ordering at high-volume stores
           - **Kiosk:** Add combo meal shortcuts to reduce decision time
           - **Delivery:** Pre-stage packaging materials during peak hours
        
        2. **Menu Engineering**
           - Analyze Premium Collection attachment rates ({menu_category_stats[menu_category_stats['Menu_Category']=='Premium Collection']['Attachment_Rate_%'].values[0] if len(menu_category_stats[menu_category_stats['Menu_Category']=='Premium Collection']) > 0 else 0:.1f}%)
           - Create "Value + Premium" hybrid bundles
           - Test dynamic pricing during off-peak hours
        
        3. **Kitchen Layout Redesign**
           - Focus on items causing bottlenecks: {', '.join(anomaly_items.index[:3].tolist()) if len(anomalies) > 0 else "No major issues"}
           - Implement "make-to-stock" for high-frequency items during predicted peaks
        
        #### **Long-Term Strategy (Quarter 1-2)**
        
        1. **Predictive Analytics Platform**
           - Build ML model to forecast hourly order volume
           - Dynamic staffing recommendations 2 hours in advance
           - Expected: 20% reduction in labor costs, 15% improvement in service speed
        
        2. **Customer Experience Optimization**
           - A/B test order confirmation displays (time estimates)
           - Implement real-time prep time tracking on kiosk screens
           - Loyalty program integration: bonus points for off-peak orders
        
        3. **Regional Performance Management**
           - Benchmark stores by region: {', '.join(selected_regions)}
           - Share best practices from top-performing locations
           - Monthly "Store of Excellence" recognition program
        
        ---
        
        ### 💰 Financial Impact Projection
        
        **Revenue Opportunities:**
        - Upsell rate improvement (70% → 75%): **+${((0.75 - upsell_rate/100) * len(df) * 2.50):,.0f}** monthly
        - Premium pairing promotion: **+${(best_pairing['Frequency'] * 1.50 * 4):,.0f}** monthly (4 weeks)
        - Peak hour throughput optimization: **+{int(len(anomalies) * 0.7 * 30)}** additional orders/month
        
        **Cost Savings:**
        - Labor optimization: **-${(len(df) / 8000 * 5000 * 0.20):,.0f}** monthly (20% efficiency gain)
        - Waste reduction (faster service): **-${(len(df) * 0.15):,.0f}** monthly
        
        **Total Projected Monthly Impact: +${((0.75 - upsell_rate/100) * len(df) * 2.50) + (best_pairing['Frequency'] * 1.50 * 4) + (len(df) / 8000 * 5000 * 0.20):,.0f}**
        
        ---
        
        ### 📊 Success Metrics & KPIs
        
        **Track Daily:**
        - Drive-Thru average time (target: <180s)
        - Peak hour prep time variance
        - Customer rating distribution
        
        **Track Weekly:**
        - Upsell attachment rate (target: >75%)
        - Channel mix and performance
        - Anomaly event count (target: <10/week)
        
        **Track Monthly:**
        - Revenue per order (RPO)
        - Labor cost as % of sales
        - Customer satisfaction index (CSAT)
        - Same-store sales growth
        
        **Track Quarterly:**
        - Market share by region
        - Employee turnover rate
        - Customer lifetime value (CLV)
        """)
    
    st.markdown("---")
    
    # === FOOTER ===
    st.markdown("""
        <p style='text-align: center; color: #888888; font-size: 0.9rem;'>
            Built with Streamlit • McDonald's Operations Analytics Platform • Powered by Data Science
        </p>
        <p style='text-align: center; color: #FFC72C; font-size: 0.8rem; font-weight: bold;'>
            🍔 I'm Lovin' It - Analytics Edition
        </p>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
