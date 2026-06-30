import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import io
import matplotlib.pyplot as plt

# --- Helper function for exporting graph ---
def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

# --- Page Config ---
st.set_page_config(page_title="Smart Sales Predictor", layout="wide")

# --- Global Black Theme Styling ---
st.markdown("""
<style>
.stApp, .main {background-color: #0d0d0d !important; color: #f5c518 !important;}
section[data-testid="stSidebar"] {background-color: #1a1a1a; color:#f5c518;}
div[data-testid="stFileUploader"] {
    background-color: #1a1a1a; border: 1px solid #f5c518; border-radius: 10px; padding: 10px;
}
label, .stSelectbox label, .stNumberInput label {color:#f5c518; font-weight:bold;}
div.stButton > button {
    background-color:#f5c518; color:#000; font-weight:bold; border-radius:10px; padding:0.6rem 1.2rem; font-size:1rem;
}
div.stButton > button:hover {background-color:#ffdb4d; color:#000;}
.stDataFrame {border:1px solid #f5c518 !important; border-radius:10px; overflow:hidden;}
[data-testid="stDataFrame"] div {color:#f5c518 !important; background-color:#0d0d0d !important;}
.stSuccess, .stInfo {border:1px solid #f5c518 !important; background-color:#1a1a1a !important; color:#f5c518 !important; border-radius:10px; font-size:1.1rem; padding:10px;}
.stPlotlyChart, .stPyplot {border:1px solid #f5c518; border-radius:10px; padding:10px; background-color:#0d0d0d;}
h1,h2,h3{color:#f5c518;}
.highlight-box {border: 2px solid #f5c518; border-radius:10px; padding:12px; margin:5px 0; background-color:#1a1a1a; color:#f5c518; font-size:1.1rem;}
div[data-baseweb="input"] input, div[data-baseweb="select"] select, div[data-baseweb="datepicker"] input {background-color:#1a1a1a !important; color:#f5c518 !important; border:1px solid #f5c518; border-radius:8px;}
div[data-baseweb="input"] input:focus, div[data-baseweb="select"] select:focus, div[data-baseweb="datepicker"] input:focus {border:2px solid #ffeb70 !important; outline:none;}
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("⚡ Smart Sales Predictor ")
st.markdown("Predict, visualize, and explore your product sales.")

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head())

    # --- Auto column detection ---
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    price_col = next((c for c in df.columns if any(x in c.lower() for x in ["price","amount","cost","rate"])), df.columns[1])
    sales_col = next((c for c in df.columns if any(x in c.lower() for x in ["sales","sold","quantity","revenue"])), df.columns[2])
    product_col = next((c for c in df.columns if any(x in c.lower() for x in ["product","item","name"])), df.columns[3])

    # --- Sidebar Selection ---
    st.sidebar.header("⚙️ Column Selection")
    date_col = st.sidebar.selectbox("Select Date Column", df.columns, index=df.columns.get_loc(date_col))
    price_col = st.sidebar.selectbox("Select Price Column", df.columns, index=df.columns.get_loc(price_col))
    sales_col = st.sidebar.selectbox("Select Sales Column", df.columns, index=df.columns.get_loc(sales_col))
    product_col = st.sidebar.selectbox("Select Product Column (optional)", df.columns, index=df.columns.get_loc(product_col))

    # --- Data Conversion ---
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df.dropna(subset=[date_col], inplace=True)
    df["Year"] = df[date_col].dt.year

    # --- Product Filter ---
    product_list = df[product_col].dropna().unique().tolist()
    selected_product = st.sidebar.selectbox("🏷️ Select Product", product_list)
    df = df[df[product_col] == selected_product]

    # --- Year Filter ---
    year_list = sorted(df["Year"].dropna().unique().tolist())
    selected_year = st.sidebar.selectbox("📆 Select Year", year_list)
    df = df[df["Year"] == selected_year]

    # --- Date Range Filter ---
    min_date, max_date = df[date_col].min(), df[date_col].max()
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    df_filtered = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

    if df_filtered.empty:
        st.warning("⚠️ No data for the selected filters.")
        st.stop()

    st.subheader(f"📊 Filtered Data ({len(df_filtered)} rows)")
    st.dataframe(df_filtered)

    # --- Train Model ---
    X = df_filtered[[price_col]]
    y = df_filtered[sales_col]
    model = LinearRegression()
    model.fit(X, y)
    df_filtered["Predicted_Sales"] = model.predict(df_filtered[[price_col]])

    # --- Animated Predicted Sales Line ---
    st.subheader("📈 Predicted Sales")
    x_labels = df_filtered[date_col].dt.strftime("%Y-%m-%d").tolist()
    predicted_vals = df_filtered["Predicted_Sales"].tolist()

    html = f"""
    <canvas id="c" width="1100" height="400"></canvas>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
    const ctx = document.getElementById('c').getContext('2d');
    const labels = {x_labels};
    const dataPoints = {predicted_vals};
    const totalPoints = dataPoints.length;
    const duration = 3000;
    const interval = duration / totalPoints;

    const dataset = {{
        label: 'Predicted Sales',
        data: [],
        borderColor: '#ffeb80',
        borderWidth: 2,
        fill: false,
        tension: 0.3,
        pointStyle: 'circle',
        pointRadius: 4
    }};

    const chart = new Chart(ctx, {{
        type: 'line',
        data: {{ labels: labels, datasets: [dataset] }},
        options: {{
            responsive: true,
            plugins: {{
                legend: {{ labels: {{ color:'#ffeb80' }} }}
            }},
            scales: {{
                x: {{ ticks: {{ color:'#ffeb80' }}, grid: {{ color:'#333' }} }},
                y: {{ ticks: {{ color:'#ffeb80' }}, grid: {{ color:'#333' }} }}
            }}
        }},
    }});

    ctx.canvas.parentNode.style.backgroundColor = '#0d0d0d';

    let currentIndex = 0;
    function addPoint() {{
        if(currentIndex < totalPoints){{
            dataset.data.push(dataPoints[currentIndex]);
            chart.update();
            currentIndex++;
            setTimeout(addPoint, interval);
        }}
    }}
    addPoint();
    </script>
    """
    st.components.v1.html(html, height=450)

    # --- Top Predicted Days (Next Year) ---
    top_predictions = df_filtered.sort_values(by="Predicted_Sales", ascending=False).head(5).copy()
    top_predictions[date_col] = top_predictions[date_col].apply(lambda d: d.replace(year=d.year + 1))

    st.subheader(f"🔥 Top 5 Predicted Sales Days for {selected_year + 1}")
    for _, row in top_predictions.iterrows():
        st.markdown(
            f"""
            <div class="highlight-box">
                📅 <b>Date:</b> {row[date_col].date()} &nbsp;&nbsp;
                💰 <b>Price:</b> ₹{row[price_col]:.2f} &nbsp;&nbsp;
                📈 <b>Predicted Sales:</b> {row['Predicted_Sales']:.2f} units
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Custom Prediction ---
    st.subheader("💡 Predict Sales for Custom Price")
    user_price = st.number_input("Enter a Price", min_value=0.0, step=0.1, value=float(df_filtered[price_col].mean()))
    if st.button("Predict Sales"):
        predicted_sales = model.predict([[user_price]])[0]
        st.markdown(
            f"""
            <div class="highlight-box">
                💰 Entered Price: ₹{user_price:.2f}<br>
                🔮 Predicted Sales: <b>{predicted_sales:.2f}</b> units
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- PNG download ---
    fig, ax = plt.subplots(figsize=(11,5))
    ax.set_facecolor("#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")
    ax.plot(df_filtered[date_col], predicted_vals, color="#ffeb80", linewidth=2)
    st.download_button(
        label="📥 Download Graph as PNG",
        data=fig_to_png(fig),
        file_name=f"predicted_sales_{selected_year}.png",
        mime="image/png",
    )

else:
    st.info("📂 Upload a dataset to get started.")
