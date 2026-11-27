# ==========================================
# Amazon India Sales Analysis (Final Clean with Month-wise Line Graph)
# ==========================================

import pandas as pd
import plotly.express as px
import json, urllib.request
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------
df = pd.read_csv("amazon_india_salesl.csv")
df["Order_Month"] = df["Order_Month"].astype(int)
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# ------------------------------------------
# 2. City → State Buyers Analysis
# ------------------------------------------
CITY_TO_STATE = {
    "Mumbai":"Maharashtra","Pune":"Maharashtra","Nagpur":"Maharashtra",
    "Bengaluru":"Karnataka","Mysuru":"Karnataka",
    "Chennai":"Tamil Nadu","Coimbatore":"Tamil Nadu",
    "Hyderabad":"Telangana","Delhi":"Delhi","Kolkata":"West Bengal",
    "Ahmedabad":"Gujarat","Surat":"Gujarat",
    "Jaipur":"Rajasthan","Lucknow":"Uttar Pradesh","Kanpur":"Uttar Pradesh"
}

df["State_Corrected"] = df["State"].map(CITY_TO_STATE)
df = df.dropna(subset=["State_Corrected"])

state_buyers = (
    df.groupby("State_Corrected")["Customer_ID"]
    .nunique()
    .reset_index(name="Buyers")
)

print("\nState-wise Buyers:\n", state_buyers.sort_values("Buyers", ascending=False))

# India Choropleth Map
url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/india_states.geojson"
with urllib.request.urlopen(url) as r:
    india_states = json.load(r)

px.choropleth(
    state_buyers,
    geojson=india_states,
    featureidkey="properties.ST_NM",
    locations="State_Corrected",
    color="Buyers",
    title="State-wise Buyers Distribution (India)",
    color_continuous_scale="Viridis"
).update_geos(fitbounds="locations", visible=False).show()

# ------------------------------------------
# 3. Order Status Distribution
# ------------------------------------------
order_status = (
    df["Order_Status"]
    .value_counts()
    .reset_index(name="Orders")
    .rename(columns={"index":"Order_Status"})
)

px.pie(
    order_status,
    names="Order_Status",
    values="Orders",
    hole=0,
    title="Order Status Ratio"
).show()

# ------------------------------------------
# 4. Category-wise Sales (Donut)
# ------------------------------------------
category_sales = (
    df.groupby("Category")["Order_Amount_INR"]
    .sum()
    .reset_index()
)

px.pie(
    category_sales,
    names="Category",
    values="Order_Amount_INR",
    hole=0.5,
    title="Category-wise Sales Distribution"
).show()

# ------------------------------------------
# 5. Category-wise Profit by Gender
# ------------------------------------------
category_gender_profit = (
    df.groupby(["Category","Customer_Gender"])["Order_Amount_INR"]
    .sum()
    .reset_index(name="Profit")
)

px.bar(
    category_gender_profit,
    x="Category",
    y="Profit",
    color="Customer_Gender",
    barmode="group",
    title="Category-wise Profit by Gender"
).update_yaxes(tickprefix="₹").show()

# ------------------------------------------
# 6. Category-wise Repeat Orders
# ------------------------------------------
repeat_buyers = df[
    df["Repeat_Buyer_Flag"]
    .astype(str)
    .str.lower()
    .isin(["yes","y","true","1","repeat"])
]

repeat_orders = (
    repeat_buyers.groupby("Category")["Order_ID"]
    .count()
    .reset_index(name="Repeat_Order_Count")
)

px.scatter(
    repeat_orders,
    x="Category",
    y="Repeat_Order_Count",
    size="Repeat_Order_Count",
    title="Category-wise Repeat Orders"
).show()



# ------------------------------------------
# 7.product categry quantity analysis
# ------------------------------------------
product_category_quantity = (
    df.groupby("Category")["Quantity"]
    .sum()
    .reset_index(name="Total_Quantity")
)
px.bar(
    product_category_quantity,
    x="Category",
    y="Total_Quantity",
    text_auto=".2s",
    title="Product Category vs Total Quantity Sold"
).show()
# ------------------------------------------
#  Monthly Profit 
# ------------------------------------------
MONTH_MAP = {
    1:"January",2:"February",3:"March",4:"April",
    5:"May",6:"June",7:"July",8:"August",
    9:"September",10:"October",11:"November",12:"December"
}

df["Order_Month_Name"] = df["Order_Month"].map(MONTH_MAP)

monthly_profit = (
    df.groupby("Order_Month_Name")["Order_Amount_INR"]
    .sum()
    .reset_index(name="Profit")
)
# ------------------------------------------
# 8. Monthly Sales by Gender
# ------------------------------------------
monthly_gender_sales = (
    df.groupby(["Order_Month_Name","Customer_Gender"])["Order_Amount_INR"]
    .sum()
    .reset_index()
)

px.area(
    monthly_gender_sales,
    x="Order_Month_Name",
    y="Order_Amount_INR",
    color="Customer_Gender",
    title="Monthly Sales by Gender"
).update_yaxes(tickprefix="₹").show()

# ------------------------------------------
# 9. Category Sales Funnel
# ------------------------------------------
px.funnel(
    category_sales.sort_values("Order_Amount_INR", ascending=False),
    x="Order_Amount_INR",
    y="Category",
    title="Highest to Lowest Sales Category"
).show()

# ------------------------------------------
# 10. Sale Event Cohort Retention Analysis
# ------------------------------------------
first_purchase = (
    df.groupby("Customer_ID")["Order_Month"]
    .min()
    .reset_index(name="First_Month")
)

df = df.merge(first_purchase, on="Customer_ID")
df["Cohort_Month"] = df["Order_Month"] - df["First_Month"]

first_event = (
    df[df["Cohort_Month"] == 0][["Customer_ID","Sale_Event"]]
    .drop_duplicates()
    .rename(columns={"Sale_Event":"Sale_Event_Cohort"})
)

df = df.merge(first_event, on="Customer_ID")

cohort_data = (
    df.groupby(["Sale_Event_Cohort","Cohort_Month"])["Customer_ID"]
    .nunique()
    .reset_index(name="Customers")
)

cohort_pivot = cohort_data.pivot(
    index="Sale_Event_Cohort",
    columns="Cohort_Month",
    values="Customers"
)

retention = cohort_pivot.divide(cohort_pivot[0], axis=0) * 100

px.imshow(
    retention.round(1),
    text_auto=True,
    title="Sale Event Cohort Retention (%)",
    color_continuous_scale="Viridis"
).show()

# ------------------------------------------
# 11. Month-wise Sales Line Graph
# ------------------------------------------
monthly_sales = (
    df.groupby(df["Order_Date"].dt.to_period("M"))
      .agg({"Order_Amount_INR": "sum"})
      .reset_index()
)
monthly_sales["Order_Date"] = monthly_sales["Order_Date"].dt.to_timestamp()

px.line(
    monthly_sales,
    x="Order_Date",
    y="Order_Amount_INR",
    title="Month-wise Sales (INR)",
    labels={"Order_Date":"Month","Order_Amount_INR":"Total Sales (INR)"},
    markers=True
).update_layout(
    xaxis=dict(dtick="M1", tickformat="%b-%Y"),
    yaxis=dict(tickprefix="₹"),
    template="plotly_white"
).show()
