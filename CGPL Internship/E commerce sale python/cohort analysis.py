import pandas as pd
import plotly.express as px
import json, urllib.request

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------
df = pd.read_csv("amazon_india_salesl.csv")
df["Order_Month"] = df["Order_Month"].astype(int)
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

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


