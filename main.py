import streamlit as st
import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import squarify

st.set_page_config(
    page_title="Retail Analysis Dashboard",
    page_icon="🛒",
    layout="wide"
)

@st.cache_data
def load_data():
    customers = pd.read_csv('customer_profiles.csv')
    products = pd.read_csv('product_inventory.csv')
    sales = pd.read_csv('sales_transaction.csv')

    conn = sqlite3.connect(':memory:')

    customers.to_sql('Customers', conn, if_exists='replace', index=False)
    products.to_sql('Products', conn, if_exists='replace', index=False)
    sales.to_sql('Sales', conn, if_exists='replace', index=False)

    sales_CTE_query = """
    WITH sales_cleaned AS (
      SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY CustomerID, ProductID, QuantityPurchased, TransactionDate, Price) AS row_num
        FROM Sales
      ) WHERE row_num = 1
    )
    """
    
    customers_CTE_query = f"""
    {sales_CTE_query},
    customers_cleaned AS (
      SELECT CustomerID, Age, Gender, COALESCE(Location, 'Unknown') AS Location, JoinDate
      FROM Customers
    )
    """

    revenue_q = f"""
    {sales_CTE_query}
    SELECT s.ProductID, SUM(s.QuantityPurchased * p.Price) as TotalRevenue, SUM(s.QuantityPurchased) as TotalQuantitySold, p.ProductName, p.Category
    FROM sales_cleaned s JOIN Products p ON s.ProductID = p.ProductID
    GROUP BY s.ProductID, p.ProductName, p.Category ORDER BY TotalRevenue desc
    """
    revenue_df = pd.read_sql(revenue_q, conn)

    category_revenue_q = f"""
    {sales_CTE_query}
    SELECT p.Category, SUM(s.QuantityPurchased * p.Price) as TotalRevenue
    FROM sales_cleaned s JOIN Products p ON s.ProductID = p.ProductID
    GROUP BY p.Category ORDER BY TotalRevenue desc
    """
    revenue_df_category = pd.read_sql(category_revenue_q, conn)

    quantile_q_v2 = f"""
    {sales_CTE_query},
    ProductSales AS (SELECT p.ProductID, p.StockLevel, SUM(s.QuantityPurchased) AS TotalQuantitySold FROM sales_cleaned s JOIN products p ON s.ProductID = p.ProductID GROUP BY p.ProductID, p.StockLevel),
    ProductKPIs AS (SELECT *, (StockLevel * 1.0 / TotalQuantitySold) AS InventoryCoverRatio FROM ProductSales WHERE TotalQuantitySold > 0),
    RankedData AS (SELECT StockLevel, TotalQuantitySold, InventoryCoverRatio, (SELECT COUNT(*) FROM ProductSales) AS TotalProducts, (SELECT COUNT(*) FROM ProductKPIs) AS TotalProductsSold, ROW_NUMBER() OVER (ORDER BY StockLevel) AS StockRank, ROW_NUMBER() OVER (ORDER BY TotalQuantitySold) AS SalesRank, ROW_NUMBER() OVER (ORDER BY InventoryCoverRatio) AS RatioRank FROM ProductKPIs)
    SELECT (SELECT StockLevel FROM RankedData WHERE StockRank = CAST(TotalProducts * 0.75 AS INTEGER)) AS Stock_Q3, (SELECT TotalQuantitySold FROM RankedData WHERE SalesRank = CAST(TotalProductsSold * 0.25 AS INTEGER)) AS Sales_Q1, (SELECT InventoryCoverRatio FROM RankedData WHERE RatioRank = CAST(TotalProductsSold * 0.90 AS INTEGER)) AS Ratio_P90
    FROM RankedData LIMIT 1;
    """
    quantile_results = pd.read_sql(quantile_q_v2, conn)
    sales_q1_thresh = quantile_results['Sales_Q1'].iloc[0]
    stock_q3_thresh = quantile_results['Stock_Q3'].iloc[0]
    ratio_p90_thresh = quantile_results['Ratio_P90'].iloc[0]

    at_risk_products_q = f"""
    {sales_CTE_query},
    product_sales_inventory AS (SELECT p.ProductID, p.ProductName, p.Category, p.StockLevel, SUM(s.QuantityPurchased) AS TotalQuantitySold FROM sales_cleaned s JOIN products p ON s.ProductID = p.ProductID GROUP BY p.ProductID, p.ProductName, p.Category, p.StockLevel),
    product_icr AS (SELECT *, CASE WHEN TotalQuantitySold > 0 THEN ROUND(StockLevel * 1.0 / TotalQuantitySold, 2) ELSE 9999 END AS InventoryCoverRatio FROM product_sales_inventory)
    SELECT ProductName, Category, TotalQuantitySold, StockLevel, InventoryCoverRatio, CASE WHEN (TotalQuantitySold < {sales_q1_thresh} AND StockLevel > {stock_q3_thresh}) OR (InventoryCoverRatio > {ratio_p90_thresh}) THEN 'High Risk - Overstocked' ELSE 'Normal' END AS RiskLabel
    FROM product_icr ORDER BY InventoryCoverRatio DESC;
    """
    product_risk_df = pd.read_sql(at_risk_products_q, conn)

    revenue_growth_q = f"""
    {sales_CTE_query},
    monthly_revenue AS (SELECT STRFTIME('%Y-%m', '20' || SUBSTR(TransactionDate, 7, 2) || '-' || SUBSTR(TransactionDate, 4, 2) || '-' || SUBSTR(TransactionDate, 1, 2)) AS SalesMonth, SUM(s.QuantityPurchased * p.Price) as MonthlyTotalRevenue FROM sales_cleaned s JOIN products p ON s.ProductID = p.ProductID GROUP BY SalesMonth ORDER BY SalesMonth),
    revenue_with_previous AS (SELECT SalesMonth, MonthlyTotalRevenue, LAG(MonthlyTotalRevenue, 1, 0) OVER (ORDER BY SalesMonth) AS PreviousMonthRevenue FROM monthly_revenue)
    SELECT SalesMonth, MonthlyTotalRevenue, PreviousMonthRevenue, CASE WHEN PreviousMonthRevenue = 0 THEN NULL ELSE ROUND(((MonthlyTotalRevenue - PreviousMonthRevenue) / PreviousMonthRevenue) * 100, 2) END AS RevenueGrowthRate
    FROM revenue_with_previous;
    """
    monthly_revenue_df = pd.read_sql(revenue_growth_q, conn)

    customer_quantile_values_q = f"""
    {sales_CTE_query},
    CustomerSpend AS (SELECT SUM(s.QuantityPurchased * p.Price) AS TotalSpend FROM customers c JOIN sales_cleaned s ON c.CustomerID = s.CustomerID JOIN products p ON s.ProductID = p.ProductID GROUP BY c.CustomerID HAVING SUM(s.QuantityPurchased * p.Price) > 0),
    RankedSpend AS (SELECT TotalSpend, ROW_NUMBER() OVER (ORDER BY TotalSpend) AS SpendRank, COUNT(*) OVER () AS CustomerCount FROM CustomerSpend)
    SELECT (SELECT TotalSpend FROM RankedSpend WHERE SpendRank = CAST(CustomerCount * 0.75 AS INTEGER)) AS P75_Threshold, (SELECT TotalSpend FROM RankedSpend WHERE SpendRank = CAST(CustomerCount * 0.90 AS INTEGER)) AS P90_Threshold, (SELECT TotalSpend FROM RankedSpend WHERE SpendRank = CAST(CustomerCount * 0.95 AS INTEGER)) AS P95_Threshold, (SELECT TotalSpend FROM RankedSpend WHERE SpendRank = CAST(CustomerCount * 0.99 AS INTEGER)) AS P99_Threshold
    FROM RankedSpend LIMIT 1;
    """
    customer_quantiles_df = pd.read_sql(customer_quantile_values_q, conn)

    
    customer_behaviour_segments_q = f"""
    {customers_CTE_query},
    TransactionGaps AS (
      SELECT
        CustomerID,
        JULIANDAY(DATE('20' || SUBSTR(TransactionDate, 7, 2) || '-' || SUBSTR(TransactionDate, 4, 2) || '-' || SUBSTR(TransactionDate, 1, 2))) -
        JULIANDAY(LAG(DATE('20' || SUBSTR(TransactionDate, 7, 2) || '-' || SUBSTR(TransactionDate, 4, 2) || '-' || SUBSTR(TransactionDate, 1, 2)), 1)
        OVER (PARTITION BY CustomerID ORDER BY DATE('20' || SUBSTR(TransactionDate, 7, 2) || '-' || SUBSTR(TransactionDate, 4, 2) || '-' || SUBSTR(TransactionDate, 1, 2))))
        AS DaysSinceLastPurchase
      FROM sales_cleaned
    ),
    CustomerFrequency AS (
      SELECT
        CustomerID,
        AVG(DaysSinceLastPurchase) AS AvgDaysBetweenPurchases
      FROM TransactionGaps
      WHERE DaysSinceLastPurchase IS NOT NULL
      GROUP BY CustomerID
    ),
    RankedBehaviour AS (
      SELECT
        AvgDaysBetweenPurchases,
        ROW_NUMBER() OVER (ORDER BY AvgDaysBetweenPurchases) AS FrequencyRank,
        COUNT(*) OVER () AS TotalRepeatCustomers
      FROM CustomerFrequency
    )
    SELECT
        MAX(CASE WHEN FrequencyRank = CAST(TotalRepeatCustomers * 0.33 AS INTEGER) THEN AvgDaysBetweenPurchases END) AS P33_Threshold,
        MAX(CASE WHEN FrequencyRank = CAST(TotalRepeatCustomers * 0.66 AS INTEGER) THEN AvgDaysBetweenPurchases END) AS P66_Threshold
    FROM RankedBehaviour;
    """
    customer_behaviour_segments_df = pd.read_sql(customer_behaviour_segments_q, conn)
    
    visualization_data_q = f"""
WITH
  sales_cleaned AS (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY CustomerID, ProductID, QuantityPurchased, TransactionDate, Price ORDER BY TransactionID) as rn
        FROM sales
    ) WHERE rn = 1
  ),
  CustomerValue AS (
    SELECT
        s.CustomerID,
        SUM(p.Price * s.QuantityPurchased) AS TotalSpend
    FROM sales_cleaned s
    JOIN products p ON s.ProductID = p.ProductID
    GROUP BY s.CustomerID
  ),
  CustomerFrequency AS (
    SELECT
        CustomerID,
        AVG(DaysSinceLastPurchase) AS AvgDaysBetweenPurchases
    FROM (
        SELECT
            CustomerID,
            JULIANDAY(DATE('20' || SUBSTR(TransactionDate, 7, 2) || '-' || SUBSTR(TransactionDate, 4, 2) || '-' || SUBSTR(TransactionDate, 1, 2))) -
            JULIANDAY(LAG(DATE('20' || SUBSTR(TransactionDate, 7, 2) || '-' || SUBSTR(TransactionDate, 4, 2) || '-' || SUBSTR(TransactionDate, 1, 2)), 1)
            OVER (PARTITION BY CustomerID ORDER BY DATE('20' || SUBSTR(TransactionDate, 7, 2) || '-' || SUBSTR(TransactionDate, 4, 2) || '-' || SUBSTR(TransactionDate, 1, 2))))
            AS DaysSinceLastPurchase
        FROM sales_cleaned
    )
    WHERE DaysSinceLastPurchase IS NOT NULL
    GROUP BY CustomerID
  ),
  FinalCustomerMatrix AS (
    -- This is the corrected CTE without the unnecessary subquery
    SELECT
        c.CustomerID,
        val.TotalSpend,
        -- Corrected ValueSegment Logic
        CASE
            WHEN val.TotalSpend > {customer_quantiles_df['P99_Threshold'][0]} THEN 'Diamond'
            WHEN val.TotalSpend > {customer_quantiles_df['P95_Threshold'][0]} THEN 'Platinum'
            WHEN val.TotalSpend > {customer_quantiles_df['P90_Threshold'][0]} THEN 'Gold'
            WHEN val.TotalSpend > {customer_quantiles_df['P75_Threshold'][0]} THEN 'Silver'
            WHEN val.TotalSpend IS NOT NULL THEN 'Bronze'
            ELSE 'Inactive'
        END AS ValueSegment,
        -- Corrected FrequencySegment Logic
        CASE
            WHEN freq.AvgDaysBetweenPurchases <= {customer_behaviour_segments_df['P33_Threshold'][0]} THEN 'Frequent'
            WHEN freq.AvgDaysBetweenPurchases <= {customer_behaviour_segments_df['P66_Threshold'][0]} THEN 'Regular'
            WHEN freq.AvgDaysBetweenPurchases > {customer_behaviour_segments_df['P66_Threshold'][0]} THEN 'Infrequent'
            WHEN val.TotalSpend IS NOT NULL THEN 'One-Time Buyer'
            ELSE 'Inactive'
        END AS FrequencySegment
    FROM
        customers c
    LEFT JOIN
        CustomerValue val ON c.CustomerID = val.CustomerID
    LEFT JOIN
        CustomerFrequency freq ON c.CustomerID = freq.CustomerID
  )
-- The final aggregation for visualization
SELECT
    ValueSegment,
    FrequencySegment,
    COUNT(CustomerID) AS CustomerCount,
    SUM(TotalSpend) AS TotalRevenue
FROM
    FinalCustomerMatrix
GROUP BY
    ValueSegment,
    FrequencySegment
"""
    viz_df = pd.read_sql(visualization_data_q, conn)

    conn.close()
    
    return {
        "revenue_df": revenue_df,
        "revenue_df_category": revenue_df_category,
        "product_risk_df": product_risk_df,
        "quantile_results": quantile_results,
        "monthly_revenue_df": monthly_revenue_df,
        "viz_df": viz_df
    }

data = load_data()

def show_executive_summary():
    st.title("Executive Summary 📈")
    st.markdown("""
    **Project Objective:** The objective of this case study is to address the retail company's stagnant growth, declining customer engagement, and a lack of clear operational insights. I have leveraged the provided sales, customer, and product datasets to develop a strategy that directly answers the key business problems outlined in the project statement.
    """)

    with st.expander("Expand to see the Methodology"):
        st.markdown("""
        To solve the stated problems, I followed the following process:

        1. **Tools and Technology used :** I have leveraged the SQLite library in conjunction with Colab Notebook and python pandas to execute my sql queries in a robust and re-usable manner and also to generate useful visualizations.
        2. **Data Cleaning:** I began by performing data cleaning. This involved de-duplicating transactions, handling missing values, and, most critically, identifying and correcting a major price anomaly that was skewing revenue reporting by a factor of 100.
        3. **Product Performance Analysis:** To address **Product Performance Variability**, instead of just identifying high and low sales, I created an `InventoryCoverRatio` metric and used percentile-based thresholds to precisely identify overstocked, slow-moving products that were hindering profitability.
        4. **Customer Segmentation & Behavioral Analysis:** To solve the problems of **Ineffective Customer Segmentation** and **Lack of Customer Behavior Insights**, I built a multi-dimensional segmentation model. I first segmented customers by value (`TotalSpend`) into five data-driven tiers. I then deepened this analysis by calculating the `AvgDaysBetweenPurchases` for each customer, allowing me to further segment them by loyalty and frequency. By combining these two models into a final 2D matrix, I was able to create detailed, actionable customer personas.
        """)

    st.subheader("Key Findings & Solutions to Business Problems")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("Solved: Product Performance Variability")
        st.markdown("""
        I discovered that the business is not suffering from poor product diversity but from inefficient inventory management. A significant amount of capital is trapped in "High-Risk" products that have both low sales and high stock.

        > **Solution:** My analysis produced a prioritized list of the top 25 most overstocked products. I recommend an immediate clearance campaign for these items to clear stock. The SQL query can be used as a recurring quarterly report to prevent future over-purchasing.
        """)

    with col2:
        st.info("Solved: Ineffective Customer Segmentation")
        st.markdown("""
        I built a five-tier segmentation model (`Diamond`, `Platinum`, `Gold`, `Silver`, `Bronze`) that revealed a classic Pareto distribution of value. I saw that the top 10% of customers are responsible for over 21% of all revenue.

        > **Solution:** I recommend implementing a tiered loyalty program and segmented marketing campaigns.
        """)
        
    st.success("Solved: Lack of Customer Behavior Insights")
    st.markdown("""
    By analyzing purchase frequency, I uncovered the critical behavioral patterns behind the value segments. I found that the most valuable customers are also highly frequent, while the largest customer segment (`Bronze`) contains a significant number of "Infrequent" and "One-Time Buyers" who are at high risk of churning.

    > **Solution:** The primary goal should be to **Retain** the high-value, frequent Buyers ( Diamond and Platinum) , **Develop** the engaged "Silver" customers to increase their value, and **Reach out to** the large base of "One-Time Buyers" to turn them into loyal repeat purchasers.
    """)

def show_product_analysis():
    st.title("🛒 Product & Inventory Performance")

    st.header("What Are Our Top-Performing Products?")
    st.markdown("After correcting for a major data pricing anomaly, the true top-performing products were identified. Revenue is healthily distributed across several key items rather than being dependent on one superstar.")
    
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    top_products = data['revenue_df'].head(10)
    sns.barplot(data=top_products, x='TotalRevenue', y='ProductName', hue='Category', dodge=False, ax=ax1, palette='viridis')
    ax1.set_title('Top 10 Products by Total Revenue', fontsize=16)
    ax1.set_xlabel('Total Revenue ($)')
    ax1.set_ylabel('Product')
    st.pyplot(fig1)

    st.header("How Can We Optimize Our Inventory?")
    st.markdown("""
    By analyzing the relationship between sales volume and stock levels, we can identify two critical areas for action:
    - **Overstock Risk (Top-Left):** Products with low sales but high inventory, which trap capital.
    - **Stockout Risk (Bottom-Right):** High-demand products with low inventory, which leads to lost sales.
    """)
    
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=data['product_risk_df'], x='TotalQuantitySold', y='StockLevel', hue='RiskLabel',
        palette={'High Risk - Overstocked': 'red', 'Normal': 'grey'}, size='InventoryCoverRatio',
        sizes=(50, 500), alpha=0.7, ax=ax2
    )
    stock_q3_thresh = data['quantile_results']['Stock_Q3'].iloc[0]
    sales_q1_thresh = data['quantile_results']['Sales_Q1'].iloc[0]
    ax2.axhline(y=stock_q3_thresh, color='red', linestyle='--', lw=1)
    ax2.axvline(x=sales_q1_thresh, color='red', linestyle='--', lw=1)
    ax2.set_title('Inventory Position Matrix', fontsize=16)
    ax2.set_xlabel('Total Quantity Sold (Demand)')
    ax2.set_ylabel('Current Stock Level (Supply)')
    st.pyplot(fig2)

    st.subheader("Action List: Top Overstocked Products")
    st.dataframe(data['product_risk_df'][data['product_risk_df']['RiskLabel'] == 'High Risk - Overstocked'].head(10))

def show_sales_trends():
    st.title("📅 Sales Trends Analysis")
    st.markdown("Analyzing sales over time helps us understand business momentum and seasonality.")
    
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=data['monthly_revenue_df'], x='SalesMonth', y='MonthlyTotalRevenue', marker='o', ax=ax1, color='cyan')
    ax1.set_title('Total Monthly Revenue', fontsize=16)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Revenue ($)')
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    
    st.header("What is our Month-over-Month (MoM) Growth?")
    st.markdown("""
    The MoM growth rate reveals high volatility. While some months show growth, they are often followed by significant declines (e.g., -7% in Feb, -11% in July). This indicates a lack of sustained, consistent growth.
    """)
    
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    monthly_revenue_df = data['monthly_revenue_df']
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in monthly_revenue_df['RevenueGrowthRate'].fillna(0)]
    sns.barplot(data=monthly_revenue_df, x='SalesMonth', y='RevenueGrowthRate', palette=colors, ax=ax2)
    ax2.axhline(y=0, color='grey', linewidth=0.8)
    ax2.set_title('Month-over-Month Revenue Growth Rate (%)', fontsize=16)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Growth Rate (%)')
    plt.xticks(rotation=45)
    st.pyplot(fig2)

def show_customer_segmentation():
    st.title("👥 Customer Segmentation Analysis")
    st.markdown("To move beyond a 'one-size-fits-all' strategy, we segmented customers based on their **Value** (Total Spend) and **Frequency** (Avg. Days Between Purchases).")

    st.header("Customer Segmentation Landscape")
    st.markdown("This heatmap shows the number of customers in each combined segment. It clearly identifies our key strategic groups, from our 'Diamond/Frequent' champions to the large pool of 'Bronze/Infrequent' customers.")

    viz_df = data['viz_df']
    value_order = ['Diamond', 'Platinum', 'Gold', 'Silver', 'Bronze', 'Inactive']
    frequency_order = ['Frequent', 'Regular', 'Infrequent', 'One-Time Buyer', 'Inactive']

    viz_df['ValueSegment'] = pd.Categorical(viz_df['ValueSegment'], categories=value_order, ordered=True)
    viz_df['FrequencySegment'] = pd.Categorical(viz_df['FrequencySegment'], categories=frequency_order, ordered=True)

    heatmap_pivot = viz_df.pivot_table(index='ValueSegment', columns='FrequencySegment', values='CustomerCount')

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(heatmap_pivot, annot=True, fmt='.0f', cmap='viridis', linewidths=.5, ax=ax)
    ax.set_title('Customer Segmentation Matrix: Number of Customers', fontsize=16)
    ax.set_xlabel('Frequency Segment')
    ax.set_ylabel('Value Segment')
    st.pyplot(fig)

    st.header("Strategic Personas")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 The Elites (e.g., Diamond/Frequent)")
        st.write("Your most valuable and loyal customers. They spend the most and shop the most often. **Goal: Retain at all costs.**")

        st.subheader("🌱 The Key Growth Prospects (e.g., Silver/Frequent)")
        st.write("Engaged and valuable, with high potential to become Elites. **Goal: Nurture and upsell.**")
        
    with col2:
        st.subheader("💔 At-Risk VIPs (e.g., Gold/Infrequent)")
        st.write("They spend a lot but purchase infrequently, making them a high-value churn risk. **Goal: Reach out with targeted communication.**")

        st.subheader("💡 The Opportunity (e.g., Bronze/One-Time Buyer)")
        st.write("Your largest group. Converting even a small fraction to repeat purchasers will have a huge impact. **Goal: Reach out and encourage a second purchase.**")

plt.style.use('dark_background')
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Executive Summary", "Product & Inventory Analysis", "Sales Trends", "Customer Segmentation"])

if page == "Executive Summary":
    show_executive_summary()
elif page == "Product & Inventory Analysis":
    show_product_analysis()
elif page == "Sales Trends":
    show_sales_trends()
elif page == "Customer Segmentation":
    show_customer_segmentation()