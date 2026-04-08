import os
from dotenv import load_dotenv
from openai import OpenAI

# ייבוא הסוכנים שלנו

from Sales_Analyst import SalesAnalyst
from Product_Analyst import ProductAnalyst
from Customer_Analyst import CustomerAnalyst

load_dotenv()

class ManagerAgent:
    def __init__(self, df):
        self.df = df
        self.ai_client = OpenAI()
    
    def _translate_to_command(self, user_text):
        """
        המוח של המנהל - מתרגם שפה חופשית לפקודות שהסוכנים מבינים
        """
        system_prompt = """
        You are the translation brain of a data system. Map the question to ONE command.
        
        SALES COMMANDS:
        - 'total_revenue' (total sales, money earned)
        - 'total_orders' (how many orders/invoices)
        - 'total_items_sold' (quantity of items sold)
        - 'average_order_value' (AOV, average spend)
        - 'top_countries_revenue' (best countries by money)
        - 'monthly_revenue' (sales by month, revenue over time)
        - 'top_products_revenue' (best selling products by money)
        - 'refund_rate' (percentage of returns/refunds)
        - 'busiest_days' (which days have most sales)
        - 'growth_rate' (monthly growth, MOM)
        - 'pareto_analysis' (80/20 rule, top products contributing to 80 percent revenue)
        - 'sales_anomalies' (unusual sales spikes)
        - 'bought_together' (frequently bought together)
        - 'sales_forecast' (prediction for next week)
        - 'sales_trend' (is business up or down, trend, overall direction)
        - 'revenue_drops' (detect drops, significant losses, bad months)

        PRODUCT COMMANDS:
        - 'product_revenue' (how much money a product made, revenue per item)
        - 'product_quantity' (units sold per product, volume)
        - 'product_avg_price' (average selling price of a product)
        - 'product_trend' (is a product selling better or worse, product growth)
        - 'product_return_rate' (which products are returned most, refund rate per item)
        - 'product_share' (contribution of a product to total sales, revenue share)
        - 'product_popularity' (most popular items by score, weighted popularity)
        - 'product_frequency' (how often is a product in a basket, purchase frequency)
        - 'product_lifecycle' (is a product growing, stable or declining, lifecycle status)
        - 'top_product' (most popular item overall)
        - 'total_unique_products' (variety of items)

        CUSTOMER COMMANDS:
        - 'total_unique_customers' (how many total distinct customers)
        - 'top_customer' (who spent the most)
        - 'top_spending_customers' (list of top spenders)
        - 'top_country' (where most orders come from)
        - 'revenue_by_country' (revenue generated per country)
        - 'most_popular_product_customer' (product that sold the most units)
        - 'repeat_customer_rate' (percentage of repeat customers)
        - 'repeat_customers' (loyalty, returning buyers, how many come back stats)
        - 'best_selling_product_per_country' (top product for each country)
        - 'high_value_loyal_customers' (VIP loyal customers)
        - 'customer_average_item_price' (average item price in store)

        Reply ONLY with the exact command word. If unsure, reply 'unknown'.
        """
        
        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.0 
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"[Manager Agent] ❌ AI Error: {e}")
            return "unknown"

    def handle_request(self, user_text):
        print(f"\n[Manager Agent] 🧠 Analyzing request: '{user_text}'")
        request_type = self._translate_to_command(user_text)
        print(f"[Manager Agent] 🎯 AI determined command: '{request_type}'")
        
        # טעינת הנתונים דרך ה-Data Agent
        df = self.df
        
        if df is None:
            return "Sorry, I couldn't get the data from the Data Agent."
        
        # --- לוגיקת ניתוב (Routing) ---
        
        # רשימות הפקודות המסודרות לפי מחלקות
        sales_commands = [
            "total_revenue", "total_orders", "total_items_sold", "average_order_value",
            "top_countries_revenue", "monthly_revenue", "top_products_revenue", 
            "refund_rate", "busiest_days", "growth_rate", "pareto_analysis", 
            "sales_anomalies", "bought_together", "sales_forecast", "sales_trend", "revenue_drops"
        ]
        
        product_commands = [
            "product_revenue", "product_quantity", "product_avg_price", "product_trend",
            "product_return_rate", "product_share", "product_popularity", 
            "product_frequency", "product_lifecycle", "top_product", "total_unique_products"
        ]
        
        customer_commands = [
            "total_unique_customers", "top_customer", "top_spending_customers", 
            "top_country", "revenue_by_country", "most_popular_product_customer", 
            "repeat_customer_rate", "repeat_customers", "best_selling_product_per_country", 
            "high_value_loyal_customers", "customer_average_item_price"
        ]

        # 1. מחלקת מכירות (Sales)
        if request_type in sales_commands:
            print("[Manager Agent] 📞 Routing to Sales Analyst...")
            analyst = SalesAnalyst(df)
            
            if request_type == "total_revenue":
                res = analyst.get_total_revenue()
                return f"💰 The total revenue is ${res:,.2f}."
            elif request_type == "total_orders":
                res = analyst.get_total_orders()
                return f"📦 Total number of orders: {res}"
            elif request_type == "average_order_value":
                res = analyst.get_average_order_value()
                return f"📊 The Average Order Value (AOV) is ${res:.2f}."
            elif request_type == "total_items_sold":
                res = analyst.get_total_items_sold()
                return f"🛒 Total items sold: {res}"
            elif request_type == "top_countries_revenue":
                res = analyst.get_top_countries_by_revenue()
                return f"🌍 Top countries by revenue: {res}"
            elif request_type == "monthly_revenue":
                res = analyst.get_monthly_revenue()
                return f"📅 Monthly Revenue Breakdown: {res}"
            elif request_type == "top_products_revenue":
                res = analyst.get_top_products_by_revenue()
                return f"🔝 Top products by revenue contribution: {res}"
            elif request_type == "refund_rate":
                res = analyst.get_refund_rate()
                return f"⚠️ Our current refund rate is {res:.2f}%."
            elif request_type == "busiest_days":
                res = analyst.get_busiest_days_of_week()
                return f"📈 Busiest days of the week: {res}"
            elif request_type == "growth_rate":
                res = analyst.get_mom_growth_rate()
                return f"🚀 Month-over-Month Growth: {res:.2f}%"
            elif request_type == "pareto_analysis":
                res = analyst.get_pareto_products_count()
                return f"⚖️ Pareto Analysis: {res} products are responsible for 80% of your total revenue."
            elif request_type == "sales_anomalies":
                res = analyst.get_sales_anomalies()
                return f"🚨 Sales Anomalies Detected: {res if res else 'No major anomalies found.'}"
            elif request_type == "sales_forecast":
                res = analyst.get_simple_sales_forecast()
                return f"🔮 Estimated sales forecast for the next 7 days: ${res:,.2f}"
            elif request_type == "bought_together":
                res = analyst.get_frequently_bought_together('WHITE HANGING HEART T-LIGHT HOLDER')
                return f"💡 People who bought that also bought: {res}"
            elif request_type == "sales_trend":
                res = analyst.get_sales_trend()
                return f"📈 Current Sales Trend: {res}"
            elif request_type == "revenue_drops":
                res = analyst.detect_revenue_drops()
                if isinstance(res, str): 
                    return res
                return f"⚠️ Significant Revenue Drops Detected: {res}"

        # 2. מחלקת מוצרים (Products)
        elif request_type in product_commands:
            print("[Manager Agent] 📞 Routing to Product Analyst...")
            analyst = ProductAnalyst(df)
            
            if request_type == "product_revenue":
                res = analyst.get_product_revenue()
                return f"💰 Revenue per Product (Top 5): {list(res.items())[:5]}"
            elif request_type == "product_quantity":
                res = analyst.get_total_products_sold()
                return f"📦 Units Sold per Product (Top 5): {list(res.items())[:5]}"
            elif request_type == "product_avg_price":
                res = analyst.get_average_price_per_product()
                return f"🏷️ Average Weighted Price per Product: {list(res.items())[:5]}"
            elif request_type == "product_return_rate":
                res = analyst.get_product_return_rate()
                return f"⚠️ Product Return Rates (High to Low): {list(res.items())[:5]}"
            elif request_type == "product_share":
                return f"📊 Revenue Share % per Product: {analyst.get_product_revenue_share()}"
            elif request_type == "product_popularity":
                return f"⭐ Product Popularity Scores (Weighted): {analyst.get_product_popularity_score()}"
            elif request_type == "product_frequency":
                return f"🛒 Purchase Frequency (% of orders): {analyst.get_product_purchase_frequency()}"
            elif request_type == "top_product":
                res = analyst.get_top_products_by_quantity(limit=1)
                return f"🏆 Our top selling product is: {res}."
            elif request_type == "total_unique_products":
                res = analyst.get_total_products_sold()
                return f"🔢 We have {len(res)} unique products in our catalog."
            elif request_type in ["product_trend", "product_lifecycle"]:
                example_product = "WHITE HANGING HEART T-LIGHT HOLDER"
                if request_type == "product_trend":
                    return f"📉 Trend for '{example_product}': {analyst.get_product_sales_trend(example_product)}"
                return f"🔄 Lifecycle Status for '{example_product}': {analyst.get_product_lifecycle_status(example_product)}"

        # 3. מחלקת לקוחות (Customers)
        elif request_type in customer_commands:
            print("[Manager Agent] 📞 Routing to Customer Analyst...")
            analyst = CustomerAnalyst(df)
            
            if request_type == "total_unique_customers":
                res = analyst.get_total_unique_customers()
                return f"👥 We have {res} unique customers."
            elif request_type == "top_customer":
                res = analyst.get_top_customer()
                return f"🥇 Our top customer ID is: {res}."
            elif request_type == "top_spending_customers":
                res = analyst.get_top_spending_customers()
                return f"💰 Top spending customers: {res}"
            elif request_type == "top_country":
                res = analyst.get_top_country()
                return f"📍 Most orders come from: {res}."
            elif request_type == "revenue_by_country":
                res = analyst.get_revenue_by_country()
                return f"🌍 Revenue by Country: {res}"
            elif request_type == "most_popular_product_customer":
                res = analyst.get_most_popular_product()
                return f"🛍️ The most popular product by units sold is: {res}."
            elif request_type == "repeat_customer_rate":
                res = analyst.get_repeat_customer_rate()
                return f"🔄 Repeat Customer Rate: {res}%."
            elif request_type == "repeat_customers":
                res = analyst.get_repeat_customers_stats()
                return (f"🔄 Customer Loyalty Stats:\n"
                        f"- Repeat Customers: {res['repeat_customers_count']}\n"
                        f"- Retention Rate: {res['repeat_customers_percentage']}%\n"
                        f"- Total Unique Customers: {res['total_unique_customers']}")
            elif request_type == "best_selling_product_per_country":
                res = analyst.get_best_selling_product_per_country()
                return f"🗺️ Best selling products per country: {res}"
            elif request_type == "high_value_loyal_customers":
                res = analyst.get_high_value_loyal_customers()
                return f"💎 VIP Loyal Customers (IDs): {res}"
            elif request_type == "customer_average_item_price":
                res = analyst.get_average_item_price()
                return f"🏷️ The average item price is ${res}."

        # במקרה שה-AI לא זיהה פקודה
        return "I'm not sure how to handle that request yet. Try asking about revenue, growth, or top products!"