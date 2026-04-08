import os
from dotenv import load_dotenv

# ספריות LangChain ו-LangGraph החדשות
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
# ייבוא האנליסטים ששדרגנו
from Sales_Analyst import SalesAnalyst
from Product_Analyst import ProductAnalyst
from Customer_Analyst import CustomerAnalyst

load_dotenv()

class ManagerAgent:
    def __init__(self, df):
        self.df = df
        
        # 1. הגדרת המודל (המוח)
        # temperature=0 מבטיח תשובות מדויקות ועקביות על נתונים
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # 2. אתחול מחלקות האנליסטים
        self.sales_analyst = SalesAnalyst(df)
        self.product_analyst = ProductAnalyst(df)
        self.customer_analyst = CustomerAnalyst(df)

        # 3. איסוף כל הכלים (Methods) לרשימה אחת
        # המודל יבחר ביניהם לפי ה-Docstrings שכתבנו!
        self.tools = [
            self.sales_analyst.get_total_revenue,
            self.sales_analyst.get_total_orders,
            self.sales_analyst.get_total_items_sold,
            self.sales_analyst.get_average_order_value,
            self.sales_analyst.get_top_countries_by_revenue,
            self.sales_analyst.get_monthly_revenue,
            self.sales_analyst.get_top_products_by_revenue,
            self.sales_analyst.get_refund_rate,
            self.sales_analyst.get_revenue_by_date_range,
            self.sales_analyst.get_busiest_days_of_week,
            self.sales_analyst.get_mom_growth_rate,
            self.sales_analyst.get_pareto_products_count,
            self.sales_analyst.get_sales_anomalies,
            self.sales_analyst.get_frequently_bought_together,
            self.sales_analyst.get_simple_sales_forecast,
            self.sales_analyst.get_sales_trend,
            self.sales_analyst.detect_revenue_drops,
            self.sales_analyst.get_repeat_customers_stats,
            
            # Product Tools
            self.product_analyst.get_total_products_sold,
            self.product_analyst.get_product_revenue,
            self.product_analyst.get_average_price_per_product,
            self.product_analyst.get_product_sales_trend,
            self.product_analyst.get_top_products_by_revenue,
            self.product_analyst.get_top_products_by_quantity,
            self.product_analyst.get_low_stock_indicator,
            self.product_analyst.get_product_conversion_rate,
            self.product_analyst.get_product_return_rate,
            self.product_analyst.get_product_revenue_share,
            self.product_analyst.get_product_growth_rate,
            self.product_analyst.get_product_popularity_score,
            self.product_analyst.get_product_profit_estimate,
            self.product_analyst.get_product_purchase_frequency,
            self.product_analyst.get_product_lifecycle_status,
        
            # Customer Tools
             self.customer_analyst.get_total_revenue,
            self.customer_analyst.get_total_unique_customers,
            self.customer_analyst.get_top_country,
            self.customer_analyst.get_total_items_sold,
            self.customer_analyst.get_average_item_price,
            self.customer_analyst.get_top_customer,
            self.customer_analyst.get_top_spending_customers,
            self.customer_analyst.get_revenue_by_country,
            self.customer_analyst.get_most_popular_product,
            self.customer_analyst.get_refund_rate,
            self.customer_analyst.get_repeat_customer_rate,
            self.customer_analyst.get_best_selling_product_per_country,
            self.customer_analyst.get_average_order_value,
            self.customer_analyst.get_monthly_revenue_trend,
            self.customer_analyst.get_high_value_loyal_customers
        
        ]

        # 4. הגדרת "הנחיות המערכת" (System Prompt)
        # כאן אנחנו נותנים לסוכן את ה"אישיות" והחוקים
        system_message = (
            "You are the AI Data Manager of an E-commerce store. "
            "You have access to sales, product, and customer tools. "
            "Your job is to provide clear, business-oriented insights. "
            "1. If the user asks a general question (like 'hello' or 'how are you'), answer naturally. "
            "2. If you need data, call the appropriate tool. "
            "3. If a question requires multiple steps, call tools sequentially. "
            "4. Always answer in the language the user speaks (Hebrew or English)."
        )
        self.memory = MemorySaver()  # לשמור את ההיסטוריה של השיחות והכלים שנקראו
        # 5. יצירת הסוכן האוטונומי (ReAct Agent)
        self.agent_executor = create_react_agent(
            self.llm, 
            tools=self.tools,
            prompt=system_message,
            checkpointer=self.memory
        )
        self.config = {"configurable": {"thread_id": "ecommerce_chat_1"}}
    
    def handle_request(self, user_text):
        print(f"\n[Manager Agent] 🧠 Thinking about: '{user_text}'...")
        try:
            # אנחנו מעבירים את ה-config עם ה-thread_id בכל קריאה
            response = self.agent_executor.invoke(
                {"messages": [("user", user_text)]},
                config=self.config
            )
            
            final_answer = response["messages"][-1].content
            return final_answer
            
        except Exception as e:
            return f"❌ Error: {e}"