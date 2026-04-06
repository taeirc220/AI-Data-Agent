import os
from dotenv import load_dotenv
from openai import OpenAI

# Importing our Data Agent
from Data_Agent import DataAgent

# Importing our newly structured Analyst Departments
from Sales_Analyst import SalesAnalyst
from Product_Analyst import ProductAnalyst
from Customer_Analyst import CustomerAnalyst

load_dotenv()

class ManagerAgent:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.ai_client = OpenAI()
    
    def _translate_to_command(self, user_text):
        """
        The brain of the Manager. Now aware of all 6 commands across 3 departments.
        """
        system_prompt = """
        You are the translation brain of a data system. The user will ask a business question.
        You must map their question to ONE of the following internal commands:
        
        SALES COMMANDS:
        - 'total_sales' (revenue, total money, sales, earnings)
        - 'average_order_value' (average order, aov, average spend per cart)
        
        PRODUCT COMMANDS:
        - 'top_product' (best seller, most popular item, top product)
        - 'total_unique_products' (how many different products, variety of items)
        
        CUSTOMER COMMANDS:
        - 'top_customer' (best customer, who spent the most, top buyer)
        - 'top_country' (top location, best country, where most orders come from)
        
        - 'unknown' (if they ask about anything else)
        
        Reply ONLY with the exact command word, nothing else. No punctuation.
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
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"[Manager Agent] ❌ AI Error: {e}")
            return "unknown"

    def handle_request(self, user_text):
        print(f"\n[Manager Agent] 🧠 Analyzing request: '{user_text}'")
        
        request_type = self._translate_to_command(user_text)
        print(f"[Manager Agent] 🎯 AI determined command: '{request_type}'")
        
        # Step 1: Always get the data first
        d_agent = DataAgent(self.data_file_path)
        df = d_agent.get_data()
        
        if df is None:
            return "Sorry, I couldn't get the data from the Data Agent."
        
        # Step 2: Route the request to the correct Department (Analyst)
        
        # --- SALES DEPARTMENT ---
        if request_type in ["total_sales", "average_order_value"]:
            print("[Manager Agent] 📞 Routing to Sales Analyst...")
            analyst = SalesAnalyst(df)
            
            if request_type == "total_sales":
                res = analyst.get_total_sales()
                return f"The total sales revenue is ${res:.2f}." if res is not None else "Error calculating sales."
            else:
                res = analyst.get_average_order_value()
                return f"The Average Order Value (AOV) is ${res:.2f}." if res is not None else "Error calculating AOV."

        # --- PRODUCT DEPARTMENT ---
        elif request_type in ["top_product", "total_unique_products"]:
            print("[Manager Agent] 📞 Routing to Product Analyst...")
            analyst = ProductAnalyst(df)
            
            if request_type == "top_product":
                res = analyst.get_top_product()
                return f"Our top selling product is: '{res}'." if res is not None else "Error finding top product."
            else:
                res = analyst.get_total_unique_products()
                return f"We sold {res} unique products." if res is not None else "Error calculating unique products."

        # --- CUSTOMER DEPARTMENT ---
        elif request_type in ["top_customer", "top_country"]:
            print("[Manager Agent] 📞 Routing to Customer Analyst...")
            analyst = CustomerAnalyst(df)
            
            if request_type == "top_customer":
                res = analyst.get_top_customer()
                return f"Our top customer is ID number: {res}." if res is not None else "Error finding top customer."
            else:
                res = analyst.get_top_country()
                return f"Most of our orders come from: {res}." if res is not None else "Error finding top country."
        
        # --- UNKNOWN ---
        else:
            return "I don't know how to handle this request yet. Try asking about sales, products, or customers!"