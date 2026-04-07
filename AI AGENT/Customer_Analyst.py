import pandas as pd

class CustomerAnalyst:
    def __init__(self, data_frame):
        # We store the dataframe here to analyze customer behavior
        self.df = data_frame.copy()
        
        # Data Preparation: Calculate Revenue for all rows to save time in functions
        if 'Quantity' in self.df.columns and 'Price' in self.df.columns:
            self.df['Revenue'] = self.df['Quantity'] * self.df['Price']
            
        # Data Preparation: Ensure InvoiceDate is a datetime object (if it exists)
        if 'InvoiceDate' in self.df.columns:
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce')

    # ==========================================
    # 🟢 LEVEL 1: EASY (Basic Aggregations & Counts)
    # ==========================================

    def get_total_revenue(self):
        """1. Returns the total revenue generated across all data."""
        try:
            return round(self.df['Revenue'].sum(), 2)
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    def get_total_unique_customers(self):
        """2. Returns the total number of unique customers."""
        try:
            return self.df['Customer ID'].nunique()
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    def get_top_country(self):
        """3. Finds the country with the most recorded transactions."""
        try:
            return self.df['Country'].value_counts().idxmax()
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    def get_total_items_sold(self):
        """4. Returns the total amount of items sold (sum of quantity)."""
        try:
            # We filter for Quantity > 0 to ignore returns/refunds
            valid_sales = self.df[self.df['Quantity'] > 0]
            return valid_sales['Quantity'].sum()
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    def get_average_item_price(self):
        """5. Returns the average price of a product in the store."""
        try:
            return round(self.df['Price'].mean(), 2)
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    # ==========================================
    # 🟡 LEVEL 2: MEDIUM (Grouping, Sorting, & Filtering)
    # ==========================================

    def get_top_customer(self):
        """6. Finds the Customer ID who generated the most revenue."""
        try:
            top_customer = self.df.groupby('Customer ID')['Revenue'].sum().idxmax()
            return int(top_customer)
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    def get_top_spending_customers(self, top_n=5):
        """7. Returns a dictionary of the top N customers and their total spend."""
        try:
            top_customers = self.df.groupby('Customer ID')['Revenue'].sum().nlargest(top_n)
            # Convert keys to int to remove decimals
            return {int(k): round(v, 2) for k, v in top_customers.items()}
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    def get_revenue_by_country(self, top_n=5):
        """8. Returns the top N countries by total revenue."""
        try:
            country_rev = self.df.groupby('Country')['Revenue'].sum().nlargest(top_n)
            return country_rev.round(2).to_dict()
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    def get_most_popular_product(self):
        """9. Finds the product description that sold the most units."""
        try:
            if 'Description' not in self.df.columns:
                return "Description column missing"
            popular_product = self.df.groupby('Description')['Quantity'].sum().idxmax()
            return popular_product
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    def get_refund_rate(self):
        """10. Calculates the percentage of transactions that were refunds (Quantity < 0)."""
        try:
            total_rows = len(self.df)
            refund_rows = len(self.df[self.df['Quantity'] < 0])
            rate = (refund_rows / total_rows) * 100
            return round(rate, 2)
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    # ==========================================
    # 🔴 LEVEL 3: HARD (Complex Grouping, Time Series, & Advanced Logic)
    # ==========================================

    def get_repeat_customer_rate(self):
        """11. Calculates the percentage of customers who made more than one purchase."""
        try:
            if 'InvoiceNo' not in self.df.columns:
                return "InvoiceNo column missing"
            
            # Count unique invoices per customer
            invoices_per_customer = self.df.groupby('Customer ID')['InvoiceNo'].nunique()
            repeat_customers = (invoices_per_customer > 1).sum()
            total_customers = invoices_per_customer.count()
            
            rate = (repeat_customers / total_customers) * 100
            return round(rate, 2)
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    def get_best_selling_product_per_country(self):
        """12. Finds the single best-selling product for each country based on quantity."""
        try:
            if 'Description' not in self.df.columns:
                return "Description column missing"
                
            # Group by Country and Product, sum quantity
            grouped = self.df.groupby(['Country', 'Description'])['Quantity'].sum().reset_index()
            # Find the index of the max quantity per country
            idx = grouped.groupby('Country')['Quantity'].idxmax()
            best_sellers = grouped.loc[idx].set_index('Country')['Description'].to_dict()
            return best_sellers
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    def get_average_order_value(self):
        """13. Calculates the Average Order Value (AOV) per invoice."""
        try:
            if 'InvoiceNo' not in self.df.columns:
                return "InvoiceNo column missing"
                
            # Sum revenue per invoice, then find the average of those sums
            revenue_per_invoice = self.df.groupby('InvoiceNo')['Revenue'].sum()
            # Remove negative/zero totals (refunds) for an accurate average
            valid_invoices = revenue_per_invoice[revenue_per_invoice > 0]
            
            return round(valid_invoices.mean(), 2)
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    def get_monthly_revenue_trend(self):
        """14. Returns a dictionary of total revenue per month."""
        try:
            if 'InvoiceDate' not in self.df.columns:
                return "InvoiceDate column missing"
                
            # Extract Year-Month (e.g., '2023-01')
            temp_df = self.df.dropna(subset=['InvoiceDate']).copy()
            temp_df['YearMonth'] = temp_df['InvoiceDate'].dt.to_period('M').astype(str)
            
            monthly_trend = temp_df.groupby('YearMonth')['Revenue'].sum()
            return monthly_trend.round(2).to_dict()
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None

    def get_high_value_loyal_customers(self, order_threshold=5, revenue_threshold=1000):
        """15. Finds 'VIP' customers who have ordered more than X times AND spent over Y amount."""
        try:
            if 'InvoiceNo' not in self.df.columns:
                return "InvoiceNo column missing"
                
            # Calculate total spend and total orders per customer
            customer_stats = self.df.groupby('Customer ID').agg(
                Total_Spend=('Revenue', 'sum'),
                Total_Orders=('InvoiceNo', 'nunique')
            )
            
            # Filter based on thresholds
            vips = customer_stats[
                (customer_stats['Total_Orders'] >= order_threshold) & 
                (customer_stats['Total_Spend'] >= revenue_threshold)
            ]
            
            # Return list of VIP customer IDs
            return [int(cid) for cid in vips.index.tolist()]
        except Exception as e:
            print(f"[Customer Analyst] ❌ Error: {e}")
            return None