import pandas as pd
import numpy as np

class ProductAnalyst:
    def __init__(self, data_frame):
        self.df = data_frame.copy()
        # חישוב הכנסה בסיסי לכל שורה (Revenue)
        self.df['Revenue'] = self.df['Quantity'] * self.df['Price']
        if 'InvoiceDate' in self.df.columns:
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])

    # 1. סך יחידות שנמכרו לכל מוצר
    def get_total_products_sold(self):
        return self.df.groupby('Description')['Quantity'].sum().to_dict()

    # 2. הכנסה מפורטת לכל מוצר
    def get_product_revenue(self):
        return self.df.groupby('Description')['Revenue'].sum().to_dict()

    # 3. מחיר ממוצע בפועל למוצר (משוקלל לפי כמויות)
    def get_average_price_per_product(self):
        grouped = self.df.groupby('Description').agg({'Revenue': 'sum', 'Quantity': 'sum'})
        avg_price = grouped['Revenue'] / grouped['Quantity']
        return avg_price.to_dict()

    # 4. מגמת מכירות למוצר (השוואת תקופות)
    def get_product_sales_trend(self, product_desc):
        if 'InvoiceDate' not in self.df.columns: return "No date data"
        prod_data = self.df[self.df['Description'] == product_desc].set_index('InvoiceDate')
        monthly = prod_data['Quantity'].resample('ME').sum()
        if len(monthly) < 2: return "Not enough history"
        change = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2]) * 100
        return f"{'Up' if change > 0 else 'Down'} ({change:.2f}%)"

    # 5. מוצרים מובילים לפי הכנסה
    def get_top_products_by_revenue(self, limit=10):
        return self.df.groupby('Description')['Revenue'].sum().nlargest(limit).to_dict()

    # 6. מוצרים מובילים לפי כמות יחידות
    def get_top_products_by_quantity(self, limit=10):
        return self.df.groupby('Description')['Quantity'].sum().nlargest(limit).to_dict()

    # 7. אינדיקטור מלאי נמוך (סימולציה לפי קצב מכירות)
    def get_low_stock_indicator(self, threshold=20):
        # כיוון שאין עמודת מלאי, נזהה מוצרים שנמכרו בכמות גדולה לאחרונה ו"אולי" חסרים
        sold_counts = self.df.groupby('Description')['Quantity'].sum()
        low_stock = sold_counts[sold_counts > 500] # דוגמה ללוגיקה
        return "Feature requires real-time inventory data"

    # 8. שיעור המרה למוצר (Product Conversion Rate)
    def get_product_conversion_rate(self):
        return "Requires Session/Traffic data to calculate"

    # 9. אחוז החזרות למוצר
    def get_product_return_rate(self):
        returns = self.df[self.df['Quantity'] < 0].groupby('Description')['Quantity'].sum().abs()
        sales = self.df[self.df['Quantity'] > 0].groupby('Description')['Quantity'].sum()
        rate = (returns / sales) * 100
        return rate.dropna().to_dict()

    # 10. נתח הכנסות של מוצר מסך המכירות
    def get_product_revenue_share(self):
        total_rev = self.df['Revenue'].sum()
        share = (self.df.groupby('Description')['Revenue'].sum() / total_rev) * 100
        return share.nlargest(10).to_dict()

    # 11. קצב צמיחה למוצר
    def get_product_growth_rate(self, product_desc):
        return self.get_product_sales_trend(product_desc) # לוגיקה דומה

    # 12. מדד פופולריות משוקלל (כמות + מספר הזמנות שונות)
    def get_product_popularity_score(self):
        stats = self.df.groupby('Description').agg({'Quantity': 'sum', 'Invoice': 'nunique'})
        # נרמול פשוט: כמות * הזמנות ייחודיות
        stats['Score'] = stats['Quantity'] * stats['Invoice']
        return stats['Score'].nlargest(10).to_dict()

    # 13. הערכת רווח (Price - Cost)
    def get_product_profit_estimate(self):
        return "Requires Cost data (COGS) to calculate"

    # 14. תדירות רכישה (בכמה אחוז מההזמנות המוצר מופיע)
    def get_product_purchase_frequency(self):
        total_orders = self.df['Invoice'].nunique()
        freq = (self.df.groupby('Description')['Invoice'].nunique() / total_orders) * 100
        return freq.nlargest(10).to_dict()

    # 15. סטטוס מחזור חיי מוצר (Growing/Stable/Declining)
    def get_product_lifecycle_status(self, product_desc):
        if 'InvoiceDate' not in self.df.columns: return "Unknown"
        prod_data = self.df[self.df['Description'] == product_desc].set_index('InvoiceDate')
        monthly = prod_data['Quantity'].resample('ME').sum()
        if len(monthly) < 3: return "New Product"
        
        last_3_months = monthly.tail(3).values
        if last_3_months[2] > last_3_months[1] > last_3_months[0]: return "Growing"
        if last_3_months[2] < last_3_months[1] < last_3_months[0]: return "Declining"
        return "Stable"