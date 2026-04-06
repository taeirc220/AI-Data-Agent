import pandas as pd

class SalesAnalyst:
    def __init__(self, data_frame):
        # We store the dataframe here so the analyst can calculate metrics
        self.df = data_frame
    
    def get_total_sales(self):
        """
        Calculates total revenue by multiplying Quantity * Price.
        """
        try:
            total_revenue = (self.df['Quantity'] * self.df['Price']).sum()
            return total_revenue
        except Exception as e:
            print(f"[Sales Analyst] ❌ Error calculating sales: {e}")
            return None

    def get_average_order_value(self):
        """
             (AOV) by dividing total revenue by total unique orders.
        """
        try:
            total_revenue = (self.df['Quantity'] * self.df['Price']).sum()
            total_orders = self.df['Invoice'].nunique()
            
            # Protect against dividing by zero if there are no orders
            if total_orders == 0:
                return 0
                
            aov = total_revenue / total_orders
            return aov
        except Exception as e:
            print(f"[Sales Analyst] ❌ Error calculating AOV: {e}")
            return None