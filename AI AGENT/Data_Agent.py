import pandas as pd
import os

class DataAgent:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_data(self):
        """
        טוען את הנתונים מהקובץ ומחזיר DataFrame של Pandas.
        אם הקובץ לא נמצא או שיש שגיאה, מחזיר None.
        """
        if not os.path.exists(self.file_path):
            print(f"[Data Agent] ❌ Error: File not found at {self.file_path}")
            return None
        
        try:
            # טעינה שקטה ללא הדפסות מיותרות
            df = pd.read_csv(self.file_path, encoding='ISO-8859-1')
            return df
        except Exception as e:
            print(f"[Data Agent] ❌ Error loading CSV: {e}")
            return None

#