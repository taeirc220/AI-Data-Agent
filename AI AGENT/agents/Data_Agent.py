import pandas as pd
import os
import sys


def _safe_print(msg: str) -> None:
    """Print with graceful fallback for terminals that don't support UTF-8 (e.g. Windows cp1252)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


class DataAgent:
    def __init__(self, file_path):
        # Resolve relative paths against this file's directory so the agent
        # works regardless of where the user launches the script from.
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        self.file_path = file_path

    def get_data(self):
        """Loads the CSV, cleans it, and returns a ready-to-use DataFrame. Returns None on failure."""
        if not os.path.exists(self.file_path):
            _safe_print(f"[Data Agent] ERROR: File not found at {self.file_path}")
            return None

        try:
            df = pd.read_csv(self.file_path, encoding='ISO-8859-1')

            rows_before = len(df)

            # Drop duplicates
            df.drop_duplicates(inplace=True)

            # Drop rows without a customer ID — can't analyze anonymous transactions
            df.dropna(subset=['Customer ID'], inplace=True)

            # Drop rows without a product description
            df.dropna(subset=['Description'], inplace=True)

            # Drop rows with zero or negative price
            df = df[df['Price'] > 0]

            # Pre-parse InvoiceDate so the executor sandbox doesn't re-parse on every call
            if 'InvoiceDate' in df.columns:
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

            rows_after = len(df)
            _safe_print(
                f"[Data Agent] Cleaned data: {rows_before - rows_after} rows removed "
                f"({rows_after} remaining)."
            )

            return df
        except Exception as e:
            _safe_print(f"[Data Agent] ERROR loading CSV: {e}")
            return None
