"""
Singleton loader for AI agents — shared across all Flask routes.

Data + SalesAnalyst load on first dashboard/KPI request (no API key, ~50 MB).
ManagerAgent loads on first chat/consultant request (LangChain, ~300 MB) so
the dashboard works immediately without triggering the heavy import.
"""
import os
import sys
import threading
import traceback

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_BASE, 'agents'))

_df    = None
_sales = None
_manager = None

_data_lock    = threading.Lock()
_manager_lock = threading.Lock()

_data_loaded    = False
_manager_loaded = False
_manager_error  = None


def get_data_agents():
    """Return (df, sales) — loads once on first call. No API key required."""
    global _df, _sales, _data_loaded

    if _data_loaded:
        return _df, _sales

    with _data_lock:
        if _data_loaded:
            return _df, _sales

        try:
            from Data_Agent import DataAgent
            from Sales_Analyst import SalesAnalyst

            csv_path = os.path.join(_BASE, "data", "online_retail_II_sampled.parquet")
            d_agent = DataAgent(csv_path)
            _df = d_agent.get_data()

            if _df is not None:
                _sales = SalesAnalyst(_df)
                print("[flask_agents] Data + SalesAnalyst loaded successfully.")
            else:
                print("[flask_agents] ERROR: DataAgent returned None — check data path.")

        except Exception:
            print("[flask_agents] ERROR loading data/SalesAnalyst:")
            traceback.print_exc()
            _df    = None
            _sales = None

        _data_loaded = True

    return _df, _sales


def get_manager():
    """Return ManagerAgent — loads once on first call. Requires OPENAI_API_KEY."""
    global _manager, _manager_loaded, _manager_error

    if _manager_loaded:
        return _manager

    with _manager_lock:
        if _manager_loaded:
            return _manager

        df, _ = get_data_agents()

        if df is not None:
            try:
                from Manager import ManagerAgent
                _manager = ManagerAgent(df)
                print("[flask_agents] ManagerAgent loaded successfully.")
            except Exception as e:
                _manager_error = str(e)
                print(f"[flask_agents] WARNING: ManagerAgent failed to load: {e}")
                traceback.print_exc()
                _manager = None

        _manager_loaded = True

    return _manager


def get_agents():
    """Legacy helper — returns (df, manager, sales). Loads everything."""
    df, sales = get_data_agents()
    manager   = get_manager()
    return df, manager, sales


def get_manager_error() -> str | None:
    return _manager_error
