"""
Code_Executor.py — Sandboxed Python execution engine for the AI Data Agent.

Features:
- Persistent session namespace: variables defined in one call survive to the next
- Captures stdout, stderr, and return values
- Detects and captures matplotlib figures as base64-encoded PNGs
- Structured logging of every execution (input, output, duration, errors)
- Thread-safe chart buffer per executor instance
"""

import sys
import io
import time
import base64
import logging
import traceback

# Force non-interactive backend BEFORE any matplotlib import elsewhere
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logger = logging.getLogger("CodeExecutor")


# ---------------------------------------------------------------------------
# Chart style — applied once when the module loads
# ---------------------------------------------------------------------------
_STYLE_APPLIED = False

def _apply_chart_style() -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # Fall back to matplotlib defaults

    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.autolayout": True,
    })
    _STYLE_APPLIED = True


_apply_chart_style()


# ---------------------------------------------------------------------------
# CodeExecutor
# ---------------------------------------------------------------------------

class CodeExecutor:
    """
    Persistent Python execution sandbox for one agent session.

    Usage::

        executor = CodeExecutor(df)
        result = executor.execute("print(df.shape)")
        charts  = executor.get_pending_charts()   # list of base64 PNG strings
    """

    def __init__(self, df=None):
        self._charts: list[str] = []   # base64 PNGs waiting to be sent to UI
        self._namespace: dict = {}
        self._reset_namespace(df)

    # ------------------------------------------------------------------
    # Namespace management
    # ------------------------------------------------------------------

    def _reset_namespace(self, df) -> None:
        """Populate the execution namespace with standard libraries + the dataframe."""
        import pandas as pd
        import numpy as np

        self._namespace = {
            # Data
            "df": df,
            # Libraries
            "pd": pd,
            "np": np,
            "plt": plt,
        }
        try:
            import seaborn as sns
            self._namespace["sns"] = sns
        except ImportError:
            pass
        try:
            import scipy.stats as stats
            self._namespace["stats"] = stats
        except ImportError:
            pass

    def update_df(self, df) -> None:
        """Refresh the dataframe reference without losing other session variables."""
        self._namespace["df"] = df

    # ------------------------------------------------------------------
    # Chart retrieval
    # ------------------------------------------------------------------

    def get_pending_charts(self) -> list[str]:
        """
        Return all charts generated since the last call, then clear the buffer.
        Each item is a base64-encoded PNG string suitable for st.image() or <img>.
        """
        charts = self._charts.copy()
        self._charts.clear()
        return charts

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def execute(self, code: str) -> dict:
        """
        Execute *code* in the persistent namespace.

        Returns a dict::

            {
                "output":      str,   # everything printed to stdout/stderr
                "error":       str,   # exception message + traceback, or ""
                "charts":      list,  # base64 PNGs produced during this call
                "duration_ms": int,
                "success":     bool,
            }
        """
        _apply_chart_style()
        plt.close("all")  # start with a clean figure state

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        output = ""
        error = ""
        charts: list[str] = []

        start = time.perf_counter()

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        try:
            compiled = compile(code, "<analyst_code>", "exec")
            exec(compiled, self._namespace)  # noqa: S102
        except Exception:
            error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        output = stdout_buf.getvalue()
        stderr_out = stderr_buf.getvalue().strip()
        if stderr_out:
            output = (output + f"\n[warnings/stderr]:\n{stderr_out}").strip()

        # ---- Capture any matplotlib figures ----------------------------
        try:
            fig_nums = plt.get_fignums()
            if fig_nums:
                for fig_num in fig_nums:
                    fig = plt.figure(fig_num)
                    buf = io.BytesIO()
                    fig.savefig(
                        buf, format="png", bbox_inches="tight",
                        dpi=150, facecolor="white", edgecolor="none",
                    )
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode("utf-8")
                    charts.append(b64)
                    self._charts.append(b64)
                plt.close("all")
        except Exception as chart_err:
            logger.warning("[CodeExecutor] Chart capture error: %s", chart_err)

        duration_ms = int((time.perf_counter() - start) * 1000)

        logger.info(
            "[CodeExecutor] success=%s | %d ms | %d chart(s) | "
            "code_preview=%r | error=%s",
            not bool(error), duration_ms, len(charts),
            (code[:80] + "...") if len(code) > 80 else code,
            (error[:120] + "...") if len(error) > 120 else (error or "none"),
        )

        return {
            "output": output.strip(),
            "error": error,
            "charts": charts,
            "duration_ms": duration_ms,
            "success": not bool(error),
        }
