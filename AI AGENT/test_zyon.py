"""
test_zyon.py — Zyon Consultant Agent test suite

Evaluation framework:
  1. Tired Owner Lens    — 30-sec attention span, zero data background
  2. Zero Jargon         — 19 banned terms; any hit = auto-fail
  3. Actionability       — specific WHO/WHAT/WHEN, not "consider optimising"
  4. So What?            — every insight links to £ in / £ out / time saved

Usage:
    pip install requests colorama
    python test_zyon.py          # Flask server must be running on port 5001
"""

import json
import re
import sys
import time
from datetime import datetime

import requests

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    # Fallback: no colours
    class _NoColour:
        def __getattr__(self, _): return ""
    Fore = Style = _NoColour()

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL          = "http://127.0.0.1:5001"
ANALYZE_URL       = f"{BASE_URL}/api/consultant/analyze"
FOLLOWUP_URL      = f"{BASE_URL}/api/consultant/followup"
RESULTS_FILE      = f"zyon_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
REQUEST_TIMEOUT   = 300   # seconds — LLM calls can be slow
PAUSE_BETWEEN     = 4     # seconds between scenarios

# ── Banned terms (exact strings from the system prompt) ───────────────────────

BANNED_TERMS = [
    "MoM", "RFM", "AOV", "CLV", "SKU", "Silhouette", "Pareto", "cohort",
    "segmentation", "clustering", "churn rate %", "FP-Growth",
    "p-value", "percentile", "median", "standard deviation", "regression",
]
# Shorter terms that need word-boundary matching to avoid false positives
BANNED_TERMS_BOUNDARY = [" support ", " confidence ", " lift "]

# ── Named strategic levers ────────────────────────────────────────────────────

STRATEGIC_LEVERS = [
    "Cash Flow Triage", "Break-Even Analysis", "Inventory Liquidation",
    "Win-Back Campaign", "Dormant Customer Reactivation",
    "High-Value Customer Retention Sequence",
    "Price Elasticity Test", "Bundle Pricing Strategy",
    "Cross-Sell Activation", "Average Order Value Lift",
    "Star Product Doubling", "Slow Mover Clearance",
    "High-Refund Product Audit", "Revenue Concentration Reduction",
    "Customer Base Diversification", "Repeat Buyer Conversion",
    "First-Time Buyer Onboarding Sequence",
]

# ── Forbidden multi-person references ────────────────────────────────────────

TEAM_REFS = [
    "your marketing team", "your sales rep", "your operations team",
    "your finance team", "your sales team", "your support team",
    "your staff", "your employees",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_prompt(goal: str, target: str = "", timeframe: str = "the next 3 months") -> str:
    """Reconstruct the exact prompt that consultant.py builds (for history entries)."""
    parts = [f"My business goal is: {goal}."]
    if target:
        parts.append(f"My target is: {target}.")
    parts.append(f"I want to achieve this in {timeframe}.")
    parts.append(
        "Please analyse my business data thoroughly — call at least 5 different tools "
        "to look at this from multiple angles — then give me a clear, practical action plan "
        "in plain English. Tell me exactly what to do, why it matters, and how to do it."
    )
    return " ".join(parts)


def post_analyze(goal: str, target: str = "", timeframe: str = "the next 3 months"):
    """POST /api/consultant/analyze. Returns (response_text, built_prompt)."""
    payload  = {"goal": goal, "target": target, "timeframe": timeframe}
    r        = requests.post(ANALYZE_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data     = r.json()
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    return data.get("response", ""), _build_prompt(goal, target, timeframe)


def post_followup(message: str, history: list):
    """POST /api/consultant/followup. Returns response_text."""
    payload = {"message": message, "history": history}
    r       = requests.post(FOLLOWUP_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data    = r.json()
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    return data.get("response", "")


def check_jargon(text: str) -> list:
    found = [t for t in BANNED_TERMS if t in text]
    found += [t.strip() for t in BANNED_TERMS_BOUNDARY if t in text]
    return found


def check_lever(text: str) -> bool:
    return any(lever in text for lever in STRATEGIC_LEVERS)


def check_team_refs(text: str) -> list:
    return [r for r in TEAM_REFS if r.lower() in text.lower()]


def check_pound(text: str) -> bool:
    return "£" in text


def check_time_component(text: str) -> bool:
    return bool(re.search(
        r"(by \w+day|this week|within \d+ (days?|weeks?)|by Friday|next week|"
        r"over the next|in the next \d+|this month|today)",
        text, re.IGNORECASE
    ))


def check_human_pct(text: str) -> bool:
    return bool(re.search(r"\d+ in \d+|about \d+ in|out of every \d+|1 in \d+", text))


def detect_mode(text: str) -> str:
    if "## Your business is facing a serious problem" in text or \
       "Before I can advise you" in text:
        return "CRISIS"
    if "One question before you start" in text:
        return "WARNING"
    if "## What's happening in your business right now" in text:
        return "HEALTHY"
    return "UNKNOWN"


def word_count(text: str) -> int:
    return len(text.split())


def score_result(checks: dict) -> int:
    """Score 0–5 from a dict of {label: bool}. jargon_found=True is an instant 1."""
    if checks.get("server_error"):
        return 0
    if checks.get("jargon_found"):
        return 1
    positives = [v for v in checks.values() if v is True]
    total     = [v for v in checks.values() if isinstance(v, bool)]
    if not total:
        return 0
    ratio = len(positives) / len(total)
    if ratio >= 0.90: return 5
    if ratio >= 0.75: return 4
    if ratio >= 0.50: return 3
    if ratio >= 0.25: return 2
    return 1


def _sep(title: str):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Style.RESET_ALL}")


def _ok(label: str, value):
    colour = Fore.GREEN if value else Fore.RED
    mark   = "[+]" if value else "[-]"
    print(f"  {colour}{mark}{Style.RESET_ALL} {label}")


# ── Test scenarios ────────────────────────────────────────────────────────────

def run_preflight() -> dict:
    """Quick health-check call to determine the live data mode."""
    _sep("PRE-FLIGHT: Business health check")
    text, _ = post_analyze(
        goal="Give me a health check of my business",
        target="",
        timeframe="the next month"
    )
    mode = detect_mode(text)
    wc   = word_count(text)
    print(f"  Detected mode : {Fore.YELLOW}{mode}{Style.RESET_ALL}")
    print(f"  Word count    : {wc}")
    print(f"  Has £         : {check_pound(text)}")
    return {"scenario": "PREFLIGHT", "mode": mode, "word_count": wc,
            "response": text, "score": None}


def run_t1() -> dict:
    _sep("T1: Grow Revenue")
    text, _ = post_analyze("I want to grow my revenue",
                            "increase sales by 20%", "the next 3 months")
    mode    = detect_mode(text)
    jargon  = check_jargon(text)
    wc      = word_count(text)
    word_ok = (wc <= 900 if mode == "WARNING" else
               wc <= 600 if mode == "HEALTHY" else True)

    checks = {
        "jargon_found":      bool(jargon),
        "jargon_clean":      not bool(jargon),
        "lever_present":     check_lever(text),
        "pound_sign":        check_pound(text),
        "word_limit_ok":     word_ok,
        "no_team_refs":      not bool(check_team_refs(text)),
        "time_component":    check_time_component(text),
        "has_section_head":  "##" in text,
    }

    s = score_result(checks)
    print(f"  Mode: {mode}  |  Words: {wc}  |  Score: {s}/5")
    for label, val in checks.items():
        if label != "jargon_found":
            _ok(label, val)
    if jargon:
        print(f"  {Fore.RED}JARGON: {jargon}{Style.RESET_ALL}")
    return {"scenario": "T1", "mode": mode, "word_count": wc,
            "checks": checks, "score": s, "jargon": jargon, "response": text}


def run_t2() -> dict:
    _sep("T2: Get More Customers")
    text, _ = post_analyze("I want to get more customers",
                            "double my customer base", "6 months")
    mode   = detect_mode(text)
    jargon = check_jargon(text)
    teams  = check_team_refs(text)

    checks = {
        "jargon_found":       bool(jargon),
        "jargon_clean":       not bool(jargon),
        "no_team_refs":       not bool(teams),
        "lever_present":      check_lever(text),
        "pound_sign":         check_pound(text),
        "acquisition_lever":  any(l in text for l in [
            "Repeat Buyer Conversion",
            "First-Time Buyer Onboarding Sequence",
            "Customer Base Diversification",
        ]),
        "has_number":         bool(re.search(r"\d+", text)),
        "time_component":     check_time_component(text),
    }

    s = score_result(checks)
    print(f"  Mode: {mode}  |  Score: {s}/5")
    for label, val in checks.items():
        if label != "jargon_found":
            _ok(label, val)
    if jargon: print(f"  {Fore.RED}JARGON: {jargon}{Style.RESET_ALL}")
    if teams:  print(f"  {Fore.RED}TEAM REFS: {teams}{Style.RESET_ALL}")
    return {"scenario": "T2", "mode": mode, "checks": checks, "score": s,
            "jargon": jargon, "team_refs": teams, "response": text}


def run_t3() -> dict:
    _sep("T3: Stop Losing Customers")
    text, _ = post_analyze("I want to stop losing customers",
                            "reduce customer drop-off", "the next 2 months")
    mode   = detect_mode(text)
    jargon = check_jargon(text)

    checks = {
        "jargon_found":         bool(jargon),
        "jargon_clean":         not bool(jargon),
        "human_percentage":     check_human_pct(text),
        "no_churn_rate_pct":    "churn rate %" not in text,
        "no_cohort":            "cohort" not in text,
        "retention_lever":      any(l in text for l in [
            "Win-Back Campaign", "Dormant Customer Reactivation",
            "High-Value Customer Retention Sequence",
        ]),
        "pound_sign":           check_pound(text),
        "revenue_recovery_est": bool(re.search(
            r"recover|win back|reclaim|bring back.*£|£.*(back|recover)",
            text, re.IGNORECASE
        )),
    }

    s = score_result(checks)
    print(f"  Mode: {mode}  |  Score: {s}/5")
    for label, val in checks.items():
        if label != "jargon_found":
            _ok(label, val)
    if jargon: print(f"  {Fore.RED}JARGON: {jargon}{Style.RESET_ALL}")
    return {"scenario": "T3", "mode": mode, "checks": checks, "score": s,
            "jargon": jargon, "response": text}


def run_t4() -> dict:
    _sep("T4: Sell Stock Smarter")
    text, _ = post_analyze(
        "I want to sell my stock smarter",
        "reduce slow-moving products and increase average spend per order",
        "the next 3 months"
    )
    mode   = detect_mode(text)
    jargon = check_jargon(text)

    checks = {
        "jargon_found":         bool(jargon),
        "jargon_clean":         not bool(jargon),
        "no_aov":               "AOV" not in text,
        "no_sku":               "SKU" not in text,
        "no_fpgrowth":          "FP-Growth" not in text,
        "product_lever":        any(l in text for l in [
            "Bundle Pricing Strategy", "Cross-Sell Activation",
            "Average Order Value Lift", "Slow Mover Clearance",
            "Star Product Doubling",
        ]),
        "pound_sign":           check_pound(text),
        "avg_order_explained":  bool(re.search(
            r"(spend|spending).{0,30}(on average|per order|each time|each visit)|"
            r"average.{0,20}(spend|order|purchase|each)",
            text, re.IGNORECASE
        )),
    }

    s = score_result(checks)
    print(f"  Mode: {mode}  |  Score: {s}/5")
    for label, val in checks.items():
        if label != "jargon_found":
            _ok(label, val)
    if jargon: print(f"  {Fore.RED}JARGON: {jargon}{Style.RESET_ALL}")
    return {"scenario": "T4", "mode": mode, "checks": checks, "score": s,
            "jargon": jargon, "response": text}


def run_t5() -> dict:
    _sep("T5: Follow-Up Context Memory")
    # Step 5a
    text_a, prompt_a = post_analyze("I want to grow my revenue",
                                    "increase sales", "3 months")
    history = [
        {"role": "user",      "content": prompt_a},
        {"role": "assistant", "content": text_a},
    ]
    time.sleep(3)

    # Step 5b
    msg_b = (
        "My top products are candles and Christmas decorations. "
        "I mostly sell in December. Can you give me more specific advice based on that?"
    )
    text_b = post_followup(msg_b, history)
    jargon_b = check_jargon(text_b)

    checks = {
        "jargon_found":        bool(jargon_b),
        "jargon_clean":        not bool(jargon_b),
        "context_used":        any(w in text_b.lower() for w in
                                   ["candle", "christmas", "december", "seasonal"]),
        "no_re_ask_products":  "what products do you sell" not in text_b.lower(),
        "no_diagnostic_block": "Before I can advise you" not in text_b,
        "pound_sign":          check_pound(text_b),
        "lever_present":       check_lever(text_b),
        "has_content":         word_count(text_b) > 100,
    }

    s = score_result(checks)
    print(f"  5b Mode: {detect_mode(text_b)}  |  Score: {s}/5")
    for label, val in checks.items():
        if label != "jargon_found":
            _ok(label, val)
    if jargon_b: print(f"  {Fore.RED}JARGON: {jargon_b}{Style.RESET_ALL}")
    return {"scenario": "T5", "checks": checks, "score": s,
            "jargon_b": jargon_b, "response_a": text_a, "response_b": text_b}


def run_t6() -> dict:
    _sep("T6: Jargon Probe (CLV / churn rate / RFM)")
    text   = post_followup(
        "What is my CLV? And what is my churn rate? And can you show me an RFM breakdown?",
        history=[]
    )
    mode   = detect_mode(text)
    jargon = check_jargon(text)

    def term_translated(t: str, keywords: list) -> bool:
        idx = text.find(t)
        if idx == -1:
            return True   # term not mentioned at all — not a fail
        window = text[max(0, idx - 80): idx + 200]
        return any(k.lower() in window.lower() for k in keywords)

    clv_ok   = term_translated("CLV",
                   ["worth to your", "how much", "lifetime", "over time", "value of"])
    rfm_ok   = term_translated("RFM",
                   ["recently", "often", "how frequently", "how often", "grouping",
                    "how recently", "frequency", "spent"])

    checks = {
        "jargon_found":      bool(jargon),
        "jargon_clean":      not bool(jargon),
        "clv_explained":     clv_ok,
        "rfm_explained":     rfm_ok,
        "no_raw_churn_pct":  "churn rate %" not in text,
        "pound_sign":        check_pound(text),
        "has_numbers":       bool(re.search(r"\d+", text)),
        "not_refusal":       word_count(text) > 150,
    }

    s = score_result(checks)
    print(f"  Mode: {mode}  |  Score: {s}/5")
    for label, val in checks.items():
        if label != "jargon_found":
            _ok(label, val)
    if jargon: print(f"  {Fore.RED}JARGON: {jargon}{Style.RESET_ALL}")
    return {"scenario": "T6", "mode": mode, "checks": checks, "score": s,
            "jargon": jargon, "response": text}


def run_t7() -> dict:
    _sep("T7: Diagnostic / CRISIS Follow-Up")
    # Step 7a
    text_a, prompt_a = post_analyze(
        "My sales have collapsed and I need help urgently",
        "understand what is happening",
        "immediately"
    )
    mode_a  = detect_mode(text_a)
    jargon_a = check_jargon(text_a)

    if mode_a == "CRISIS":
        checks_a = {
            "jargon_found":            bool(jargon_a),
            "jargon_clean":            not bool(jargon_a),
            "has_diagnostic_section":  "Before I can advise you" in text_a,
            "no_action_plan_yet":      "## Your Immediate Actions" not in text_a,
            "no_marketing_tactics":    not any(t in text_a.lower() for t in [
                "discount code", "loyalty programme", "thank you email",
                "promotional campaign", "run a campaign",
            ]),
        }
    else:
        print(f"  {Fore.YELLOW}Note: mode is {mode_a}, not CRISIS. "
              f"Testing WARNING/HEALTHY diagnostic quality.{Style.RESET_ALL}")
        checks_a = {
            "jargon_found":   bool(jargon_a),
            "jargon_clean":   not bool(jargon_a),
            "lever_present":  check_lever(text_a),
            "pound_sign":     check_pound(text_a),
            "has_diagnostic": bool(re.search(r"question|before|tell me|what (has|happened)",
                                             text_a, re.IGNORECASE)),
        }

    history = [
        {"role": "user",      "content": prompt_a},
        {"role": "assistant", "content": text_a},
    ]
    time.sleep(3)

    # Step 7b — provide diagnostic answers
    msg_b = (
        "My main sales channel is fine, nothing has changed there. "
        "But I did lose my two biggest wholesale customers last month — "
        "they both moved to a competitor who was cheaper. I still have my retail customers."
    )
    text_b  = post_followup(msg_b, history)
    mode_b  = detect_mode(text_b)
    jargon_b = check_jargon(text_b)

    checks_b = {
        "jargon_found":           bool(jargon_b),
        "jargon_clean":           not bool(jargon_b),
        "no_repeat_channel_q":    "has your main sales channel" not in text_b.lower(),
        "wholesale_context_used": any(w in text_b.lower() for w in
                                      ["wholesale", "large account", "b2b",
                                       "biggest customer", "two customers",
                                       "major customer"]),
        "has_actions_now":        check_lever(text_b) or
                                  "## Your Immediate Actions" in text_b or
                                  "## Your Action" in text_b,
        "pound_sign":             check_pound(text_b),
        "no_generic_ads":         not any(t in text_b.lower() for t in [
            "social media campaign", "run ads", "advertising budget",
            "paid advertising",
        ]),
    }

    s_a = score_result(checks_a)
    s_b = score_result(checks_b)
    s   = (s_a + s_b) // 2

    print(f"  7a mode: {mode_a}  |  7a score: {s_a}/5")
    print(f"  7b mode: {mode_b}  |  7b score: {s_b}/5")
    print(f"  Combined: {s}/5")
    print(f"\n  7a checks:")
    for label, val in checks_a.items():
        if label != "jargon_found":
            _ok(f"  {label}", val)
    print(f"\n  7b checks:")
    for label, val in checks_b.items():
        if label != "jargon_found":
            _ok(f"  {label}", val)
    if jargon_a: print(f"  {Fore.RED}7a JARGON: {jargon_a}{Style.RESET_ALL}")
    if jargon_b: print(f"  {Fore.RED}7b JARGON: {jargon_b}{Style.RESET_ALL}")

    return {"scenario": "T7", "mode_a": mode_a, "mode_b": mode_b,
            "checks_a": checks_a, "checks_b": checks_b,
            "score_a": s_a, "score_b": s_b, "score": s,
            "jargon_a": jargon_a, "jargon_b": jargon_b,
            "response_a": text_a, "response_b": text_b}


def run_t8() -> dict:
    _sep("T8: Adversarial Pushback + Vague Input")
    # Step 8a — baseline
    text_a, prompt_a = post_analyze(
        "I want to get more repeat customers",
        "increase repeat purchase rate",
        "2 months"
    )
    history = [
        {"role": "user",      "content": prompt_a},
        {"role": "assistant", "content": text_a},
    ]
    time.sleep(3)

    # Step 8b — adversarial
    pushback = (
        "That won't work for me. I have absolutely no money to spend on any of this. "
        "I can't send emails, I don't have a mailing list, and I work 80 hours a week "
        "already so I don't have time either."
    )
    text_b  = post_followup(pushback, history)
    jargon_b = check_jargon(text_b)

    checks_b = {
        "jargon_found":          bool(jargon_b),
        "jargon_clean":          not bool(jargon_b),
        "no_email_campaign":     "email campaign" not in text_b.lower() and
                                 "mailing list" not in text_b.lower(),
        "no_paid_advertising":   not any(t in text_b.lower() for t in
                                         ["advertising", "run ads", "paid campaign"]),
        "zero_cost_action":      any(t in text_b.lower() for t in [
            "free", "no cost", "zero cost", "without spending", "handwritten",
            "personal", "phone call", "direct message", "include a note",
            "packaging", "product description", "rename", "face to face",
            "in person", "word of mouth",
        ]),
        "acknowledges_constraint": any(t in text_b.lower() for t in [
            "no money", "no budget", "understand", "limited", "tight",
            "no time", "80 hours", "without spending",
        ]),
        "not_repeating_verbatim": text_b.strip() != text_a.strip(),
        "pound_sign":            check_pound(text_b),
    }

    time.sleep(3)

    # Step 8c — vague input, empty history
    text_c  = post_followup("just help me", history=[])
    jargon_c = check_jargon(text_c)

    checks_c = {
        "jargon_found":    bool(jargon_c),
        "jargon_clean":    not bool(jargon_c),
        "not_refusal":     word_count(text_c) > 250,
        "lever_present":   check_lever(text_c),
        "pound_sign":      check_pound(text_c),
        "has_structure":   "##" in text_c,
        "no_only_question": not (
            text_c.strip().endswith("?") and word_count(text_c) < 80
        ),
    }

    s_b = score_result(checks_b)
    s_c = score_result(checks_c)
    s   = (s_b + s_c) // 2

    print(f"  8b score: {s_b}/5  |  8c score: {s_c}/5  |  Combined: {s}/5")
    print(f"\n  8b checks (adversarial):")
    for label, val in checks_b.items():
        if label != "jargon_found":
            _ok(f"  {label}", val)
    print(f"\n  8c checks (vague):")
    for label, val in checks_c.items():
        if label != "jargon_found":
            _ok(f"  {label}", val)
    if jargon_b: print(f"  {Fore.RED}8b JARGON: {jargon_b}{Style.RESET_ALL}")
    if jargon_c: print(f"  {Fore.RED}8c JARGON: {jargon_c}{Style.RESET_ALL}")

    return {"scenario": "T8", "checks_b": checks_b, "checks_c": checks_c,
            "score_b": s_b, "score_c": s_c, "score": s,
            "jargon_b": jargon_b, "jargon_c": jargon_c,
            "response_a": text_a, "response_b": text_b, "response_c": text_c}


# ── Main runner ───────────────────────────────────────────────────────────────

def main():
    print(f"\n{Fore.GREEN}{'='*60}")
    print("  ZYON CONSULTANT AGENT — EVALUATION SUITE")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(f"  Server : {BASE_URL}")
    print(f"  Results: {RESULTS_FILE}")

    # Verify server is reachable
    try:
        r = requests.get(BASE_URL, timeout=10)
        print(f"  Status : {r.status_code} OK\n")
    except Exception as e:
        print(f"\n{Fore.RED}Cannot reach server at {BASE_URL}: {e}{Style.RESET_ALL}")
        print("Start the server first:  python flask_app.py")
        sys.exit(1)

    results = []

    # Pre-flight
    try:
        pf = run_preflight()
        results.append(pf)
    except Exception as e:
        print(f"{Fore.RED}Pre-flight failed: {e}{Style.RESET_ALL}")

    scenarios = [
        ("T1: Grow Revenue",           run_t1),
        ("T2: Get More Customers",     run_t2),
        ("T3: Stop Losing Customers",  run_t3),
        ("T4: Sell Stock Smarter",     run_t4),
        ("T5: Follow-Up Memory",       run_t5),
        ("T6: Jargon Probe",           run_t6),
        ("T7: Diagnostic / CRISIS",    run_t7),
        ("T8: Adversarial + Vague",    run_t8),
    ]

    for name, fn in scenarios:
        try:
            result = fn()
            results.append(result)
        except Exception as e:
            print(f"\n{Fore.RED}ERROR in {name}: {e}{Style.RESET_ALL}")
            results.append({"scenario": name, "score": 0,
                             "error": str(e), "server_error": True})
        time.sleep(PAUSE_BETWEEN)

    # ── Summary ──────────────────────────────────────────────────────────────
    scored = [r for r in results if r.get("score") is not None]
    total  = sum(r["score"] for r in scored)
    max_s  = len(scored) * 5
    pct    = (total / max_s * 100) if max_s else 0

    print(f"\n{Fore.GREEN}{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}{Style.RESET_ALL}")

    for r in results:
        sc = r.get("score")
        if sc is None:
            continue
        colour = Fore.GREEN if sc >= 4 else Fore.YELLOW if sc == 3 else Fore.RED
        name   = r.get("scenario", "?")
        print(f"  {name:<35} {colour}{sc}/5{Style.RESET_ALL}")

    overall_colour = Fore.GREEN if pct >= 75 else Fore.RED
    label          = "PASS" if pct >= 75 else "FAIL"
    print(f"\n  Overall : {total}/{max_s}  ({pct:.1f}%)  "
          f"[{overall_colour}{label}{Style.RESET_ALL}]")
    print(f"  Pass threshold: 75% ({int(max_s * 0.75)}/{max_s})")

    # Hard blocker checks
    print(f"\n{Fore.YELLOW}Hard blockers:{Style.RESET_ALL}")
    blockers = []

    zeros = [r for r in results if r.get("score") == 0]
    if zeros:
        msg = f"Server errors in: {[r['scenario'] for r in zeros]}"
        print(f"  {Fore.RED}BLOCKER: {msg}{Style.RESET_ALL}")
        blockers.append(msg)

    t6 = next((r for r in results if r.get("scenario") == "T6"), None)
    if t6 and t6.get("score", 5) < 3:
        msg = f"T6 Jargon Probe scored {t6['score']}/5 — jargon compliance failed"
        print(f"  {Fore.RED}BLOCKER: {msg}{Style.RESET_ALL}")
        blockers.append(msg)

    t7 = next((r for r in results if r.get("scenario") == "T7"), None)
    if t7 and t7.get("mode_a") == "CRISIS":
        if not t7.get("checks_a", {}).get("has_diagnostic_section"):
            msg = "T7 CRISIS mode did not produce diagnostic questions before action plan"
            print(f"  {Fore.RED}BLOCKER: {msg}{Style.RESET_ALL}")
            blockers.append(msg)

    if not blockers:
        print(f"  {Fore.GREEN}None — no hard blockers triggered{Style.RESET_ALL}")

    # ── Human review reminder ─────────────────────────────────────────────────
    print(f"\n{Fore.YELLOW}Human review required for:{Style.RESET_ALL}")
    print("  - Tired Owner Lens   : does the first 2-3 sentences give an instant verdict?")
    print("  - So What? quality   : does each insight link to GBP in / GBP out / time saved?")
    print("  - T4 product pairs   : are specific product names mentioned for bundling?")
    print("  - T5b specificity    : is the follow-up more targeted than T5a?")
    print("  - T7b causal fit     : is the advice genuinely tailored to wholesale loss?")
    print("  - T8b zero-cost feel : are the alternatives actually doable with no budget?")

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Full responses saved to: {Fore.CYAN}{RESULTS_FILE}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
