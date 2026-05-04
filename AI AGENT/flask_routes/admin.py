import json
from pathlib import Path

from flask import Blueprint, render_template, session, redirect
from flask_routes.utils import login_required

admin_bp = Blueprint('admin', __name__)

_LOG_PATH = Path(__file__).parent.parent / 'data' / 'admin_log.json'

_ADMIN_USERS = ('Or', 'Taeir')


@admin_bp.route('/admin')
@login_required
def admin_panel():
    if session.get('username') not in _ADMIN_USERS:
        return redirect('/dashboard')
    entries = json.loads(_LOG_PATH.read_text('utf-8')) if _LOG_PATH.exists() else []
    entries = sorted(entries, key=lambda e: e.get('timestamp', ''), reverse=True)

    strategy_entries = [e for e in entries if e.get('event') == 'strategy_generated']

    from collections import Counter
    unique_emails   = len({e.get('email', '') for e in strategy_entries if e.get('email')})
    goal_counts     = Counter(e.get('goal_label', '') for e in strategy_entries if e.get('goal_label'))
    biz_counts      = Counter(e.get('business_type', '') for e in strategy_entries if e.get('business_type'))
    top_goal        = goal_counts.most_common(1)[0][0] if goal_counts else '—'
    top_biz         = biz_counts.most_common(1)[0][0] if biz_counts else '—'

    summary = {
        'total_strategies': len(strategy_entries),
        'unique_emails':    unique_emails,
        'top_goal':         top_goal,
        'top_biz':          top_biz,
    }
    return render_template('admin.html', entries=strategy_entries, summary=summary)
