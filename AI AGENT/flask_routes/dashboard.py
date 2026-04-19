from flask import Blueprint, render_template, session, jsonify
from flask_routes.utils import login_required

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/dashboard')
@login_required
def main():
    return render_template('dashboard.html', username=session.get('username'))


@dashboard_bp.route('/api/health')
def api_health():
    """Diagnostic endpoint — shows what loaded successfully on startup."""
    from flask_agents import get_agents, get_manager_error
    df, manager, sales = get_agents()
    return jsonify({
        'data_loaded':    df is not None,
        'manager_loaded': manager is not None,
        'sales_loaded':   sales is not None,
        'records':        len(df) if df is not None else 0,
        'manager_error':  get_manager_error(),
    })


@dashboard_bp.route('/api/kpis')
@login_required
def api_kpis():
    from flask_agents import get_agents
    df, manager, sales = get_agents()

    if df is None:
        return jsonify({'error': 'Data not loaded — check server logs for details.'}), 503

    try:
        mom_data = sales.get_mom_growth_rate()
        mom = mom_data.get('latest', 0.0) if isinstance(mom_data, dict) else float(mom_data or 0)
        return jsonify({
            'total_revenue': sales.get_total_revenue(),
            'total_orders':  sales.get_total_orders(),
            'total_items':   sales.get_total_items_sold(),
            'aov':           sales.get_average_order_value(),
            'refund_rate':   sales.get_refund_rate(),
            'mom_growth':    mom,
            'records':       len(df),
        })
    except Exception as e:
        return jsonify({'error': f'KPI calculation failed: {e}'}), 500


@dashboard_bp.route('/api/charts')
@login_required
def api_charts():
    from flask_agents import get_agents
    df, manager, sales = get_agents()

    if df is None:
        return jsonify({'error': 'Data not loaded — check server logs for details.'}), 503

    try:
        return jsonify({
            'monthly_revenue': sales.get_monthly_revenue(),
            'top_countries':   sales.get_top_countries_by_revenue(limit=5),
            'top_products':    sales.get_top_products_by_revenue(limit=8),
            'hourly_sales':    sales.get_hourly_sales_distribution(),
        })
    except Exception as e:
        return jsonify({'error': f'Chart data failed: {e}'}), 500
