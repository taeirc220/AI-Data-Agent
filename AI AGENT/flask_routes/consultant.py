from flask import Blueprint, render_template, session, request, jsonify
from flask_routes.utils import login_required

consultant_bp = Blueprint('consultant', __name__)


@consultant_bp.route('/consultant')
@login_required
def consultant():
    return render_template('consultant.html', username=session.get('username'))


@consultant_bp.route('/api/consultant/profile', methods=['POST'])
@login_required
def api_consultant_profile():
    data          = request.get_json(silent=True) or {}
    name          = data.get('name', '').strip()
    email         = data.get('email', '').strip()
    business_type = data.get('business_type', '').strip()

    if not name or not business_type:
        return jsonify({'error': 'Name and business type are required.'}), 400

    VALID_TYPES = {'e-commerce', 'retail', 'restaurant', 'services', 'other'}
    if business_type not in VALID_TYPES:
        return jsonify({'error': 'Invalid business type.'}), 400

    session['business_profile'] = {'name': name, 'email': email, 'business_type': business_type}
    return jsonify({'ok': True})


@consultant_bp.route('/api/consultant/profile', methods=['DELETE'])
@login_required
def api_consultant_profile_delete():
    session.pop('business_profile', None)
    return jsonify({'ok': True})


@consultant_bp.route('/api/consultant/health_preview')
@login_required
def api_consultant_health_preview():
    from flask_agents import get_manager
    manager = get_manager()

    if manager is None:
        return jsonify({'error': 'Data not available.'}), 503

    try:
        sales_analyst      = manager.sales_analyst
        customer_analyst   = manager.customer_analyst
        prediction_analyst = manager.prediction_analyst

        mom_data  = sales_analyst.get_mom_growth_rate()
        mom_val   = mom_data.get('latest', 0.0) if isinstance(mom_data, dict) else 0.0

        churn_data = prediction_analyst.get_churn_risk_summary()
        churn_pct  = churn_data.get('churn_risk_pct', 0.0) if isinstance(churn_data, dict) else 0.0

        repeat_pct = customer_analyst.get_repeat_customer_rate()

        def traffic_light(val, green_thresh, yellow_thresh, higher_is_better=True):
            if higher_is_better:
                if val >= green_thresh:  return 'green'
                if val >= yellow_thresh: return 'yellow'
                return 'red'
            else:
                if val <= green_thresh:  return 'green'
                if val <= yellow_thresh: return 'yellow'
                return 'red'

        return jsonify({
            'mom_growth': {
                'label':   'Monthly growth',
                'value':   round(mom_val, 1),
                'display': 'Growing' if mom_val > 5 else 'Stable' if mom_val > -5 else 'Declining',
                'color':   traffic_light(mom_val, 5, -5),
            },
            'churn_risk': {
                'label':   'Customers at risk',
                'value':   round(churn_pct, 1),
                'display': 'Low risk' if churn_pct < 30 else 'Moderate' if churn_pct < 55 else 'High risk',
                'color':   traffic_light(churn_pct, 30, 55, higher_is_better=False),
            },
            'repeat_rate': {
                'label':   'Repeat buyers',
                'value':   round(repeat_pct, 1),
                'display': 'Strong' if repeat_pct > 35 else 'Average' if repeat_pct > 20 else 'Needs work',
                'color':   traffic_light(repeat_pct, 35, 20),
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@consultant_bp.route('/api/consultant/analyze', methods=['POST'])
@login_required
def api_consultant_analyze():
    data      = request.get_json(silent=True) or {}
    goal      = data.get('goal', '').strip()
    target    = data.get('target', '').strip()
    timeframe = data.get('timeframe', 'the next 3 months').strip()

    if not goal:
        return jsonify({'error': 'No goal provided.'}), 400

    # Inject business profile context if available
    profile       = session.get('business_profile', {})
    profile_name  = profile.get('name', '')
    profile_type  = profile.get('business_type', '')

    # Build a rich natural-language prompt for Zyon
    parts = []
    if profile_name and profile_type:
        parts.append(
            f"[Context: You are speaking with {profile_name}, "
            f"who runs a {profile_type} business.]"
        )
    parts.append(f"My business goal is: {goal}.")
    if target:
        parts.append(f"My target is: {target}.")
    parts.append(f"I want to achieve this in {timeframe}.")
    parts.append(
        "Please analyse my business data thoroughly — call at least 5 different tools "
        "to look at this from multiple angles — then give me a clear, practical action plan "
        "in plain English. Tell me exactly what to do, why it matters, and how to do it."
    )
    prompt = " ".join(parts)

    from flask_routes.utils import resolve_manager
    manager = resolve_manager(session.get('session_id'))

    if manager is None:
        return jsonify({'error': 'Agent not available. Please check server logs.'}), 503

    try:
        for step in manager.handle_consultant_request(prompt):
            if step['type'] == 'result':
                return jsonify({
                    'response': step['content'],
                    'agent':    step.get('agent_label', 'Consultant (Zyon)'),
                })
        return jsonify({'error': 'No response generated.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@consultant_bp.route('/api/consultant/followup', methods=['POST'])
@login_required
def api_consultant_followup():
    data    = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])

    if not message:
        return jsonify({'error': 'Empty message.'}), 400
    if len(message) > 2000:
        return jsonify({'error': 'Message too long.'}), 400

    from flask_routes.utils import resolve_manager
    manager = resolve_manager(session.get('session_id'))

    if manager is None:
        return jsonify({'error': 'Agent not available.'}), 503

    try:
        for step in manager.handle_consultant_request(message, history=history):
            if step['type'] == 'result':
                return jsonify({
                    'response': step['content'],
                    'agent':    step.get('agent_label', 'Consultant (Zyon)'),
                })
        return jsonify({'error': 'No response.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
