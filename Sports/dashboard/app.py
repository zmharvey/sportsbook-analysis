"""
Flask application factory for the Sports Betting Dashboard
"""
import os
import sys
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from database import Database


def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(24)

    # Initialize database
    db = Database()

    @app.route('/')
    def index():
        """Stats dashboard home page"""
        return render_template('index.html')

    @app.route('/ev')
    def ev_plays():
        """EV plays page"""
        return render_template('ev.html')

    @app.route('/arb')
    def arb_plays():
        """Arbitrage plays page"""
        return render_template('arb.html')

    @app.route('/props')
    def prop_plays():
        """Player props page"""
        return render_template('props.html')

    @app.route('/browse-props')
    def browse_props():
        """Browse all available player props page"""
        return render_template('browse_props.html')

    @app.route('/underdog-props')
    def underdog_props():
        """Underdog props page - shows all Underdog props with EV, sorted by fair probability"""
        return render_template('underdog_props.html')

    @app.route('/live-plays')
    def live_plays():
        """Today's plays page - shows all plays from today separated into upcoming, live, and settled"""
        return render_template('live_plays.html')

    @app.route('/plays')
    def all_plays():
        """All historical plays page"""
        return render_template('plays.html')

    @app.route('/graph/<event_id>')
    def odds_graph(event_id):
        """Odds movement graph page"""
        market = request.args.get('market', 'h2h')
        outcome = request.args.get('outcome', '')
        return render_template('graph.html', event_id=event_id, market=market, outcome=outcome)

    @app.route('/prop-graph/<event_id>')
    def prop_odds_graph(event_id):
        """Player prop odds movement graph page"""
        player_name = request.args.get('player', '')
        prop_type = request.args.get('prop_type', '')
        line = request.args.get('line', '')
        return render_template('prop_graph.html', 
                               event_id=event_id, 
                               player_name=player_name, 
                               prop_type=prop_type, 
                               line=line)

    @app.route('/settings')
    def settings():
        """Settings page"""
        return render_template('settings.html')

    @app.route('/events')
    def events_browser():
        """Browse all active events/lines"""
        return render_template('events.html')

    # API endpoints
    @app.route('/api/stats')
    def api_stats():
        """Get summary statistics for dashboard home"""
        stats = db.get_play_stats()
        return jsonify(stats)

    @app.route('/api/ev-plays')
    def api_ev_plays():
        """Get current +EV plays"""
        sport = request.args.get('sport', '')
        min_ev = float(request.args.get('min_ev', 0))
        plays = db.get_recent_plays(limit=100)

        # Filter plays
        filtered = []
        for play in plays:
            if sport and play.get('sport', '') != sport:
                continue
            if play.get('ev_percent', 0) < min_ev:
                continue
            # Only show plays from last 24 hours that haven't started
            filtered.append(play)

        return jsonify(filtered)

    @app.route('/api/arb-plays')
    def api_arb_plays():
        """Get current arbitrage opportunities by scanning fresh data"""
        # Get recent odds from database and check for arbs
        arbs = db.get_recent_arb_opportunities()
        return jsonify(arbs)

    @app.route('/api/prop-plays')
    def api_prop_plays():
        """Get recent player prop plays"""
        sport = request.args.get('sport', '')
        prop_type = request.args.get('prop_type', '')
        min_ev = float(request.args.get('min_ev', 0))

        plays = db.get_recent_prop_plays(limit=200)

        # Filter plays
        filtered = []
        for play in plays:
            if sport and play.get('sport', '') != sport:
                continue
            if prop_type and play.get('prop_type', '') != prop_type:
                continue
            if play.get('sent_ev_percent', 0) < min_ev:
                continue
            filtered.append(play)

        return jsonify(filtered)

    @app.route('/api/all-props')
    def api_all_props():
        """Get all available player props (not just alerted plays)"""
        sport = request.args.get('sport', '')
        player_search = request.args.get('player', '')
        prop_type = request.args.get('prop_type', '')
        bookmaker = request.args.get('bookmaker', '')

        props = db.get_all_available_props(
            sport=sport if sport else None,
            player_search=player_search if player_search else None,
            prop_type=prop_type if prop_type else None,
            bookmaker=bookmaker if bookmaker else None
        )

        return jsonify(props)

    @app.route('/api/underdog-props')
    def api_underdog_props():
        """Get all Underdog props with fair probabilities and EV, sorted by fair probability"""
        props = db.get_underdog_props_with_ev()
        return jsonify(props)

    @app.route('/api/prop-stats')
    def api_prop_stats():
        """Get player prop statistics"""
        stats = db.get_prop_play_stats()
        return jsonify(stats)

    @app.route('/api/prop-play/<int:play_id>/result', methods=['POST'])
    def api_update_prop_result(play_id):
        """Update the result of a prop play"""
        data = request.get_json()
        result = data.get('result')  # 'win', 'loss', 'push'

        if result not in ['win', 'loss', 'push', None]:
            return jsonify({'success': False, 'message': 'Invalid result'}), 400

        try:
            profit = db.update_prop_play_result_and_profit(play_id, result)
            return jsonify({
                'success': True,
                'message': f'Result updated to {result}',
                'profit_units': profit
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

    @app.route('/api/prop-profit-history')
    def api_prop_profit_history():
        """Get cumulative profit history for props graphing"""
        history = db.get_prop_profit_history()
        return jsonify(history)

    @app.route('/api/combined-stats')
    def api_combined_stats():
        """Get combined stats for plays and props"""
        stats = db.get_combined_stats()
        return jsonify(stats)

    @app.route('/api/current-value')
    def api_current_value():
        """Get current value stats comparing sent odds to current odds (pre-game only)"""
        stats = db.get_current_value_stats()
        return jsonify(stats)

    @app.route('/api/clv-stats')
    def api_clv_stats():
        """Get CLV stats for games that have started"""
        stats = db.get_clv_stats()
        return jsonify(stats)

    @app.route('/api/clv-breakdown')
    def api_clv_breakdown():
        """Get CLV stats broken down by bookmaker, sport, play type, etc."""
        breakdown = db.get_clv_breakdown()
        return jsonify(breakdown)

    @app.route('/performance')
    def performance():
        """Performance page - shows CLV breakdowns by different dimensions"""
        return render_template('performance.html')

    @app.route('/api/live-plays')
    def api_live_plays():
        """Get only live plays (upcoming games) with search/filter"""
        search = request.args.get('search', '')
        type_filter = request.args.get('type', '')
        sport_filter = request.args.get('sport', '')

        plays = db.get_live_plays_combined(search=search, type_filter=type_filter, sport_filter=sport_filter)
        return jsonify(plays)

    @app.route('/api/all-plays')
    def api_all_plays():
        """Get all historical plays (EV + props) with search/pagination"""
        search = request.args.get('search', '')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))

        plays = db.get_all_plays_combined(search=search, page=page, per_page=per_page)
        return jsonify(plays)

    @app.route('/api/play/<play_type>/<int:play_id>')
    def api_get_play_combined(play_type, play_id):
        """Get a single play by type and ID"""
        play = db.get_play_by_id_combined(play_id, play_type)
        if play:
            return jsonify(play)
        return jsonify({'error': 'Play not found'}), 404

    @app.route('/api/play/<play_type>/<int:play_id>/result', methods=['POST'])
    def api_update_play_result_combined(play_type, play_id):
        """Update the result of any play type"""
        data = request.get_json()
        result = data.get('result')

        if result not in ['win', 'loss', 'push', None]:
            return jsonify({'success': False, 'message': 'Invalid result'}), 400

        try:
            profit = db.update_play_result_combined(play_id, play_type, result)
            return jsonify({
                'success': True,
                'message': f'Result updated to {result}',
                'profit_units': profit
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

    @app.route('/api/combined-profit-history')
    def api_combined_profit_history():
        """Get cumulative profit history for all plays"""
        history = db.get_combined_profit_history()
        return jsonify(history)

    @app.route('/api/odds-history/<event_id>')
    def api_odds_history(event_id):
        """Get odds history for graphing"""
        market = request.args.get('market', 'h2h')
        outcome = request.args.get('outcome', '')

        history = db.get_odds_history(event_id, market, outcome)
        return jsonify(history)

    @app.route('/api/prop-odds-history/<event_id>')
    def api_prop_odds_history(event_id):
        """Get prop odds history for graphing"""
        player_name = request.args.get('player', '')
        prop_type = request.args.get('prop_type', '')
        line = request.args.get('line', None)

        if line is not None:
            line = float(line)

        history = db.get_prop_odds_history(event_id, player_name, prop_type, line)
        return jsonify(history)

    @app.route('/api/events')
    def api_events():
        """Get list of events for browsing"""
        sport = request.args.get('sport', '')
        search = request.args.get('search', '')

        events = db.get_events(sport=sport, search=search)
        return jsonify(events)

    @app.route('/api/settings', methods=['GET'])
    def api_get_settings():
        """Get current settings"""
        try:
            settings = {
                'MIN_EV_PERCENT': getattr(config, 'MIN_EV_PERCENT', 5),
                'MAX_EV_PERCENT': getattr(config, 'MAX_EV_PERCENT', 20),
                'MIN_PROFIT_PERCENT': getattr(config, 'MIN_PROFIT_PERCENT', 1),
                'MAX_PROFIT_PERCENT': getattr(config, 'MAX_PROFIT_PERCENT', 20),
                'SPORTS': getattr(config, 'SPORTS', []),
                'INCLUDE_BOOKMAKERS': getattr(config, 'INCLUDE_BOOKMAKERS', []),
                'EXCLUDE_BOOKMAKERS': getattr(config, 'EXCLUDE_BOOKMAKERS', []),
                'DFS_BOOKMAKERS': getattr(config, 'DFS_BOOKMAKERS', []),
                'COLLECTOR_POLL_INTERVAL_PEAK': getattr(config, 'POLL_INTERVAL_PEAK', 300),
                'COLLECTOR_POLL_INTERVAL_OFFPEAK': getattr(config, 'POLL_INTERVAL_OFFPEAK', 1800),
                'PEAK_HOURS_START': getattr(config, 'PEAK_HOURS_START', 12),
                'PEAK_HOURS_END': getattr(config, 'PEAK_HOURS_END', 22),
                'ALERT_ON_ARB': getattr(config, 'ALERT_ON_ARB', True),
                'ALERT_ON_EV': getattr(config, 'ALERT_ON_EV', True),
                'ALERT_ON_PROP_EV': getattr(config, 'ALERT_ON_PROP_EV', True),
                'ENABLE_PLAYER_PROPS': getattr(config, 'ENABLE_PLAYER_PROPS', True),
                'PROP_HOURS_BEFORE_GAME': getattr(config, 'PROP_HOURS_BEFORE_GAME', 24),
                'MAX_PROP_EVENTS_PER_CYCLE': getattr(config, 'MAX_PROP_EVENTS_PER_CYCLE', 0),
                'DATA_RETENTION_DAYS': getattr(config, 'DATA_RETENTION_DAYS', 7),
                'SHARP_BOOKMAKER': getattr(config, 'SHARP_BOOKMAKER', 'pinnacle'),
                'DEVIG_METHOD': getattr(config, 'DEVIG_METHOD', 'power'),
            }
            return jsonify(settings)
        except Exception as e:
            import traceback
            return jsonify({'error': f'Failed to load settings: {str(e)}', 'traceback': traceback.format_exc()}), 500

    @app.route('/api/settings', methods=['POST'])
    def api_save_settings():
        """Save settings to config_overrides.json"""
        data = request.get_json()
        overrides_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config_overrides.json'
        )

        try:
            # Convert frontend keys to config keys
            config_data = {}
            for key, value in data.items():
                if key == 'COLLECTOR_POLL_INTERVAL_PEAK':
                    config_data['POLL_INTERVAL_PEAK'] = value
                elif key == 'COLLECTOR_POLL_INTERVAL_OFFPEAK':
                    config_data['POLL_INTERVAL_OFFPEAK'] = value
                elif key in ['PEAK_HOURS_START', 'PEAK_HOURS_END', 'MIN_EV_PERCENT', 'MAX_EV_PERCENT', 
                             'MIN_PROFIT_PERCENT', 'MAX_PROFIT_PERCENT', 'SHARP_BOOKMAKER', 'DEVIG_METHOD',
                             'ALERT_ON_EV', 'ALERT_ON_PROP_EV', 'ALERT_ON_ARB', 'ENABLE_PLAYER_PROPS',
                             'PROP_HOURS_BEFORE_GAME', 'MAX_PROP_EVENTS_PER_CYCLE', 'DATA_RETENTION_DAYS',
                             'SPORTS', 'INCLUDE_BOOKMAKERS', 'DFS_BOOKMAKERS']:
                    # These keys match config keys exactly, pass through
                    config_data[key] = value
                # Ignore any other keys that don't match config structure
            
            with open(overrides_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            return jsonify({'success': True, 'message': 'Settings saved. Restart collector to apply.'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

    @app.route('/api/collector-status')
    def api_collector_status():
        """Get collector status"""
        status = db.get_collector_status()
        return jsonify(status)

    @app.route('/api/upcoming-events')
    def api_upcoming_events():
        """Get all upcoming events that haven't started"""
        sport = request.args.get('sport', '')
        search = request.args.get('search', '')
        events = db.get_upcoming_events_with_odds(sport=sport, search=search)
        return jsonify(events)

    @app.route('/api/play/<int:play_id>/result', methods=['POST'])
    def api_update_play_result(play_id):
        """Update the result of a play"""
        data = request.get_json()
        result = data.get('result')  # 'win', 'loss', 'push'

        if result not in ['win', 'loss', 'push', None]:
            return jsonify({'success': False, 'message': 'Invalid result'}), 400

        try:
            profit = db.update_play_result_and_profit(play_id, result)
            return jsonify({
                'success': True,
                'message': f'Result updated to {result}',
                'profit_units': profit
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

    @app.route('/api/play/<int:play_id>')
    def api_get_play(play_id):
        """Get a single play by ID"""
        play = db.get_play_by_id(play_id)
        if play:
            return jsonify(play)
        return jsonify({'error': 'Play not found'}), 404

    @app.route('/api/profit-history')
    def api_profit_history():
        """Get cumulative profit history for graphing"""
        history = db.get_profit_history()
        return jsonify(history)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
