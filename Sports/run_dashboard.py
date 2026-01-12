#!/usr/bin/env python3
"""
Sports Betting Dashboard - Entry Point

Run this script to start the dashboard web server.

Usage:
    python run_dashboard.py [--host HOST] [--port PORT] [--debug]

Options:
    --host      Host to bind to (default: 127.0.0.1)
    --port      Port to bind to (default: 5000)
    --debug     Enable debug mode with auto-reload
"""

import argparse
import webbrowser
import threading
import time
from dashboard.app import create_app


def open_browser(port: int):
    """Open browser after a short delay to allow server to start."""
    time.sleep(1.5)
    webbrowser.open(f'http://127.0.0.1:{port}')


def main():
    parser = argparse.ArgumentParser(description='Sports Betting Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    args = parser.parse_args()

    app = create_app()

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║         Sports Betting Dashboard                         ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Server running at: http://{args.host}:{args.port:<5}                  ║
    ║  Press Ctrl+C to stop                                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Open browser automatically unless disabled
    if not args.no_browser and not args.debug:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    # Run the Flask app
    # In production, use gunicorn instead: gunicorn --config gunicorn_config.py "dashboard.app:create_app()"
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=args.debug
    )


if __name__ == '__main__':
    main()
