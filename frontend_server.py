#!/usr/bin/env python3
"""
Simple HTTP server for serving the enhanced candlestick interface
"""

import http.server
import socketserver
import os
import sys

# Change to the webapp directory
os.chdir('/home/user/webapp')

# Set up the server
PORT = 3000
Handler = http.server.SimpleHTTPRequestHandler

class CustomHandler(Handler):
    def log_message(self, format, *args):
        """Custom log message to include timestamp"""
        sys.stdout.write(f"{self.log_date_time_string()} - {format % args}\n")
        sys.stdout.flush()

if __name__ == "__main__":
    try:
        with socketserver.TCPServer(("0.0.0.0", PORT), CustomHandler) as httpd:
            print(f"Frontend server running on port {PORT}")
            print(f"Serving directory: {os.getcwd()}")
            print(f"Enhanced Candlestick Interface: http://localhost:{PORT}/enhanced_candlestick_interface.html")
            sys.stdout.flush()
            httpd.serve_forever()
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)