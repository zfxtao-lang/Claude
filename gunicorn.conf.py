"""Gunicorn configuration for the gateway."""
import multiprocessing

bind = "127.0.0.1:5000"
workers = min(multiprocessing.cpu_count() * 2 + 1, 4)  # cap at 4 for lightweight server
threads = 4
timeout = 180  # long timeout for AI API calls
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"
