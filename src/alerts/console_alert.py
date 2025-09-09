import time


def alert_console(message: str):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[ALERT] {timestamp} - {message}")