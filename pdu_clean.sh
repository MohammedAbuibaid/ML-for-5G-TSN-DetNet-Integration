#!/bin/bash

# Set variables (same as original script)
IFACE="wwan1"
WATCHDOG_SCRIPT="/usr/local/bin/quectel_watchdog_${IFACE}.sh"
SYSTEMD_UNIT="/etc/systemd/system/quectel-watchdog-${IFACE}.service"
PID_FILE="/var/run/quectel_cm_${IFACE}.pid"
LOGFILE="/var/log/quectel_watchdog_${IFACE}.log"

echo "Stopping and disabling systemd service..."
sudo systemctl stop quectel-watchdog-${IFACE}.service 2>/dev/null
sudo systemctl disable quectel-watchdog-${IFACE}.service 2>/dev/null

echo "Removing systemd unit and watchdog script..."
sudo rm -f "$SYSTEMD_UNIT"
sudo rm -f "$WATCHDOG_SCRIPT"

echo "Removing PID and log files..."
sudo rm -f "$PID_FILE"
# Uncomment the next line if you want to remove the log file too
# sudo rm -f "$LOGFILE"

echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "✅ Cleanup completed. You can now safely re-run your setup script."
