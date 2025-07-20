#!/bin/bash

# Set paths and config
IFACE="wwan1"

WATCHDOG_SCRIPT="/usr/local/bin/quectel_watchdog_${IFACE}.sh"
SYSTEMD_UNIT="/etc/systemd/system/quectel-watchdog-${IFACE}.service"

#QCM_PATH="/home/abuibaid/quectel/quectel-CM/quectel-CM"
QCM_PATH="/home/ws/Downloads/quectel-CM/quectel-CM"

APN="default"
LOGFILE="/var/log/quectel_watchdog_${IFACE}.log"

# Create watchdog script
cat <<EOF > "$WATCHDOG_SCRIPT"
#!/bin/bash

APN="$APN"
IFACE="$IFACE"
QCM_PATH="$QCM_PATH"
LOGFILE="$LOGFILE"

log() {
    echo "[\$(date '+%F %T')] \$1" | tee -a "\$LOGFILE"
}

start_qcm() {
    log "Starting Quectel-CM..."
    "\$QCM_PATH" -s "\$APN" -i "\$IFACE" &
    echo \$! > /var/run/quectel_cm_\$IFACE.pid
}

stop_qcm() {
    if [ -f /var/run/quectel_cm_\$IFACE.pid ]; then
        kill -9 \$(cat /var/run/quectel_cm_\$IFACE.pid) 2>/dev/null
        rm /var/run/quectel_cm_\$IFACE.pid
    fi
}

check_connection() {
    ip link show "\$IFACE" > /dev/null 2>&1
    if [ \$? -ne 0 ]; then
        log "Interface \$IFACE not found."
        return 1
    fi

    ip route | grep -q "\$IFACE"
    if [ \$? -ne 0 ]; then
        log "No default route via \$IFACE."
        return 1
    fi

    return 0
}

while true; do
    check_connection
    if [ \$? -ne 0 ]; then
        log "Connection lost. Restarting Quectel-CM."
        stop_qcm
        sleep 1
        start_qcm
        sleep 7
    fi
    sleep 3
done
EOF

# Make script executable
chmod +x "$WATCHDOG_SCRIPT"

# Create systemd service
cat <<EOF > "$SYSTEMD_UNIT"
[Unit]
Description=Quectel 5G Connection Watchdog for $IFACE
After=network.target

[Service]
ExecStart=$WATCHDOG_SCRIPT
Restart=always
RestartSec=1
User=root

[Install]
WantedBy=multi-user.target
EOF

# Apply and start service
systemctl daemon-reload
systemctl enable --now quectel-watchdog-${IFACE}.service

echo "✅ Quectel watchdog for $IFACE is set up and running."

# ✅ To run:
# chmod +x setup_quectel_watchdog.sh
# sudo ./setup_quectel_watchdog.sh
