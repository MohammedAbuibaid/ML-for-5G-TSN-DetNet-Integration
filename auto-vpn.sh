#!/bin/bash

################################################################################
#                CONFIGURATION PARAMETERS — EDIT AS NEEDED                     #
################################################################################

# Replace this with the actual MEC IP
MEC_IP=143.110.221.29

# WiFi Interface and default gateway (used to remove system default routes)
WIFI_IF="wlp0s20f3"

# VPN configuration files
UE0_VPN_CONFIG="ue1vpn.ovpn"
UE1_VPN_CONFIG="ue2vpn.ovpn"

# Quectel interface and APN config
UE_APN="ltemobile.apn"
UE0_IF="wwan0"

# Policy routing tables
UE0_ROUTING_TABLE="rttun0"
UE1_ROUTING_TABLE="rttun1"
UE0_PHY_TABLE="rtwwan0"
UE1_PHY_TABLE="rtwwan1"

# VPN tunnel interface
UE0_TUN_IF="tun0"



################################################################################
#                          SCRIPT STARTS HERE                                  #
################################################################################

echo "## Resetting all configurations ##"

#########################################
# Step 0: Disconnect any existing VPN sessions
#########################################

openvpn3 session-manage --config "$UE0_VPN_CONFIG" --disconnect >/dev/null 2>&1
openvpn3 session-manage --config "$UE1_VPN_CONFIG" --disconnect >/dev/null 2>&1

#########################################
# Step 1: Clean old policy-based routing rules
#########################################

clean_ip_rule() {
    local table=$1
    local ip=$(ip rule list | grep "$table" | grep -Eo "([0-9]{1,3}\.){3}[0-9]{1,3}" | grep -v "127.0.0.1" | head -1)
    if [ -n "$ip" ]; then
        ip rule del from "$ip" table "$table" >/dev/null 2>&1
    fi
}

echo "Step 1/2: Delete old policy-based routing rules"

for table in "$UE0_ROUTING_TABLE" "$UE1_ROUTING_TABLE" "$UE0_PHY_TABLE" "$UE1_PHY_TABLE"; do
    clean_ip_rule "$table"
done

#########################################
# Step 2: Delete default gateway from main routing table
#########################################

echo "Step 2/2: Remove default gateway from main table"

WIFI_GW=$(ip route list | grep "dev $WIFI_IF proto dhcp" | awk '{print $3}' | head -1)

if [ -n "$WIFI_GW" ]; then
    ip route del default via "$WIFI_GW" dev "$WIFI_IF" >/dev/null 2>&1
fi

#########################################
# Step 3: Setup VPN over UE0
#########################################

echo "## Establishing PDU session over UE0 ($UE0_IF) ##"
"$QUECTEL_CM_PATH" -s "$UE_APN" -i "$UE0_IF" &

# Optional: monitor serving cell info
# bash comServingCell-ue0.sh

echo "## Starting OpenVPN tunnel for UE0 ##"
openvpn3 session-start --config "$UE0_VPN_CONFIG"

#########################################
# Step 4: Setup policy-based routing for UE0
#########################################

echo "## Setup policy-based routing to split OpenVPN over UE0 ##"

# Cleanup routing rules again for safety
for i in {1..2}; do
    clean_ip_rule "$UE0_ROUTING_TABLE"
done

sleep 1

# Add new IP rules
echo "Step 2/5: Add new policy-based routing rules"
UE0_PHY_IP=$(ifconfig "$UE0_IF" | grep -Eo "inet ([0-9]{1,3}\.){3}[0-9]{1,3}" | awk '{print $2}' | head -1)
TUN0_IP=$(ifconfig "$UE0_TUN_IF" | grep -Eo "inet ([0-9]{1,3}\.){3}[0-9]{1,3}" | awk '{print $2}' | head -1)

if [ -n "$UE0_PHY_IP" ]; then
    ip rule add from "$UE0_PHY_IP" table "$UE0_ROUTING_TABLE" pref 32000
fi

if [ -n "$TUN0_IP" ]; then
    ip rule add from "$TUN0_IP" table "$UE0_ROUTING_TABLE" pref 32001
fi

sleep 1

# Add routes to routing table
echo "Step 3/5: Add IP routing rules to $UE0_ROUTING_TABLE"

UE0_LAN=$(ip route list | grep "dev $UE0_IF proto kernel scope link src" | awk '{print $1}' | head -1)
UE0_GW=$(ip route list | grep "default via .* dev $UE0_IF" | awk '{print $3}' | head -1)

ip route add "$UE0_LAN" dev "$UE0_IF" src "$UE0_PHY_IP" table "$UE0_ROUTING_TABLE"
ip route add "$MEC_IP" via "$UE0_GW" dev "$UE0_IF" table "$UE0_ROUTING_TABLE"
ip route add 10.8.0.0/24 dev "$UE0_TUN_IF" proto kernel scope link src 10.8.0.2 table "$UE0_ROUTING_TABLE"

sleep 1

# Clean up main table
echo "Step 4/5: Delete conflicting routes from main table"

ip route del 0.0.0.0/1 via 10.8.0.1 dev "$UE0_TUN_IF" >/dev/null 2>&1
ip route del 128.0.0.0/1 via 10.8.0.1 dev "$UE0_TUN_IF" >/dev/null 2>&1
ip route del default via "$UE0_GW" dev "$UE0_IF" >/dev/null 2>&1
ip route del 10.8.0.0/24 dev "$UE0_TUN_IF" >/dev/null 2>&1
ip route del "$UE0_LAN" dev "$UE0_IF" >/dev/null 2>&1
ip route del "$MEC_IP" via "$UE0_GW" dev "$UE0_IF" >/dev/null 2>&1

sleep 1

echo "## UE0 VPN and routing setup completed successfully ##"
