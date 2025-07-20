#!/bin/bash

SERIAL="/dev/ttyUSB2"
DEBUG=true   # Set to false to disable debug output

while true; do
  RAW=$(echo -ne 'AT+QENG="servingcell"\r' | busybox microcom -t 5 "$SERIAL")

  # Clean output (remove command echo + OK)
  PARSED=$(echo "$RAW" | sed '1d;$d')

  # Extract lines for mode detection
  SA_MODE=$(echo "$PARSED" | grep '+QENG: "servingcell"' | grep ',"NR5G-SA",')
  
  NSA_MODE=$(echo "$PARSED" | grep '+QENG: "NR5G-NSA"')
  NSA_MODE_LTE_LINE=$(echo "$PARSED" | grep '+QENG: "LTE"')
  
  LTE_MODE=$(echo "$PARSED" | grep '+QENG: "servingcell"' | grep ',"LTE",')



  # Debug: Print raw modem response
  if [ "$DEBUG" = true ]; then
    echo "---- RAW MODEM RESPONSE @ $(date +%T) ----"
    echo "$RAW"

    echo "---- PARSED ----"
    echo "$PARSED"
    echo "SA_MODE: $SA_MODE"
    echo "NSA_MODE: $NSA_MODE"
    echo "NSA_MODE_LTE_LINE: $NSA_MODE_LTE_LINE"
    echo "LTE_MODE: $LTE_MODE"
    echo "-----------------------------------------"

  fi



  MODE="UNKNOWN"
  RSRP="NA"
  RSRQ="NA"
  SINR="NA"
  TIME=$(date +%T)

  if [[ -n "$SA_MODE" ]]; then
    MODE="NR5G-SA"
    RSRP=$(echo "$SA_MODE" | cut -d',' -f13)
    RSRQ=$(echo "$SA_MODE" | cut -d',' -f14)
    SINR=$(echo "$SA_MODE" | cut -d',' -f15)

  elif [[ -n "$NSA_MODE" ]]; then
    MODE="NR5G-NSA"
    # Use LTE line for signal values
    RSRP=$(echo "$NSA_MODE_LTE_LINE" | cut -d',' -f12)
    RSRQ=$(echo "$NSA_MODE_LTE_LINE" | cut -d',' -f13)
    SINR=$(echo "$NSA_MODE_LTE_LINE" | cut -d',' -f15)

  elif [[ -n "$LTE_MODE" ]]; then
    MODE="LTE"
    RSRP=$(echo "$LTE_MODE" | cut -d',' -f14)
    RSRQ=$(echo "$LTE_MODE" | cut -d',' -f15)
    SINR=$(echo "$LTE_MODE" | cut -d',' -f17)
  fi

  echo "$TIME, $MODE, RSRP=$RSRP, RSRQ=$RSRQ, SINR=$SINR"
  sleep 5
done
