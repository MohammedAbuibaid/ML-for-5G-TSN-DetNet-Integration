import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output
import threading
import time
import serial
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pyudev
from plotly.subplots import make_subplots

import os
os.system("fuser -k 8050/tcp > /dev/null 2>&1")

app = dash.Dash(__name__)

app.title = "Live Dual UE Dashboard"

# Shared data buffer for live updates
data_buffer = []
data_lock = threading.Lock()

def parse_fields(line, mode):
    fields = line.strip().split(',')
    if mode == "NR5G-SA":
        return fields[13], fields[14], fields[15]
    elif mode == "NR5G-NSA":
        return fields[11], fields[12], fields[14]
    elif mode == "LTE":
        return fields[13], fields[14], fields[16]
    return "NA", "NA", "NA"

def find_ue_ports():
    context = pyudev.Context()
    ports_by_dev = {}
    for device in context.list_devices(subsystem='tty'):
        if 'ttyUSB' in device.sys_name:
            parent = device.find_parent('usb', 'usb_device')
            if parent is not None:
                vid = parent.properties.get('ID_VENDOR_ID')
                pid = parent.properties.get('ID_MODEL_ID')
                if vid == '2c7c':  # Quectel vendor ID
                    devname = parent.sys_name
                    if devname not in ports_by_dev:
                        ports_by_dev[devname] = []
                    ports_by_dev[devname].append(f"/dev/{device.sys_name}")
    selected_ports = []
    for dev_ports in ports_by_dev.values():
        sorted_ports = sorted(dev_ports)
        if len(sorted_ports) >= 3:
            selected_ports.append(sorted_ports[2])
    return sorted(selected_ports)

def monitor_ue(name, port_path):
    try:
        ser = serial.Serial(port_path, baudrate=115200, timeout=5)
    except serial.SerialException:
        print(f"[{name}] ERROR: Cannot open port {port_path}")
        return

    while True:
        ser.write(b'AT+QENG="servingcell"\r')
        time.sleep(1)
        response = ser.read_all().decode(errors='ignore').strip()

        lines = response.splitlines()
        lines = [line for line in lines if "+QENG:" in line]

        mode = "UNKNOWN"
        rsrp = rsrq = sinr = "NA"

        sa_line = next((l for l in lines if '"NR5G-SA"' in l), None)
        nsa_line = next((l for l in lines if '"NR5G-NSA"' in l), None)
        lte_line_nsa = next((l for l in lines if l.startswith('+QENG: "LTE"')), None)
        lte_line = next((l for l in lines if ',"LTE",' in l and '"servingcell"' in l), None)

        if sa_line:
            mode = "NR5G-SA"
            rsrp, rsrq, sinr = parse_fields(sa_line, mode)
        elif nsa_line and lte_line_nsa:
            mode = "NR5G-NSA"
            rsrp, rsrq, sinr = parse_fields(lte_line_nsa, mode)
        elif lte_line:
            mode = "LTE"
            rsrp, rsrq, sinr = parse_fields(lte_line, mode)

        try:
            rsrp_f = float(rsrp)
            rsrq_f = float(rsrq)
            sinr_f = float(sinr)
            with data_lock:
                data_buffer.append({
                    "UE": name,
                    "RSRP": rsrp_f,
                    "RSRQ": rsrq_f,
                    "SINR": sinr_f,
                    "Time": datetime.now(),
                })
                if len(data_buffer) > 1000:
                    data_buffer[:] = data_buffer[-1000:]
        except ValueError:
            pass
        time.sleep(0.5)

def cluster_loop():
    while True:
        time.sleep(10)
        with data_lock:
            if len(data_buffer) < 10:
                continue
            df = pd.DataFrame(data_buffer)
        df.drop('Time', axis=1, inplace=True)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[['RSRP', 'RSRQ', 'SINR']])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(scaled)
        labels = kmeans.labels_
        with data_lock:
            for i in range(len(labels)):
                data_buffer[i]['Cluster'] = labels[i]

ue_ports = find_ue_ports()
if len(ue_ports) >= 2:
    threading.Thread(target=monitor_ue, args=("ue0", ue_ports[0]), daemon=True).start()
    threading.Thread(target=monitor_ue, args=("ue1", ue_ports[1]), daemon=True).start()
else:
    print("Not enough Quectel UE ports found. Found:", ue_ports)

threading.Thread(target=cluster_loop, daemon=True).start()

app.layout = html.Div([
    html.H1("Live UE Signal & Clustering Dashboard", style={'textAlign': 'center'}),
    dcc.Interval(id='interval', interval=1000, n_intervals=0),

    html.Div([
        html.Div([
            html.H3("UE0 Radio Measurements", style={'textAlign': 'center'}),
            dcc.Graph(id='ue0_radio')
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.H3("UE1 Radio Measurements", style={'textAlign': 'center'}),
            dcc.Graph(id='ue1_radio')
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ]),

    html.Div([
        html.Div([
            html.H3("UE0 Clusters", style={'textAlign': 'center'}),
            dcc.Graph(id='ue0_cluster')
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.H3("UE1 Clusters", style={'textAlign': 'center'}),
            dcc.Graph(id='ue1_cluster')
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ])
])

@app.callback(
    [Output('ue0_radio', 'figure'),
     Output('ue1_radio', 'figure'),
     Output('ue0_cluster', 'figure'),
     Output('ue1_cluster', 'figure')],
    Input('interval', 'n_intervals')
)
def update_graph(n):
    with data_lock:
        df = pd.DataFrame(data_buffer)
        if df.empty or 'UE' not in df.columns:
            return go.Figure(), go.Figure(), go.Figure(), go.Figure()

    fig_ue0 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            subplot_titles=("RSRP", "RSRQ", "SINR"))
    fig_ue1 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            subplot_titles=("RSRP", "RSRQ", "SINR"))
    fig_c0 = go.Figure()
    fig_c1 = go.Figure()

    for ue, fig_r, fig_c in [("ue0", fig_ue0, fig_c0), ("ue1", fig_ue1, fig_c1)]:
        sub = df[df['UE'] == ue]
        if sub.empty:
            continue
        fig_r.add_trace(go.Scatter(x=sub['Time'], y=sub['RSRP'], mode='lines+markers', name='RSRP'), row=1, col=1)
        fig_r.add_trace(go.Scatter(x=sub['Time'], y=sub['RSRQ'], mode='lines+markers', name='RSRQ'), row=2, col=1)
        fig_r.add_trace(go.Scatter(x=sub['Time'], y=sub['SINR'], mode='lines+markers', name='SINR'), row=3, col=1)
        # fig_r.update_layout(title=f"{ue.upper()} Radio Conditions", height=600)

        if 'Cluster' in sub:
            fig_c.add_trace(go.Scatter(x=sub['Time'], y=sub['Cluster'], mode='markers', name='Cluster'))
            fig_c.update_layout(
                # title=f"{ue.upper()} Clusters",
                yaxis=dict(tickmode='array', tickvals=[0, 1, 2], range=[-0.5, 2.5])
            )

    return fig_ue0, fig_ue1, fig_c0, fig_c1

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8050)
