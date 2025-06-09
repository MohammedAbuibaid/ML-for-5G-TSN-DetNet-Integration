import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import threading
import pandas as pd
import numpy as np
import joblib, json
from sklearn.decomposition import PCA
from datetime import datetime
import serial
import time
import pyudev
import subprocess

import os
os.system("fuser -k 8050/tcp > /dev/null 2>&1")




# Load saved artifacts
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("cluster_mapping.json", "r") as f:
    cluster_map = json.load(f)

# Setup PCA using fitted training data
pca = PCA(n_components=2)
pca.fit(scaler.transform(kmeans.cluster_centers_))  # Use centroids for fit

# Buffers for live data
ue0_buffer = []
ue1_buffer = []
lock = threading.Lock()



tx_data = []  # Each entry will be: {'Time': timestamp, 'wwan0': value, 'wwan1': value}
last_tx_counts = {'wwan0': None, 'wwan1': None}
last_timestamp = None

pps_time_series = []

MAX_PPS_THRESHOLD = 200000

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

def get_tx_packets(interface):
    result = subprocess.run(["ip", "-s", "link", "show", interface], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = result.stdout.splitlines()
    
    # TX packets are typically on the 6th line (index 5), but you may need to adjust depending on system output
    for i, line in enumerate(lines):
        if "TX:" in line:
            tx_line = lines[i + 1]  # next line contains the packet count
            tx_packets = int(tx_line.strip().split()[0])
            return tx_packets
    return 0

initial_tx_counts = {
    'wwan0': get_tx_packets('wwan0'),
    'wwan1': get_tx_packets('wwan1')
}

# Serial reader
def monitor_ue(name, port, buffer):
    ser = None
    while True:
        try:
            if ser is None or not ser.is_open:
                ser = serial.Serial(port, 115200, timeout=3)
                print(f"[{name}] Connected to {port}")

            ser.write(b'AT+QENG="servingcell"\r')
            time.sleep(0.25)
            response = ser.read_all().decode(errors='ignore')
            lines = [line for line in response.splitlines() if '+QENG:' in line]

            # Basic mode matching
            if any('NR5G-SA' in l for l in lines):
                target = next(l for l in lines if 'NR5G-SA' in l)
                fields = target.split(',')
                rsrp, rsrq, sinr = float(fields[13]), float(fields[14]), float(fields[15])
            elif any('NR5G-NSA' in l for l in lines):
                lte = next(l for l in lines if '+QENG: "LTE"' in l)
                fields = lte.split(',')
                rsrp, rsrq, sinr = float(fields[11]), float(fields[12]), float(fields[14])
            elif any(',"LTE",' in l for l in lines):
                target = next(l for l in lines if ',"LTE",' in l)
                fields = target.split(',')
                rsrp, rsrq, sinr = float(fields[13]), float(fields[14]), float(fields[16])
            else:
                continue

            features = np.array([[rsrp, rsrq, sinr]])
            scaled = scaler.transform(features)
            cluster = kmeans.predict(scaled)[0]
            mapped = cluster_map[str(cluster)]
            pc = pca.transform(scaled)[0]

            with lock:
                buffer.append({
                    "Time": datetime.now(),
                    "RSRP": rsrp,
                    "RSRQ": rsrq,
                    "SINR": sinr,
                    "Cluster": mapped,
                    "PC1": -1 * pc[0],
                    "PC2": -1 * pc[1]
                })

            time.sleep(0.25)

        except Exception as e:
            print(f"[{name}] Error: {e}")
            if ser:
                try: ser.close()
                except: pass
            ser = None
            time.sleep(3)

# Start threads
ue_ports = find_ue_ports()
if len(ue_ports) >= 2:
    threading.Thread(target=monitor_ue, args=("ue0", ue_ports[0], ue0_buffer), daemon=True).start()
    threading.Thread(target=monitor_ue, args=("ue1", ue_ports[1], ue1_buffer), daemon=True).start()
else:
    print("Not enough Quectel UE ports found. Found:", ue_ports)


# Dash app
app = dash.Dash(__name__)
app.title = "Live UE Test Dashboard"




app.layout = html.Div([
    html.H2("Real-Time UE Testing Dashboard", style={'textAlign': 'center'}),
    dcc.Interval(id="interval", interval=2000, n_intervals=0),

    html.Div([
        html.Div([
            html.H4("UE0 Radio Measurements", style={'textAlign': 'center'}),
            dcc.Graph(id='ue0_signal', style={'height': '300px'}),
            # html.H4("UE0 Cluster Timeline", style={'textAlign': 'center'}),
            dcc.Graph(id='ue0_cluster', style={'height': '150px'}),
            html.H4("UE0 State Timeline", style={'textAlign': 'center'}),  # 👈 Added
            dcc.Graph(id='ue0_state', style={'height': '150px'})            # 👈 Added
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H4("UE1 Radio Measurements", style={'textAlign': 'center'}),
            dcc.Graph(id='ue1_signal', style={'height': '300px'}),
            # html.H4("UE1 Cluster Timeline", style={'textAlign': 'center'}),
            dcc.Graph(id='ue1_cluster', style={'height': '150px'}),
            html.H4("UE1 State Timeline", style={'textAlign': 'center'}),  # 👈 Added
            dcc.Graph(id='ue1_state', style={'height': '150px'})            # 👈 Added
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
 
        # html.Div([
        #     html.H4("TSN/DetNet Replication Function", style={'textAlign': 'center'}),
        #     dcc.Graph(id='tsn-replication', style={'height': '150px'})
        # ], style={'width': '48%', 'display': 'inline-block'}),


        html.Div([
            html.H4("TSN/DetNet Replication Function", style={'textAlign': 'center'}),
            dcc.Graph(id='tsn-replication', style={'height': '150px'})
        ], style={
            'width': '80%',  # Adjust as needed
            'margin': '0 auto',  # Center the block
            'display': 'block',  # Ensure it's a block-level element
            'textAlign': 'center'  # Optional, for contents inside
        }),


        html.Div([
            # html.H4("TX Packets Per Second", style={'textAlign': 'center'}),
            dcc.Graph(id='pps_plot', style={'height': '250px'})
        ], style={'width': '80%', 'margin': '0 auto'})


    ])
])





def generate_figs(buffer, label):
    if not buffer:
        empty_df = pd.DataFrame()  # Return an actual empty DataFrame
        return go.Figure(), go.Figure(), go.Figure(), empty_df

    df = pd.DataFrame(buffer)
    df = df.sort_values(by="Time")
    df["Time"] = pd.to_datetime(df["Time"])

    # Signal figure
    signal_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    signal_fig.add_trace(go.Scatter(x=df["Time"], y=df["RSRP"], mode='lines+markers', name='RSRP'), row=1, col=1)
    signal_fig.add_trace(go.Scatter(x=df["Time"], y=df["RSRQ"], mode='lines+markers', name='RSRQ'), row=2, col=1)
    signal_fig.add_trace(go.Scatter(x=df["Time"], y=df["SINR"], mode='lines+markers', name='SINR'), row=3, col=1)



    # Cluster timeline with dynamic marker colors
    cluster_colors = {0: 'green', 1: 'orange', 2: 'red'}
    marker_colors = [cluster_colors.get(c, 'gray') for c in df['Cluster']]

    cluster_fig = go.Figure()

    cluster_fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["Cluster"],
        mode='markers+lines',
        marker=dict(color=marker_colors, size=8, symbol='circle'),
        line=dict(color='black', width=1),  # Optional: keep or remove this line
        name='Cluster'
    ))

    cluster_fig.update_yaxes(
        tickvals=[0, 1, 2],
        ticktext=['Green', 'Orange', 'Red'],
        title="Cluster",
        range=[-0.5, 2.5]
    )

    cluster_fig.update_layout(
        title=f"{label} Clustering Timeline",
        height=250,
        showlegend=False
    )

    # ---------- Hysteresis-based State Tracking ----------

    # Mapping logic
    current_state = 'Red'
    state_counters = {'Green': 0, 'Orange': 0, 'Red': 0}
    transition_thresholds = {'Green': 21, 'Orange': 11, 'Red': 0}
    actual_states = []

    cluster_to_state = {0: 'Green', 1: 'Orange', 2: 'Red'}
    state_to_number = {'Green': 0, 'Orange': 1, 'Red': 2}

    def reset_counters():
        for k in state_counters:
            state_counters[k] = 0

    for _, row in df.iterrows():
        new_cluster_state = cluster_to_state[row['Cluster']]
        if (new_cluster_state == 'Orange' and current_state == 'Green') or \
           (new_cluster_state == 'Red' and current_state != 'Red'):
            current_state = new_cluster_state
            reset_counters()
        else:
            state_counters[new_cluster_state] += 1
            if current_state == 'Red' and new_cluster_state == 'Orange' and state_counters['Orange'] >= transition_thresholds['Orange']:
                current_state = 'Orange'
                reset_counters()
            elif current_state == 'Orange' and new_cluster_state == 'Green' and state_counters['Green'] >= transition_thresholds['Green']:
                current_state = 'Green'
                reset_counters()
            elif current_state == 'Red' and new_cluster_state == 'Green' and state_counters['Green'] >= transition_thresholds['Green']:
                current_state = 'Green'
                reset_counters()
        actual_states.append(state_to_number[current_state])

    df['State'] = actual_states

    # State Plot
    actual_colors = {0: 'green', 1: 'orange', 2: 'red'}
    actual_marker_colors = [actual_colors.get(s, 'gray') for s in df['State']]
    
    
    actual_fig = go.Figure()
    actual_fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["State"],
        mode='markers+lines',
        marker=dict(color=actual_marker_colors, size=8, symbol='circle'),
        line=dict(color='black', width=1),
        name='State'
    ))
    
    
    
    actual_fig.update_yaxes(
        tickvals=[0, 1, 2],
        ticktext=['Green', 'Orange', 'Red'],
        title="State",
        range=[-0.5, 2.5]
    )

    actual_fig.update_layout(title=f"{label} State Timeline", height=250, showlegend=False)


    return signal_fig, cluster_fig, actual_fig, df


@app.callback(
    Output('ue0_signal', 'figure'),
    Output('ue0_cluster', 'figure'),
    Output('ue0_state', 'figure'),
    Output('ue1_signal', 'figure'),
    Output('ue1_cluster', 'figure'),
    Output('ue1_state', 'figure'),
    Output('tsn-replication', 'figure'),
    Output("pps_plot", "figure"),
    Input('interval', 'n_intervals')
)

def update_graphs(n):
    fig0_sig, fig0_clu, fig0_state, df0 = generate_figs(ue0_buffer, "UE0")
    fig1_sig, fig1_clu, fig1_state, df1 = generate_figs(ue1_buffer, "UE1")

    # TSN logic based on both states
    if df0.empty or df1.empty or 'State' not in df0 or 'State' not in df1:
        # If any state missing → ON
        time_series = df0["Time"] if not df0.empty else df1["Time"]
        tsn_state = [1] * len(time_series)
    else:
        # Use aligned minimum length
        min_len = min(len(df0), len(df1))
        tsn_state = []
        for s0, s1 in zip(df0['State'][:min_len], df1['State'][:min_len]):
            if (s0 == 0 and s1 in [0, 1]) or (s1 == 0 and s0 in [0, 1]):
                tsn_state.append(0)  # OFF
            else:
                tsn_state.append(1)  # ON
        time_series = df0["Time"][:min_len]

    # Always build tsn_df outside the conditions
    tsn_df = pd.DataFrame({
        "Time": time_series,
        "TSN State": tsn_state
    })

    # 📊 Create TSN figure
    fig_tsn = go.Figure()
    fig_tsn.add_trace(go.Scatter(
        x=tsn_df["Time"],
        y=tsn_df["TSN State"],
        mode="lines+markers",
        marker=dict(color=tsn_df["TSN State"].map({0: "green", 1: "red"})),
        line=dict(shape="hv"),
        name="TSN State"
    ))
    fig_tsn.update_layout(
        title="TSN/DetNet Replication Function",
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=["OFF", "ON"],
            range=[-0.5, 1.5]
        ),
        height=120,
        margin=dict(t=40, b=40)
    )


############################3


    # Measure TX packets
    current_time = datetime.now()
    tx_pps = {}


    global last_tx_counts, last_timestamp, pps_time_series

    interfaces = ['wwan0', 'wwan1']
    current_time = datetime.now()
    total_tx_now = 0
    total_tx_prev = 0

    for iface in interfaces:
        tx_now = get_tx_packets(iface) - initial_tx_counts[iface]

        if last_tx_counts[iface] is not None:
            total_tx_now += tx_now
            total_tx_prev += last_tx_counts[iface]
        last_tx_counts[iface] = tx_now

    # Only compute if we have a previous timestamp
    if last_timestamp is not None:
        time_diff = (current_time - last_timestamp).total_seconds()
        total_pps = (total_tx_now - total_tx_prev) / time_diff if time_diff > 0 else 0

        if total_pps <= MAX_PPS_THRESHOLD:
            pps_time_series.append({'Time': current_time, 'PPS': total_pps})

    last_timestamp = current_time


    # Convert to DataFrame and create figure
    pps_df = pd.DataFrame(pps_time_series)
    pps_fig = go.Figure()
    if not pps_df.empty:
        pps_fig.add_trace(go.Scatter(
            x=pps_df["Time"],
            y=pps_df["PPS"],
            mode="lines+markers",
            name="Total PPS",
            line=dict(color="blue")
        ))
    pps_fig.update_layout(
        title="Total TX Packets Per Second (PPS)",
        xaxis_title="Time",
        yaxis_title="PPS",
        height=200,
        margin=dict(t=40, b=40)
    )





    return fig0_sig, fig0_clu, fig0_state, fig1_sig, fig1_clu, fig1_state, fig_tsn, pps_fig






if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8050)
