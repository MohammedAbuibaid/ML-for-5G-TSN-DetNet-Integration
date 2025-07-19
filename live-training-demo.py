import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output
import threading
import time
import serial
# from serial import SerialException
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pyudev
from plotly.subplots import make_subplots

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import joblib
import json

import os
os.system("fuser -k 8050/tcp > /dev/null 2>&1")

training_complete = False

app = dash.Dash(__name__)

app.title = "Live Training Dashboard"

# Shared data buffer for live updates
data_buffer = []
data_lock = threading.Lock()

# Clustering status shared
clustering_status = "Waiting for data..."


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
    ser = None
    while True:
        try:
            if ser is None or not ser.is_open:
                ser = serial.Serial(port_path, baudrate=115200, timeout=5)
                print(f"[{name}] Connected to {port_path}")

            ser.write(b'AT+QENG="servingcell"\r')
            time.sleep(0.25)
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
            except ValueError:
                pass

            time.sleep(0.25)

        except (serial.SerialException, OSError) as e:
            print(f"[{name}] Connection error: {e}. Retrying in 3 seconds...")
            if ser:
                try:
                    ser.close()
                except:
                    pass
                ser = None
            time.sleep(3)


def cluster_loop():
    global clustering_status
    max_data_points = 300
    target_score = 0.4
    training_stopped = False

    while not training_stopped:
        time.sleep(5)
        with data_lock:
            if len(data_buffer) < 5:
                continue
            buffer_snapshot = data_buffer.copy()
        df = pd.DataFrame(buffer_snapshot)

        df.drop('Time', axis=1, inplace=True)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[['RSRP', 'RSRQ', 'SINR']])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(scaled)

        labels = kmeans.labels_
        score = silhouette_score(scaled, labels)

        clustering_status = f"Samples: {len(df)}, Silhouette Score: {score:.2f}"

        with data_lock:
            for i in range(len(labels)):
                data_buffer[i]['Cluster'] = labels[i]

        if score >= target_score and len(df) >= max_data_points:
            print(f"\n✅ Training complete. Score: {score:.2f}, Samples: {len(df)}")
            print("🔽 Saving training data and model...")

            df['Time'] = [d['Time'] for d in buffer_snapshot]

            centroids = kmeans.cluster_centers_
            inverse_centroids = scaler.inverse_transform(centroids)

            # Sort clusters by SINR descending
            sorted_clusters = sorted(
                [(i, c) for i, c in enumerate(inverse_centroids)],
                key=lambda x: -x[1][2]
            )

            # Create mapping: old_cluster → new_sorted_cluster
            new_cluster_map = {old: new for new, (old, _) in enumerate(sorted_clusters)}

            # Overwrite cluster IDs
            df['Cluster'] = [new_cluster_map[label] for label in kmeans.labels_]
            with data_lock:
                for i in range(len(buffer_snapshot)):
                    buffer_snapshot[i]['Cluster'] = new_cluster_map[labels[i]]

            # Save model components
            joblib.dump(kmeans, "kmeans_model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            with open("cluster_mapping.json", "w") as f:
                json.dump(new_cluster_map, f)

            # Save labeled dataset
            df.to_csv("training_dataset.csv", index=False)

            training_stopped = True
            global training_complete
            training_complete = True

            # PCA and Plotting
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(scaled)

            inverse_centroids = np.array([inverse_centroids[old] for old, _ in sorted_clusters])
            centroids_reduced = pca.transform(np.array([centroids[old] for old, _ in sorted_clusters]))

            colors = ['green', 'orange', 'red']
            labels_text = ['Green Cluster Centroid', 'Orange Cluster Centroid', 'Red Cluster Centroid']

            plt.figure(figsize=(7, 5), dpi=150)

            for cluster_id in range(3):
                mask = df['Cluster'] == cluster_id
                plt.scatter(
                    -1 * reduced_features[mask, 0],
                    -1 * reduced_features[mask, 1],
                    c=colors[cluster_id],
                    alpha=0.7,
                    s=50,
                    edgecolor='k',
                )

            legend_handles = []
            for cluster_id in range(3):
                reduced_centroid = centroids_reduced[cluster_id]
                true_centroid = inverse_centroids[cluster_id]
                label = f"{labels_text[cluster_id]}: RSRP={true_centroid[0]:.1f}, RSRQ={true_centroid[1]:.1f}, SINR={true_centroid[2]:.1f}"
                handle = plt.scatter(
                    -1 * reduced_centroid[0],
                    -1 * reduced_centroid[1],
                    color=colors[cluster_id],
                    marker='X',
                    s=200,
                    label=label
                )
                legend_handles.append(handle)

            plt.xlabel('Principal Component 1', fontsize=14)
            plt.ylabel('Principal Component 2', fontsize=14)
            plt.legend(
                handles=legend_handles,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.35),
                ncol=1,
                fontsize=10,
                frameon=False
            )
            plt.tight_layout()
            plt.savefig("cluster_vis_pca.pdf", bbox_inches='tight')
            plt.show()






ue_ports = find_ue_ports()
if len(ue_ports) >= 2:
    threading.Thread(target=monitor_ue, args=("ue0", ue_ports[0]), daemon=True).start()
    threading.Thread(target=monitor_ue, args=("ue1", ue_ports[1]), daemon=True).start()
else:
    print("Not enough Quectel UE ports found. Found:", ue_ports)

threading.Thread(target=cluster_loop, daemon=True).start()

app.layout = html.Div([
    html.H1("Live UE Signal & Clustering Dashboard", style={'textAlign': 'center'}),
    # dcc.Interval(id='interval', interval=250, n_intervals=0),
    dcc.Interval(id='interval', interval=250, n_intervals=0, disabled=False),

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
    ]),

    html.Div([
        html.Div(id='clustering-status', style={
            'textAlign': 'center',
            'color': 'blue',
            'fontWeight': 'bold',
            'fontSize': '18px',
            'marginTop': '10px'
        })
    ])
])



@app.callback(
    [Output('ue0_radio', 'figure'),
    Output('ue1_radio', 'figure'),
    Output('ue0_cluster', 'figure'),
    Output('ue1_cluster', 'figure'),
    Output('clustering-status', 'children'),
    Output('interval', 'disabled')],
    Input('interval', 'n_intervals')
)


def update_graph(n):
    global clustering_status
    with data_lock:
        df = pd.DataFrame(data_buffer)      
        
        if df.empty or 'UE' not in df.columns:
            return go.Figure(), go.Figure(), go.Figure(), go.Figure(), "No data"

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

        if 'Cluster' in sub:
            fig_c.add_trace(go.Scatter(x=sub['Time'], y=sub['Cluster'], mode='markers', name='Cluster'))
            fig_c.update_layout(
                yaxis=dict(tickmode='array', tickvals=[0, 1, 2], range=[-0.5, 2.5])
            )

    # return fig_ue0, fig_ue1, fig_c0, fig_c1, clustering_status
    
    return fig_ue0, fig_ue1, fig_c0, fig_c1, clustering_status, training_complete


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8050, host='0.0.0.0')
