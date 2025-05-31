import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
app.title = "Live UE Dashboard"

def load_data():
    df = pd.read_csv('./CSV/ml-FRER-vis.csv')
    return df
    
app.layout = html.Div([
    html.H1("Live UE Data Monitoring", style={'textAlign': 'center'}),
    dcc.Interval(id='interval', interval=1000, n_intervals=0),  # every 1 second
    dcc.Graph(id='plot1'),
    dcc.Graph(id='plot2'),
    dcc.Graph(id='plot3'),
    dcc.Graph(id='plot4'),
    dcc.Graph(id='plot5'),
    dcc.Graph(id='plot6'),
])

@app.callback(
    [Output('plot1', 'figure'),
     Output('plot2', 'figure'),
     Output('plot3', 'figure'),
     Output('plot4', 'figure'),
     Output('plot5', 'figure'),
     Output('plot6', 'figure')],
    Input('interval', 'n_intervals')
)
def update_graph(n):
    df = load_data()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=df['UE1<RSRP>'], name='UE1 RSRP'))
    fig1.add_trace(go.Scatter(y=df['UE1<SINR>'], name='UE1 SINR'))
    fig1.add_trace(go.Scatter(y=df['UE1<RSRQ>'], name='UE1 RSRQ', yaxis='y2'))
    fig1.update_layout(title="UE1 Radio Conditions", yaxis2=dict(overlaying='y', side='right'))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=df['UE1-Cluster'], mode='markers', name='Cluster'))
    fig2.add_trace(go.Scatter(y=df['UE1-Cluster-Hist'], mode='markers', name='State'))
    fig2.update_layout(title="UE1 Clusters and States")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=df['UE0<RSRP>'], name='UE0 RSRP'))
    fig3.add_trace(go.Scatter(y=df['UE0<SINR>'], name='UE0 SINR'))
    fig3.add_trace(go.Scatter(y=df['UE0<RSRQ>'], name='UE0 RSRQ', yaxis='y2'))
    fig3.update_layout(title="UE0 Radio Conditions", yaxis2=dict(overlaying='y', side='right'))

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=df['UE0-Cluster'], mode='markers', name='Cluster'))
    fig4.add_trace(go.Scatter(y=df['UE0-Cluster-Hist'], mode='markers', name='State'))
    fig4.update_layout(title="UE0 Clusters and States")

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(y=df['Replication ON/OFF'], mode='markers+lines', name='Replication'))
    fig5.update_layout(title="FRER Replication Status")

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(y=df['Transmitted Datagrams'], name='Tx'))
    fig6.add_trace(go.Scatter(y=df['Datagrams Loss Ratio %'], name='Loss Ratio', yaxis='y2'))
    fig6.update_layout(title="Datagrams and Loss Ratio", yaxis2=dict(overlaying='y', side='right'))

    return fig1, fig2, fig3, fig4, fig5, fig6

if __name__ == '__main__':
    app.run(debug=True)
