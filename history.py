import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Output, State, Input

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
import alpaca_trade_api as alpaca

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Checklist(
        id='toggle-rangeslider',
        options=[{'label': 'Include Rangeslider', 
                  'value': 'slider'}],
        value=['slider']
    ),
    dcc.Graph(id="graph"),
])

@app.callback(
    Output("graph", "figure"), 
    [Input("toggle-rangeslider", "value")])

def display_candlestick(value):
    api = alpaca.REST('PK7Z5SUF67ICPDK04R2M', 'ITAqIWxumbD67keejeh7yXTnrgSfnlZZZiXb759t', 'https://paper-api.alpaca.markets')
    #value1 = str(' " ' + value1 + ' " ')
    df = api.get_barset("AAPL", "day", limit=400).df["AAPL"]
    s = df.index.to_pydatetime()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # include candlestick with rangeselector
    fig.add_trace(go.Candlestick(x=[s[l].date() for l in range(len(s))],
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close']),
            secondary_y=False)

    # include a go.Bar trace for volumes
    fig.add_trace(go.Bar(x=[s[l].date() for l in range(len(s))], y=df['volume'], marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, opacity=0.4),
            secondary_y=True)
    fig.layout.xaxis1.type = "category"

    fig.layout.yaxis2.showgrid=False
    fig.update_yaxes(title_text="<b>VOLUME</b>", secondary_y=True)
    fig.update_yaxes(title_text="<b>STOCK PRICE</b>", secondary_y=False)
    fig.update_layout(xaxis_rangeslider_visible='slider' in value, showlegend=False,
            height=600, width=1300)

    return fig

app.run_server(debug=True)
