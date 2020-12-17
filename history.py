import dash                                                   # pip3 install dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from dash.dependencies import Output, State, Input

import plotly.graph_objects as go                                     
from plotly.subplots import make_subplots

import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
import alpaca_trade_api as alpaca                                     # pip3 install alpaca-trade-api

from openpyxl import Workbook                                         # pip3 install openpyxl

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df11 = pd.read_excel(r'X:\Upwork\projects\plotting_trade_data\data_ohlc.xlsx',engine='openpyxl',)  # pip3 install xlrd

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Center(html.Div([dcc.Input(id='input-on-submit', type='text', placeholder="Enter a ticker"), 
            dcc.Input(id='input-on-submit1', type='number', placeholder='Number of OHLCs (MAX1000)', max=1000, style={'width': '25%'}),
            html.Div(dcc.Dropdown(id='demo-dropdown', options=[{'label': '1 Minute', 'value': '1Min'},
            {'label': '5 Minutes', 'value': '5Min'},
            {'label': '15 Minutes', 'value': '15Min'},
            {'label': '1 Day', 'value': 'day'}], placeholder="Span of each OHLC", value = "day"), style={'width': '20%'}),
            html.Div(html.Button('Submit', id='submit-val', n_clicks=0))])),
    html.Div(dcc.Checklist(id='toggle-rangeslider', value=['slider'])),
    html.Div(dcc.Graph(id="graph")),
    html.Center(html.Div(children=[
        html.H4(children='Last 10 saved records'),
        generate_table(df11)]))   
        
])

@app.callback(
        Output("graph", "figure"),
        Input('submit-val', 'n_clicks'),
        Input('input-on-submit1', "value"),
        Input("toggle-rangeslider", "value"),
        Input('demo-dropdown', 'value'), 
        State('input-on-submit', "value")
        )
def display_candlestick(n_clicks, tspan, togg, itspan, ticker):

    if n_clicks>0 and ticker!= None :
        api = alpaca.REST('PK7Z5SUF67ICPDK04R2M', 'ITAqIWxumbD67keejeh7yXTnrgSfnlZZZiXb759t', 'https://paper-api.alpaca.markets')
        df = api.get_barset(ticker, itspan, limit=tspan).df[ticker]

        s = df.index.to_pydatetime()
    
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # include candlestick with rangeselector
        fig.add_trace(go.Candlestick(x=s,
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close']), secondary_y=False)

        # include a go.Bar trace for volumes
        fig.add_trace(go.Bar(x=s, y=df['volume'], marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, opacity=0.4),
            secondary_y=True)

        fig.layout.yaxis2.showgrid=False
        fig.update_yaxes(title_text="<b>VOLUME</b>", secondary_y=True)
        fig.update_yaxes(title_text="<b>STOCK PRICE</b>", secondary_y=False)
    
        fig.update_layout(xaxis_rangeslider_visible='slider' in togg, showlegend=False,
                height=750, width=1300, title=ticker)
        fig.update_layout(
            xaxis=dict(
              rangeselector=dict(
              buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
           )
        )
        return fig   
    else:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.layout.yaxis2.showgrid=False
        fig2.update_yaxes(title_text="<b>VOLUME</b>", secondary_y=True)
        fig2.update_yaxes(title_text="<b>STOCK PRICE</b>", secondary_y=False)
        fig2.update_layout(xaxis_rangeslider_visible='slider' in togg, showlegend=False,
                height=750, width=1300, title="Ticker")            
        return fig2


if __name__ == '__main__':
    app.run_server(debug=True)
