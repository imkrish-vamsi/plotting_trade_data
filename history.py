import dash                                                   # pip3 install dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from dash.dependencies import Output, State, Input

import plotly.graph_objects as go                                     
from plotly.subplots import make_subplots
import plotly.express as px

import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
import alpaca_trade_api as alpaca                                     # pip3 install alpaca-trade-api

from ipywidgets import interactive, HBox, VBox
from openpyxl import Workbook, load_workbook                                         # pip3 install openpyxl
import yaml

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, 
                    suppress_callback_exceptions=True, prevent_initial_callbacks=True)

app.layout = html.Div(children = [dcc.ConfirmDialog(id='confirm', message='DATA SAVED!!'),
            html.Center(html.Div(id='output-confirm')),
            html.Center(html.Div(children = [dcc.Input(id='input-on-submit', type='text', placeholder="Enter a ticker"), 
            dcc.Input(id='input-on-submit1', type='number', placeholder='Number of OHLCs (MAX1000)', max=1000, style={'width': '25%'}),
            dcc.Dropdown(id='demo-dropdown', options=[{'label': '1 Minute', 'value': '1Min'},
            {'label': '5 Minutes', 'value': '5Min'},
            {'label': '15 Minutes', 'value': '15Min'},
            {'label': '1 Day', 'value': 'day'}], placeholder="Span of each OHLC", value = "day", style={'width': '40%'})])),
            html.Br(),
            html.Center(html.Div(html.Button('Submit', id='submit-val', n_clicks=0))),
            html.Div(dcc.Graph(id="graph")),
            html.Div(dcc.Graph(id='carGraph')), #Graph that displays all data
            html.Div(dcc.Graph(id='filterGraph')), #Graph that shows only filtered data
            html.Div(id='display'),  #To show format of selectData
            html.Center(html.Button("Grab Data", id='submit-val1', n_clicks=0)),
            html.Center(html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line'})),
            html.Div(id='nouse1', style={'display':'none'}),  # dummy boxes
            html.Div(id='nouse', style={'display':'none'})      # dummy boxes

  
]) 

@app.callback(
        Output("graph", "figure"),
        Input('submit-val', 'n_clicks'),
        Input('input-on-submit1', "value"),
        Input('demo-dropdown', 'value'), 
        State('input-on-submit', "value")
    )
def display_candlestick(n_clicks, tspan, itspan, ticker):

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

        fig.update_layout(showlegend=False, title=ticker, xaxis = {"showspikes": True}, yaxis = {"showspikes": True})
        fig.update_layout(xaxis=dict(rangeselector=dict(
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
        rangeslider=dict(visible=True), type="date"))
        return fig

    else:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.layout.yaxis2.showgrid=False
        fig2.update_yaxes(title_text="<b>VOLUME</b>", secondary_y=True)
        fig2.update_yaxes(title_text="<b>STOCK PRICE</b>", secondary_y=False)
        fig2.update_layout(#xaxis_rangeslider_visible='slider' in togg,
                 showlegend=False, title="Ticker")            
        return fig2

@app.callback(Output('confirm', 'displayed'),
              Input('submit-val1', 'n_clicks'))
def display_confirm(n_clicks1):
    if n_clicks1>0:
        return True
    return False

@app.callback(Output('output-confirm', 'children'),
              Input('submit-val1', 'n_clicks'))
def update_output(n_clicks):
    if n_clicks:
        return 'You Grabbed the Data {} times in this session!'.format(n_clicks)

@app.callback(Output('nouse1', 'children'),
        Input("submit-val1", 'n_clicks'),
        State("nouse", 'children'),
        State('input-on-submit1', "value"),
        State('demo-dropdown', 'value'), 
        State('input-on-submit', "value")
)
def making_dataset(n_clicks1, pts, tspan, itspan, ticker):
    if n_clicks1:    
        api = alpaca.REST('PK7Z5SUF67ICPDK04R2M', 'ITAqIWxumbD67keejeh7yXTnrgSfnlZZZiXb759t', 'https://paper-api.alpaca.markets')
        df = api.get_barset(ticker, itspan, limit=tspan).df[ticker]

        pts = yaml.load(pts)
        rang1 = pts[0]['text']
        rang2 = pts[len(pts)-1]['text']
        rang = [rang1, rang2]

        df.index = [x.strftime('%Y-%m-%d %H:%M:%S') for x in df.index]
        i = 0
        for str_data_time in rang:
            output = str_data_time.split(".")[0]
            rang[i] = output
            i += 1
        df = df.truncate(before = rang1, after = rang2) 
        
        df3 = pd.DataFrame({'SYMBOL': ticker,
                   'TIME': df.index, 'OPEN': df['open'],
                   'HIGH': df['high'], 'LOW': df['low'],
                   'CLOSE': df['close'], 'VOLUME': df['volume']})
        writer = pd.ExcelWriter(r"X:\Upwork\projects\plotting_trade_data\data_ohlc.xlsx", engine='openpyxl')
        # try to open an existing workbook
        writer.book = load_workbook(r'X:\Upwork\projects\plotting_trade_data\data_ohlc.xlsx')
        # copy existing sheets
        writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        # read existing file
        reader = pd.read_excel(r'X:\Upwork\projects\plotting_trade_data\data_ohlc.xlsx', engine='openpyxl')
        # write out the new sheet
        df3.to_excel(writer,index=False,header=False,startrow=len(reader)+1)
        writer.close()
        n_clicks1 = 0
        return html.Div(html.H4(children="Running!!"))
 

@app.callback(
    Output('textarea-example-output', 'children'),
    Input('submit-val1', 'n_clicks'),
    Input('submit-val', 'n_clicks')
    )
def update_tables(n_clicks1, n_clicks):
    if n_clicks>0 or n_clicks1>0:    
        df11 = pd.read_excel(r'X:\Upwork\projects\plotting_trade_data\data_ohlc.xlsx',engine='openpyxl')  # pip3 install xlrd
        return html.Div(children=[
            html.Br(),
            html.Br(),
            html.H4(children='Last 10 saved records (Press submit to refresh)'),
                html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in df11.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(df11.iloc[i][col]) for col in df11.columns
                    ]) for i in range(len(df11)-10, len(df11))
                ])
            ])
        ]) 


@app.callback(Output('carGraph','figure'), Input('submit-val','n_clicks'),
            State('input-on-submit1', "value"),
            State('demo-dropdown', 'value'), 
            State('input-on-submit', "value"))            
def testfunc(clicks, tspan, itspan, ticker):
    api = alpaca.REST('PK7Z5SUF67ICPDK04R2M', 'ITAqIWxumbD67keejeh7yXTnrgSfnlZZZiXb759t', 'https://paper-api.alpaca.markets')
    df = api.get_barset(ticker, itspan, limit=tspan).df[ticker]    
    k = df.index.to_pydatetime()
    
    trace1 = go.Scatter(x=[k[i].date() for i in range(len(k))], y=df['close'],mode='markers+lines',text=[x.strftime('%Y-%m-%d %H:%M:%S') for x in df.index])
    layout = go.Layout(title='Use lasso or box tool to select', xaxis=dict(rangeselector=dict(
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
        rangeslider=dict(visible=True), type="category"), height = 500)
    
    return {'data':[trace1],'layout':layout}

# Show result of selecting data with either box select or lasso
    
@app.callback(Output('display','children'), Output('nouse','children'), [Input('carGraph','selectedData')])
def selectData(selectData):
    if selectData:
        return str('Points in the following range will be added to the dataset: {}'.format(selectData)), str(selectData['points'])
    else:
        return str('None selected')
#Extract the 'text' component and use it to filter the dataframe and then create another graph
    
@app.callback(Output('filterGraph','figure'),[Input('carGraph','selectedData')], State('input-on-submit1', "value"),
            State('demo-dropdown', 'value'), 
            State('input-on-submit', "value"))
def selectData3(selectData, tspan, itspan, ticker):
    if selectData:
        api = alpaca.REST('PK7Z5SUF67ICPDK04R2M', 'ITAqIWxumbD67keejeh7yXTnrgSfnlZZZiXb759t', 'https://paper-api.alpaca.markets')
        df4 = api.get_barset(ticker, itspan, limit=tspan).df[ticker]
        filtList = []
        for i in range(len(selectData['points'])):
            filtList.append(selectData['points'][i]['text'])
        df4['time'] = [x.strftime('%Y-%m-%d %H:%M:%S') for x in df4.index]
        filtList = [str(x) for x in filtList]
        df4['time']=[str(x) for x in df4['time']]

        filtCars = df4[df4['time'].isin(filtList)]
    
        trace2 = go.Scatter(x=filtCars.index.to_pydatetime(), y=filtCars['close'], mode='markers+lines', text=filtCars.index)
        layout2 = go.Layout(title='Grabbed Data')
        return {'data':[trace2],'layout':layout2}  

    else:
        trace2 = go.Scatter()
        layout2 = go.Layout()
        return {'data':[trace2],'layout':layout2} 

if __name__ == '__main__':
    app.run_server(debug=True)
