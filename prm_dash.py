# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 12:31:09 2025

@author: chris williams
"""

import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
from urllib.parse import parse_qs, urlparse, unquote
import pandas as pd
import plotly.express as px
from dash import dash_table

from dose_report import plot_individual_figure, make_subtables, load_arpansa_dose_monitoring_csv


#%%
# Load data
df= load_arpansa_dose_monitoring_csv('exp_data.csv')

#%%
app = dash.Dash(__name__, requests_pathname_prefix='/prm/', routes_pathname_prefix='/prm/', title='CHS PRM Results')
server = app.server


app.layout = html.Div([
    html.H1("Canberra Health Services Personal Radiation Dose Monitoring Results ", style={'textAlign': 'center'}),
    dcc.Location(id='url', refresh=False),
    
    html.Div(
        [
            dcc.Dropdown(
            id='centre-dropdown',
            options=[{'label': i, 'value': i} for i in sorted(df['Centre'].unique())],
            placeholder="Showing all centres - select a centre to filter",
            searchable=True,
            style={'width': '50%'}
        ),
        dcc.Dropdown(
            id='user-dropdown',
            options=[{'label': i, 'value': i} for i in sorted(df['Wearer'].unique())],
            placeholder="Select a wearer...",
            searchable=True,
            style={'width': '50%'}
        ),
        dcc.RadioItems(
            options=[
            {"label": "Area monitoring (don't show people)", "value": "show_area_monitoring"},
            {"label": "Restrict missing D.O.B. (show people)", "value": "show_humans"},
            {"label": "Everything (show all wearers/areas)", "value": "show_all"}
            ],
            value="show_humans", # Default selected value(s)
            id="wearer_type_radio",
            inline=True # Display options horizontally
            )
        ],
        id='dropdown-container',
    ),
    html.H2('',id='area-info', style={'textAlign':'center'}),
    dcc.Graph(id='plot'),
        html.Hr(),
    dash_table.DataTable(
        id='data-table',
        columns=[#{'id':'CentreNo','name':'Center Number'},
                 {'id':'Wearer','name':'Name'},
                 {'id':'WearingStopDate', 'name':'Wearing end date'},
                 {'id':'PhotonHp10','name':'Absorbed dose (uSv, Hp10)'}],
        data=[],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'}
    )
])

#Callbacks

@app.callback(
    Output('user-dropdown','options'),
    [Input('centre-dropdown','value'), Input('wearer_type_radio', 'value')]
    )
def update_users(selected_centre, wearer_type):
    if wearer_type == 'show_humans':
        tdf = df.loc[df.DOB.notna()]
    elif wearer_type == 'show_area_monitoring':
        tdf = df.loc[df.DOB.isna()]
    else:
        tdf = df
        
    if selected_centre is not None:
        return sorted(tdf.loc[tdf.Centre==selected_centre].Wearer.unique())
    else:
        return sorted(tdf.Wearer.unique())


@app.callback(
    [Output('plot', 'figure'), Output('area-info','children'), Output('data-table', 'data')],
    Input('user-dropdown', 'value')
)
def update_plot(selected_value):
    try:
        wdf, cdf = make_subtables(df, selected_value)
    except IndexError:
        raise PreventUpdate
    fig = plot_individual_figure(wdf, cdf, df)
    text = f'{wdf.Centre.iloc[-1]} - {wdf.Occupation.iloc[-1]} - {selected_value}'
    output_table = prepare_presentation_table(wdf)
    return fig, text, output_table


# Helper functions

def prepare_presentation_table(wearer_df):
    "Reformat the dose history dataframe for display"
    out_df = wearer_df.loc[:,['Wearer','WearingStopDate','PhotonHp10']].copy() #'CentreNo', removed
    out_df.PhotonHp10 = out_df.PhotonHp10.fillna('<50').astype(str)
    out_df = out_df.sort_values('WearingStopDate',ascending=False)

    return out_df.to_dict('records')


if __name__ == '__main__':
    app.run(debug=True, port=8050)

