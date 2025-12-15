# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 12:31:09 2025

@author: chris williams
"""

import dash
from dash import dcc, html, Input, Output, State
from urllib.parse import parse_qs, urlparse, unquote
import pandas as pd
import plotly.express as px

from dose_report import make_area_figure, load_arpansa_dose_monitoring_csv 

# Load data
df= load_arpansa_dose_monitoring_csv('exp_data.csv')

# Create the Dash app
app = dash.Dash(__name__, requests_pathname_prefix='/prmarea/', routes_pathname_prefix='/prmarea/', title='CHS PRM Area Results')
server = app.server  # Needed for deployment

# Layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(
        [dcc.Dropdown(
            id='centre_dropdown',
            options=[{'label': i, 'value': i} for i in sorted(df['Centre'].unique())],
            placeholder="Select an item...",
            searchable=True,
            multi=True,
            style={'width': '50%'},
            value=['MPRE']
        ),
        dcc.Dropdown(
            id='occupation_dropdown',
            options=[{'label': i, 'value': i} for i in df['Occupation'].unique()],
            placeholder="Select an item...",
            searchable=True,
            multi=True,
            value = [o for o in df['Occupation'].unique()],
            style={'width': '50%'}
        ),
        dcc.Dropdown(
            id='date_dropdown',
            options=[{'label': i, 'value': i} for i in sorted(df['quarter'].unique())],
            placeholder="Select a date...",
            value = '2022 1',
            style={'width': '50%'}
        ),
        dcc.RadioItems(
            options=[
            {"label": "Separate centres", "value": "separate_centres"},
            {"label": "Separate Occupations", "value": "separate_occupations"},
            {"label": "No subgroups", "value": "no_separation"}
            ],
            value="separate_occupations", # Default selected value(s)
            id="options_checkbox",
            inline=True # Display options horizontally
            ),
        ],
        id='dropdown-container',
    ),
    dcc.Graph(id='plot',
              style={'height':'700px'})
])


@app.callback(
    [Output('occupation_dropdown', 'options'), Output('occupation_dropdown','value')],
    Input('centre_dropdown','value')
    )
def update_occupation_choices(values):
    if values is not None:
        occupations = df.loc[df.Centre.isin(values)].Occupation.unique()
        #print(occupations)
        return occupations, occupations
    else:
        return

# Callback to update plot
@app.callback(
    Output('plot', 'figure'),
    [Input('centre_dropdown', 'value'), Input('occupation_dropdown', 'value'), Input('date_dropdown','value'), Input('options_checkbox','value')]
)
def update_plot(centre_values, occupation_values, date_value, options):
    separate_centres = 'separate_centres' == options
    separate_occupations = 'separate_occupations' == options
    tdf = df.loc[df.quarter >= date_value]
    fig = make_area_figure(tdf, centres = centre_values, occupations = occupation_values, separate_centres=separate_centres, separate_occupations=separate_occupations)
    fig.update_layout(title=f"Area PRM results")
    return fig







if __name__ == '__main__':
    app.run(debug=True, port=8051)








