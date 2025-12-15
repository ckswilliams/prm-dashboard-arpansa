# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:30:14 2025

@author: chris williams
"""


import pandas as pd


import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm, binom
from scipy.optimize import minimize
import yaml

import plotly.graph_objects as go
import plotly.express as px


#%% Load in all the centre numbers. .

#Load from separate files
# note that setting a centre to blank will remove it from the dashboard.
# Occupations or centres not present will show up as numbers.

from pathlib import Path

HOME_DIR = Path(__file__).parent

with open(HOME_DIR / "centre_number_map.yaml", "r", encoding="utf-8") as f:
    centre_number_map = yaml.safe_load(f)
centre_number_map = {str(k):v for k,v in centre_number_map.items()}

with open(HOME_DIR / "occupation_map.yaml", "r", encoding="utf-8") as f:
    occupation_map = yaml.safe_load(f)
#occupation_map = {str(k):v for k,v in occupation_map.items()}


#%% debug and test
if __name__ == '__main__':
    import os
    import pathlib
    pathlib.Path(__file__).parent
    os.chdir(r'pathlib.Path(__file__).parent')
    df = load_arpansa_dose_monitoring_csv('exp_data.csv')
    
#%%


def load_arpansa_dose_monitoring_csv(fn):
    df= pd.read_csv(fn, index_col=False)
    df.Wearer = df.Wearer.str.strip()
    df.loc[df.PhotonHp10.isna(),'PhotonHp10'] = df.loc[df.PhotonHp10.isna(),'PhotonHp07'] # probably unnecessary, but if hp07 is available but hp10 isn't, use hp07 anywya
    df.WearingStopDate = pd.to_datetime(df.WearingStopDate, dayfirst=True)
    df = df.loc[~(df.PhotonHp10 > 10000)]
    df = df.loc[df.Wearer != 'NOT USED'] # remove 'unnused' badges
    df = df.loc[df.Wearer != 'CONTROL'] # remove control badges
    df = df.loc[df.Wearer != 'WEARER UNKNOWN'] # remove wearer unknown badges
    df = df.loc[df.Wearer != 'EXPOSED'] # remove wearer unknown badges

    df = df.loc[pd.to_numeric(df.Wearer,errors='coerce').isna()] # remove any wearer that just has a number
    df = df.loc[df.Wearer.notna()] # remove any missing wearer info
    df = df.loc[df.WearingStopDate.notna()] # remove anything that isn't assciated with a particular period of time
    df = df.loc[~df.Wearer.str.contains(r'\d')] # in fact, remove anything with a numeric
    df['quarter'] = df.WearingStopDate.dt.year.astype(str) + ' ' + df.WearingStopDate.dt.quarter.astype(str)
    df = df.copy()
    
    df['Centre'] = df.CentreNo.map(centre_number_map).fillna(df.CentreNo)
    df.Centre = df.Centre.astype('str')
    df = df.loc[~(df.Centre=='')] # Remove any centres that are assigned blank in the name mapping
    df['Occupation'] = df.Occupation.map(occupation_map).fillna(df.Occupation)
    df.Occupation = df.Occupation.astype('str')
    return df



#%% area figures

def make_area_figure(df, centres=None, occupations=None, separate_centres = True, separate_occupations = True, has_dob=True):
    
    
    cdf = df.copy()
    cdf.loc[cdf.PhotonHp10 < 50, 'PhotonHp10'] = np.nan
    if centres is not None:
        cdf = cdf.loc[lambda x: x.Centre.isin(centres)]
    if occupations is not None:
        cdf = cdf.loc[lambda x: x.Occupation.isin(occupations)]
    
    groupers = ['quarter']
    if separate_centres:
        groupers.append('Centre')
        category_col='Centre'
    elif separate_occupations:
        groupers.append('Occupation')
        category_col='Occupation'
    else:
        category_col=None
    
    cdf.PhotonHp10 = cdf.PhotonHp10.fillna(50)
    l = cdf.groupby(groupers).agg(
        #statistical_mean = ('PhotonHp10', lambda x: estimate_lognormal_mean_and_sigma(x,50)), #TOO computationally heavy to do for every quarter for every sub group, and also very innaccurate for small groups.
        mean = ("PhotonHp10",'mean'),
        max = ('PhotonHp10', 'max'),
        total_wearers=('PhotonHp10','count')
        )

    l = l.reset_index()
    
    return quarterly_overlay(l, quarter_col='quarter',category_col=category_col, y1_col='mean', y2_col='max',
                      title='PRM group results')
    


#%% individual figures

def make_subtables(df, wearer):
    wdf=df.loc[lambda x: x.Wearer==wearer]
    
    centers = wdf.CentreNo.unique()
    if len(centers) > 1:
        gfsdfgd
    
    cdf = df.loc[lambda x: x.CentreNo==centers[0]]
    cdf = cdf.copy()
    cdf.loc[cdf.PhotonHp10 < 50, 'PhotonHp10'] = np.nan
    
    adf = df.copy()
    adf.loc[adf.PhotonHp10 < 50, 'PhotonHp10'] = np.nan
    
    return wdf, cdf
    

def add_horizontal_line(fig, y, color='red', name='Threshold', dash='dash'):
    # Add dummy trace for legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color=color, dash=dash),
        name=name
    ))

    # Convert existing shapes to list and append new shape
    existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    existing_shapes.append(dict(
        type='line',
        xref='paper', x0=0, x1=1,
        yref='y', y0=y, y1=y,
        line=dict(color=color, dash=dash)
    ))
    fig.update_layout(shapes=existing_shapes)

    
    
def plot_individual_figure(wdf, cdf, adf):
    fig = px.bar(wdf, x='WearingStopDate', y='PhotonHp10')
    fig.data[0].name = "Wearer data"  # Rename the first trace
    fig.data[0].showlegend = True
    
    add_horizontal_line(fig, 50, "green", "Minimum reportable value")
       
    mean_c, std_c = estimate_lognormal_mean_and_sigma(cdf.PhotonHp10, 50)
    mean_a, std_a = estimate_lognormal_mean_and_sigma(adf.PhotonHp10, 50)
    
    add_horizontal_line(fig, mean_c, "orange", "Centre estimated average")
    add_horizontal_line(fig, mean_a, "red", "All staff estimated average")
    
    fig.update_layout(
        yaxis=dict(range=[0, max(wdf["PhotonHp10"].max(),mean_c)*1.08], title='Photon Absorbed dose (uSv, Hp10)'),
        xaxis=dict(title = 'Wearing Period End')
    )
    
    
    return fig
    

#%% Overkill approach for estimating population mean for a truncated, skewed, normal distrubtion.


import numpy as np
from scipy.stats import norm, truncnorm, binom
from scipy.optimize import minimize

def estimate_lognormal_mean_and_sigma(values, threshold):
    observed = values.dropna()
    observed = observed[observed > threshold].values
    n_total = len(values)
    n_above = len(observed)
    n_below = n_total - n_above

    if n_above == 0:
        print("No values above threshold.")
        return np.nan, np.nan

    # Log-transform observed values
    log_obs = np.log(observed)
    log_threshold = np.log(threshold)

    def neg_log_likelihood(params):
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        a = (log_threshold - mu) / sigma
        ll_obs = truncnorm.logpdf(log_obs, a=a, b=np.inf, loc=mu, scale=sigma)
        p_above = 1 - norm.cdf((log_threshold - mu) / sigma)
        ll_binom = binom.logpmf(n_above, n_total, p_above)
        return -np.sum(ll_obs) - ll_binom

    initial_mu = np.mean(log_obs)
    initial_sigma = np.std(log_obs)

    try:
        result = minimize(
            neg_log_likelihood,
            x0=[initial_mu, initial_sigma],
            bounds=[(None, None), (1e-6, None)]
        )
        if result.success:
            mu_hat, sigma_hat = result.x
            mean_lognormal = np.exp(mu_hat + 0.5 * sigma_hat**2)
            return mean_lognormal, sigma_hat
        else:
            print("Optimization failed:", result.message)
            fallback_mean = np.exp(initial_mu + 0.5 * initial_sigma**2)
            return fallback_mean, initial_sigma
    except Exception as e:
        print("Error during optimization:", e)
        fallback_mean = np.exp(initial_mu + 0.5 * initial_sigma**2)
        return fallback_mean, initial_sigma


#%%

def quarterly_overlay(
    agg: pd.DataFrame,
    quarter_col='quarter',
    category_col='category',
    y1_col='mean',
    y2_col='max',
    title='Quarterly categories: overlay Max over Mean'
):
    
    # Build multi-category x
    if category_col is not None:
        x = [agg[quarter_col], agg[category_col]]
    else:
        x = agg[quarter_col]
        
    fig = go.Figure()

    # Baseline (opaque)
    fig.add_bar(
        name='Mean',
        x=x,
        y=agg[y1_col],
        marker_color='#1f77b4',
        opacity=1.0,
   #     hovertemplate='Quarter=%{x[0]}<br>Category=%{x[1]}<br>Mean: %{y:.3f}<extra></extra>',
    )

    # Overlay (semi-transparent, same x)
    fig.add_bar(
        name='Max',
        x=x,
        y=agg[y2_col],
        marker_color='#ff7f0e',
        opacity=0.35,
   #     hovertemplate='Quarter=%{x[0]}<br>Category=%{x[1]}<br>Max: %{y:.3f}<extra></extra>',
    )

    fig.update_layout(
        barmode='overlay',      # key: overlay at same x
        bargap=0.10,
        bargroupgap=0.12,
        title=title,
        xaxis_title='Quarter',
        yaxis_title='Dose reading (uSv)',
        legend_title='Series'
    )

    # outline on the overlay so the taller bar edges are clear
    fig.data[1].marker.line.width = 1.2
    fig.data[1].marker.line.color = '#ff7f0e'

    return fig


