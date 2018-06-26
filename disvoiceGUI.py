
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 2018
@author: J. C. Vasquez-Correa
GITA research group - University of Antioquia - UdeA
Pattern recognition Lab - University of Erlangen Nuremberg
http:/jcvasquezc.wix.com/home
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools

from scipy.io.wavfile import read
from playsound import playsound
import plotly.graph_objs as go
import numpy as np
from scipy.io.wavfile import read
import os,sys,io
import easygui
import sounddevice

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import scipy.signal as sp
import pandas as pd
import base64
import pysptk

from scipy.integrate import cumtrapz
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error


path_app = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_app+"/phonation")
from phonation import phonationVowels
from GCI import SE_VQ_varF0, IAIF, get_vq_params
from glottal import glottal_features

sys.path.append(path_app+"/articulation")
from articulation import articulation_continuous

sys.path.append(path_app+"/prosody")
from prosody import prosody_static, intonation_duration


app = dash.Dash(__name__)
server = app.server
server.secret_key = os.environ.get('SECRET_KEY', 'my-secret-key')




app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})


app.css.append_css({'external_url': 'https://codepen.io/Booligoosh/pen/mKPpQp.css'})
#app.css.append_css({"external_url": "https://codepen.io/jakob-e/pen/deEPPG.css"})

colors = {
    'background': '#111111',
    'text': '#7F7F7F'
}

global signal, fs




def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )



app.layout = html.Div([

        html.Div([
            html.Title("Disvoice:"),
        ]),
        html.Div(["Disvoice:"], style={"font-size":"200%"}),
        html.Div(["Feature visualization of pathological voices"], style={"font-size":"150%"}),

        html.Hr(),
        html.Div([
            html.Button('Load audio',id='bt_load'),
            html.Div(id="play_text",children=""),
            html.Button('Play', id='bt_play'),
            html.Div(id='file_audio_disp', children=""),
        ], style={"columnCount":3, 'marginBottom': 25}),

        html.Div([
            html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[], mode = 'lines')],
                        layout=go.Layout(
                            title='Speech signal',
                            xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_speech',
                    ),
                ], className="six columns"),

            html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Heatmap(x=[], y=[], z=[])],
                        layout=go.Layout(
                            title='Spectrogram',
                            xaxis= dict(title= 'Time (s)', ticklen= 5),
                            yaxis= dict(title= 'Frequency (Hz)', ticklen= 5),
                            margin={'l': 50, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_spec',
                    )
                    ], className="six columns"),
        ], className="row"),

        html.Hr(),

        html.Div(["1. Phonation analysis"], style={"font-size":"150%"}),

        html.P('Phonation features are based on perturbation measures of the fundamental frequency an amplitude, and from the reconstructed glottal signal using an inverse-filter algorithm'),

        html.Div(["1.1. Perturbation"], style={"font-size":"120%"}),

        html.P('Perturbation features are computed from the fundamental frequency contour (Jitter and Amplitude Perturbation quotient (APQ)), and from the amplitude of the signal (Shimmer and Pitch Perturbation Quotient (PPQ))'),

        html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[], mode = 'lines')],
                        layout=go.Layout(
                            title='Fundamental Frequency (Hz)',
                            xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                            margin={'l': 60, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_F0',
                    ),
        ], style={'marginBottom': 25, 'marginTop': 25}),



        html.Div([
            html.Div(children=[
                ], className="six columns", id="feat_pert"),

            html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[], mode = 'lines')],
                        layout=go.Layout(height=300,
                            title='Perturbation features',
                            xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                            margin={'l': 60, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_pert_feat',
                    )


                    ], className="six columns"),
        ], className="row"),

        html.Hr(),
        html.Div(["1.2. Glottal"], style={"font-size":"120%"}),

        html.P('Glottal features are computed from the reconstructe glottal flow signal using an iterative adapatative inverse filter (IAIF) method'),

        html.P('Temporal features computed from the glottal flow include the variability of the glottal closure instants (GCI), the quasi opening qotient (QOQ), and the normalized amplitude quotient (NAQ)'),

        html.P('Spectral features computed from the glottal flow include the difference between the first two harmonics of the glottal flow (H1-H2), and the harmonics richness factor (HRF)'),

        html.Div([
            dcc.Graph(
                figure=go.Figure(
                    data=[go.Scatter(x=[], y=[], mode = 'lines', name="Speech signal", xaxis='x1', yaxis='y1'),
                          go.Scatter(x=[], y=[], mode = 'lines', name="Glottal flow", xaxis='x2', yaxis='y2'),
                          go.Scatter(x=[], y=[], name="GCI", xaxis='x2', yaxis='y2'),
                          go.Scatter(x=[], y=[], mode = 'lines', name="Glottal flow derivative", xaxis='x3', yaxis='y3')],
                    layout=go.Layout(
                        showlegend=True,
                        xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2, domain=[0, 1]),
                        yaxis= dict(ticklen= 5, zeroline= False, gridwidth= 2, domain=[0, .3]),
                        xaxis2= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2, domain=[0, 1]),
                        yaxis2= dict(ticklen= 5, zeroline= False, gridwidth= 2, domain=[.33, .63], anchor='x2'),
                        xaxis3= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2, domain=[0, 1], anchor='y3'),
                        yaxis3= dict(ticklen= 5, zeroline= False, gridwidth= 2, domain=[.66, 1]),
                        margin={'l': 40, 'b': 40, 't': 40, 'r': 40}
                    ),
                ),
                id='plot_glottal',
                ),
        ], style={'marginBottom': 25, 'marginTop': 25}),

        html.Div([
            html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[], mode = 'lines')],
                        layout=go.Layout(
                            title='Glottal spectrum',
                            xaxis= dict(title= 'Frequency (Hz)', ticklen= 5, zeroline= False, gridwidth= 2),
                            yaxis= dict(title= 'Amplitude (dB)', ticklen= 5, zeroline= False, gridwidth= 2),
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_glottalfft',
                    ),
                ], style={'marginBottom': 25, 'marginTop': 25}, className="six columns"),

            html.Div([

                dcc.RadioItems(id="spec_gradio",
                    options=[
                        {'label': 'Glottal flow', 'value': 'GF'},
                        {'label': 'Glottal flow derivative', 'value': 'GFD'}
                    ],
                    value='GF', labelStyle={'display': 'inline-block'}
                ),
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Heatmap(x=[], y=[], z=[])],
                        layout=go.Layout(
                            title='Spectrogram Glottal',
                            xaxis= dict(title= 'Time (s)', ticklen= 5),
                            yaxis= dict(title= 'Frequency (Hz)', ticklen= 5),
                            margin={'l': 50, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='glottal_spec',
                    )
                    ], className="six columns"),
        ], className="row"),




        html.Div([
            html.Div(children=[
                ], className="four columns", id="feat_glottal"),

            html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[], mode = 'lines')],
                        layout=go.Layout(height=400,
                            title='Glottal features',
                            xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                            margin={'l': 60, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_glottal_feat1',
                    ),

                    ], className="four columns"),


            html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[], mode = 'lines')],
                        layout=go.Layout(height=400,
                            title='Glottal features',
                            xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                            margin={'l': 60, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_glottal_feat2',
                    ),

                    ], className="four columns"),

        ], className="row"),



        html.Hr(),

        html.Div(["2. Articulation analysis"], style={"font-size":"150%"}),

        html.P('Articulation analysis is based on the contour of the formant frequencies (resonances that appear in the vocal tract), and from the energy content distributed in 22 frequency bands according to the Bark scale the transitions from voiced to unvoiced segments (offset) and from unvoiced to voiced segments (onset)'),

        html.Div([
            html.Div(children=[
                ], className="four columns", id="feat_formants"),

            html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[], mode = 'lines')],
                        layout=go.Layout(height=300,
                            title='Formants',
                            xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                            margin={'l': 60, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_formants1',
                    ),

                    ], className="four columns"),


            html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[], mode = 'lines')],
                        layout=go.Layout(height=300,
                            #title='Glottal features',
                            xaxis= dict(title= 'F1 (Hz)', ticklen= 5, zeroline= False, gridwidth= 2),
                            yaxis= dict(title= 'F2 (Hz)', ticklen= 5, zeroline= False, gridwidth= 2),
                            margin={'l': 60, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_formants2',
                    ),

                    ], className="four columns"),

        ], style={'marginBottom': 25, 'marginTop': 25}, className="row"),



        html.Div([

            html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[], mode = 'lines')],
                        layout=go.Layout(
                            title='Bark band energies in Onset/Offset Transitions',
                            xaxis= dict(title= 'Frequency (kHz)', ticklen= 5, zeroline= False, gridwidth= 2),
                            margin={'l': 60, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_BBE_cont',
                    ),

                    ], className="six columns"),

            html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[dict(values=np.zeros(22), labels=["Band "+str(j) for j in range(1,23)], name="Onset", hoverinfo="label+percent+name",hole=0.4, type="pie", domain= {"x": [0, .48]}),
                            dict(values=np.zeros(22), labels=["Band "+str(j) for j in range(1,23)], name="Offset", hoverinfo="label+percent+name",hole=0.4, type="pie", domain= {"x": [0.52, 1]},textposition="inside")
                        ],
                        layout=go.Layout(
                            title='Energy content in Bark scale',

                            annotations=[{"text":"Onset", "x":0.20, "y":0.5,"showarrow": False},{"text":"Offset", "x":0.80, "y":0.5, "showarrow": False}],
                            margin={'l': 60, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_BBEpie',
                    ),

                    ], className="six columns"),

        ], className="row"),



        html.Hr(),

        html.Div(["3. Prosody analysis"], style={"font-size":"150%"}),

        html.P('Prosdy analysis contains features related to the fundamental frequency, energy, and duration of the segments that form the speech signal'),



        html.Div(["3.1. Fundamental frequency based features"], style={"font-size":"150%"}),

        html.Div([
            html.Div(children=[
                ], className="six columns", id="feat_pros_pitch"),

            html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[], mode = 'lines')],
                        layout=go.Layout(
                            height=300,
                            title='Pitch',
                            xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                            margin={'l': 60, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_pros_pitch',
                    )
                    ], className="six columns"),
        ], className="row"),

        html.Hr(),

        html.Div(["3.2. Energy based features"], style={"font-size":"150%"}),

        html.Div([
            html.Div(children=[
                ], className="six columns", id="feat_pros_energy"),

            html.Div([
                html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[], mode = 'lines')],
                        layout=go.Layout(height=300,
                            title='Energy',
                            xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                            yaxis= dict(title= 'Energy (dB)', ticklen= 5, zeroline= False, gridwidth= 2),
                            margin={'l': 60, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_pros_energy',
                    )], style={'marginBottom': 25, 'marginTop': 25}),
                html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[], mode = 'lines')],
                        layout=go.Layout(height=300,
                            xaxis= dict(title= '# of segments', ticklen= 5, zeroline= False, gridwidth= 2),
                            yaxis= dict(title= 'Energy (dB)', ticklen= 5, zeroline= False, gridwidth= 2),
                            margin={'l': 60, 'b': 40, 't': 40, 'r': 40}
                        ),
                    ),
                    id='plot_pros_energy2',
                    )])
                    ], className="six columns"),
        ], className="row"),


        html.Hr(),

        html.Div(["3.3. Duration based features"], style={"font-size":"150%"}),

        html.Div([
            html.Div(children=[
                ], className="six columns", id="feat_pros_duration1"),
            html.Div(children=[
                ], className="six columns", id="feat_pros_duration2"),
        ], className="row"),



],style={'width': '100%','padding': ['0%'  '5%' '0%' '5%']})


@app.callback(
    dash.dependencies.Output('feat_pros_duration1', 'children'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def table_feat_pros_energy(fname):
    feat_names=["Voiced rate [# voiced /s]", "Unvoiced rate [# unvoiced /s]", "Pause rate [# pauses /s]", "Voiced duration [ms]", "Unvoiced duration [ms]", "Pause duration [ms]",
                "\% Voiced", "\% Unvoiced", "% Pause "]
    if fname==None:
        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':["","","","","","","","",""],
                                    'Value Standard deviation':["","","","","","","","",""]
                                    })
        Table=generate_table(dataframe_feat)
        return Table
    if os.path.exists(fname):
        F0, logE, mF0, sF0, mmF0, mlogE, slogE, mmlogE, Vrate, avgdurv, stddurv, Silrate, avgdurs, stddurs, F0varsemi=prosody_static(fname, False)
        avgF0slopes,stdF0slopes,MSEF0, SVU,VU,UVU,VVU,VS,US,URD,VRD,URE,VRE,PR,maxvoicedlen,maxunvoicedlen,minvoicedlen,minunvoicedlen,rvuv,energyslope,RegCoefenergy,msqerrenergy,RegCoeff0,meanNeighborenergydiff,stdNeighborenergydiff, F0_rec, F0, venergy, uenergy  = intonation_duration(fname, flag_plots=False)

        VRavg=np.round(Vrate,2)
        DVavg=np.round(avgdurv, 2)
        DVstd=np.round(stddurv, 2)
        PRavg=np.round(Silrate,2)
        DPavg=np.round(avgdurs, 2)
        DPstd=np.round(stddurs, 2)
        DUavg=np.round((DVavg/VU),2)
        DUstd=np.round(URD,2)
        URavg=np.round(rvuv,2)
        VDegavg=np.round(100*DVavg/(DVavg+DUavg+DPavg),2)
        UDegavg=np.round(100*DUavg/(DVavg+DUavg+DPavg),2)
        PDegavg=np.round(100*DPavg/(DVavg+DUavg+DPavg),2)

        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':[VRavg, URavg, PRavg, DVavg, DUavg, DPavg, VDegavg, UDegavg, PDegavg],
                                    'Value Standard deviation':["", "", "", DVstd, DUstd, DPstd, "-", "-", "-"]
                                    })
        Table=generate_table(dataframe_feat)
    else:
        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':["","","","","","","","",""],
                                    'Value Standard deviation':["","","","","","","","",""]
                                    })
        Table=generate_table(dataframe_feat)
    return Table















@app.callback(
    dash.dependencies.Output('plot_pros_energy2', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def update_plot_energy(fname):
    global signal, fs
    Onshade='rgba(44, 160, 101, 0.5)'
    if fname==None:
        return ""
    print(fname)
    if os.path.exists(fname):

        F0, logE, mF0, sF0, mmF0, mlogE, slogE, mmlogE, Vrate, avgdurv, stddurv, Silrate, avgdurs, stddurs, F0varsemi=prosody_static(fname, False)
        avgF0slopes,stdF0slopes,MSEF0, SVU,VU,UVU,VVU,VS,US,URD,VRD,URE,VRE,PR,maxvoicedlen,maxunvoicedlen,minvoicedlen,minunvoicedlen,rvuv,energyslope,RegCoefenergy,msqerrenergy,RegCoeff0,meanNeighborenergydiff,stdNeighborenergydiff, F0_rec, F0, venergy, uenergy  = intonation_duration(fname, flag_plots=False)

        pnz=np.where(F0>0)
        sp=spearmanr(F0, F0_rec)

        pv=np.polyfit(np.arange(len(venergy)),venergy,1)
        pu=np.polyfit(np.arange(len(uenergy)),uenergy,1)
        recEv=np.arange(len(venergy))*pv[0]+pv[1]
        recEu=np.arange(len(uenergy))*pu[0]+pu[1]

    else:
        venergy=[]
        uenergy=[]
        recEv=[]
        recEu=[]
    return go.Figure(data=[go.Scatter(x=np.arange(len(venergy)), y=venergy, mode = 'lines+markers', name="Energy voiced segments"),
                           go.Scatter(x=np.arange(len(uenergy)), y=uenergy, mode = 'lines+markers', name="Energy unvoiced segments"),
                           go.Scatter(x=np.arange(len(recEv)), y=recEv, mode = 'lines+markers', name="Linear regression Energy voiced segments"),
                           go.Scatter(x=np.arange(len(recEu)), y=recEu, mode = 'lines+markers', name="Linear regression Energy unvoiced segments")],
                     layout=go.Layout(showlegend=True, xaxis= dict(title= '# segments', ticklen= 5, zeroline= False, gridwidth= 2),
                                    yaxis= dict(title= 'Energy (dB)', ticklen= 5, zeroline= False, gridwidth= 2 , range=[min(uenergy), max(venergy)+4]),
                                    margin={'l': 60, 'b': 40, 't': 50, 'r': 40},
                     ))

@app.callback(
    dash.dependencies.Output('plot_pros_energy', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def update_plot_eergy(fname):
    global signal, fs
    Onshade='rgba(44, 160, 101, 0.5)'
    if fname==None:
        return ""
    print(fname)
    if os.path.exists(fname):
        fs, signal=read(fname)
        F0, logE, mF0, sF0, mmF0, mlogE, slogE, mmlogE, Vrate, avgdurv, stddurv, Silrate, avgdurs, stddurs, F0varsemi=prosody_static(fname, False)
        tad=len(signal)/(fs*len(logE))
        t=np.arange(len(logE))*tad
    else:
        t=[]
        F0=[]
        F0_rec=[]
    return go.Figure(data=[go.Scatter(x=t, y=logE, mode = 'lines', name="Energy contour")],
                     layout=go.Layout(title="Energy contour", xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                                    yaxis= dict(title= 'Energy (dB)', ticklen= 5, zeroline= False, gridwidth= 2),
                                    margin={'l': 60, 'b': 40, 't': 50, 'r': 40},
                     ))


@app.callback(
    dash.dependencies.Output('feat_pros_energy', 'children'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def table_feat_pros_energy(fname):
    feat_names=["Energy (dB)", "Voiced Energy (dB)", "Unvoiced Energy (dB)", "Tilt energy voiced", "Tilt energy unvoiced", "MSE energy regression voiced segments", "MSE energy regression unvoiced segments",
                "Spearman Correlation for energy regression in voiced segments", "Spearman Correlation for energy regression in unvoiced segments"]
    if fname==None:
        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':["","","","","","","","",""],
                                    'Value Standard deviation':["","","","","","","","",""]
                                    })
        Table=generate_table(dataframe_feat)
        return Table
    if os.path.exists(fname):
        F0, logE, mF0, sF0, mmF0, mlogE, slogE, mmlogE, Vrate, avgdurv, stddurv, Silrate, avgdurs, stddurs, F0varsemi=prosody_static(fname, False)
        avgF0slopes,stdF0slopes,MSEF0, SVU,VU,UVU,VVU,VS,US,URD,VRD,URE,VRE,PR,maxvoicedlen,maxunvoicedlen,minvoicedlen,minunvoicedlen,rvuv,energyslope,RegCoefenergy,msqerrenergy,RegCoeff0,meanNeighborenergydiff,stdNeighborenergydiff, F0_rec, F0, venergy, uenergy  = intonation_duration(fname, flag_plots=False)

        pnz=np.where(F0>0)
        sp=spearmanr(F0, F0_rec)

        Eavg=np.round(mlogE, 2)
        Estd=np.round(slogE, 2)
        EVavg=np.round(np.mean(venergy),2)
        EVstd=np.round(np.std(venergy),2)
        EUavg=np.round(np.mean(uenergy),2)
        EUstd=np.round(np.std(uenergy),2)

        pv=np.polyfit(np.arange(len(venergy)),venergy,1)
        TiltEV=np.round(pv[0],2)
        pu=np.polyfit(np.arange(len(uenergy)),uenergy,1)
        TiltEU=np.round(pu[0],2)
        recEv=np.arange(len(venergy))*pv[0]+pv[1]
        recEu=np.arange(len(uenergy))*pu[0]+pu[1]
        MSEv=np.round(mean_squared_error(venergy, recEv),2)
        MSEu=np.round(mean_squared_error(uenergy, recEu),2)

        corrEv=np.round(spearmanr(venergy, recEv)[0],2)
        corrEu=np.round(spearmanr(uenergy, recEu)[0],2)

        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':[Eavg, EVavg, EUavg, TiltEV, TiltEU, MSEv, MSEu, corrEv, corrEu],
                                    'Value Standard deviation':[Estd, EVstd, EUstd, "-", "-", "-", "-", "-", "-"]
                                    })
        Table=generate_table(dataframe_feat)
    else:
        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':["","","","","","","","",""],
                                    'Value Standard deviation':["","","","","","","","",""]
                                    })
        Table=generate_table(dataframe_feat)
    return Table


@app.callback(
    dash.dependencies.Output('plot_pros_pitch', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def update_plot(fname):
    global signal, fs
    Onshade='rgba(44, 160, 101, 0.5)'
    if fname==None:
        return ""
    print(fname)
    if os.path.exists(fname):
        avgF0slopes,stdF0slopes,MSEF0, SVU,VU,UVU,VVU,VS,US,URD,VRD,URE,VRE,PR,maxvoicedlen,maxunvoicedlen,minvoicedlen,minunvoicedlen,rvuv,energyslope,RegCoefenergy,msqerrenergy,RegCoeff0,meanNeighborenergydiff,stdNeighborenergydiff, F0_rec, F0, venergy, uenergy  = intonation_duration(fname, flag_plots=False)
        tad=len(signal)/(fs*len(F0))
        t=np.arange(len(F0))*tad
    else:
        t=[]
        F0=[]
        F0_rec=[]
    return go.Figure(data=[go.Scatter(x=t, y=F0, mode = 'lines', name="F0"),go.Scatter(x=t, y=F0_rec, mode = 'lines', name="Reconstructed F0 with a linear regression", fillcolor=Onshade), ],
                     layout=go.Layout(height=300, xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                                    yaxis= dict(title= 'Frequency (Hz)', ticklen= 5, zeroline= False, gridwidth= 2,  range=[0, max(F0)+20]),
                                    margin={'l': 60, 'b': 40, 't': 50, 'r': 40},
                     ))

@app.callback(
    dash.dependencies.Output('feat_pros_pitch', 'children'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def table_feat_pros_pitch(fname):
    feat_names=["F0 (Hz)", "Tilt F0", "MSE F0 regression", "Spearman Correlation for F0 regression"]
    if fname==None:
        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':["","","",""],
                                    'Value Standard deviation':["","","",""]
                                    })
        Table=generate_table(dataframe_feat)
        return Table
    if os.path.exists(fname):

        avgF0slopes,stdF0slopes,MSEF0, SVU,VU,UVU,VVU,VS,US,URD,VRD,URE,VRE,PR,maxvoicedlen,maxunvoicedlen,minvoicedlen,minunvoicedlen,rvuv,energyslope,RegCoefenergy,msqerrenergy,RegCoeff0,meanNeighborenergydiff,stdNeighborenergydiff, F0_rec, F0, venergy, uenergy  = intonation_duration(fname, flag_plots=False)
        pnz=np.where(F0>0)
        sp=spearmanr(F0, F0_rec)

        F0avg=np.round(np.mean(F0[pnz]), 2)
        F0std=np.round(np.std(F0[pnz]), 2)
        tiltF0avg=np.round(avgF0slopes, 2)
        tiltF0std=np.round(stdF0slopes, 2)
        MSEF0avg=np.round(MSEF0,2)
        RegF0avg=np.round(sp[0],2)

        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':[F0avg, tiltF0avg, MSEF0avg, RegF0avg],
                                    'Value Standard deviation':[F0std, tiltF0std, "-", "-"]
                                    })
        Table=generate_table(dataframe_feat)
    else:
        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':["","","",""],
                                    'Value Standard deviation':["","","",""]
                                    })
        Table=generate_table(dataframe_feat)
    return Table

@app.callback(
    dash.dependencies.Output('plot_BBEpie', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def plot_BBE2(fname):

    if fname==None:
        return ""
    if os.path.exists(fname):

        BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1, DF1, DDF1, F2, DF2, DDF2=articulation_continuous(fname, False, pitch_method="rapt")
        BBEonavg=np.mean(BBEon,0)
        BBEoffavg=np.mean(BBEoff,0)
        BBEonavg=BBEonavg+np.abs(np.min(BBEonavg))
        BBEoffavg=BBEoffavg+np.abs(np.min(BBEoffavg))
    else:
        BBEonavg=[]
        BBEoffavg=[]

    return go.Figure(
            data=[dict(values=BBEonavg, labels=["Band "+str(j) for j in range(1,23)], name="Onset", hoverinfo="label+percent+name",hole=0.4, type="pie", domain= {"x": [0, .48]},textposition="inside"),
                dict(values=BBEoffavg, labels=["Band "+str(j) for j in range(1,23)], name="Offset", hoverinfo="label+percent+name",hole=0.4, type="pie", domain= {"x": [0.52, 1]},textposition="inside")
            ],
            layout=go.Layout(
                title='Energy content in Bark scale',

                annotations=[{"text":"Onset", "x":0.20, "y":0.5,"showarrow": False},{"text":"Offset", "x":0.80, "y":0.5, "showarrow": False}],
                margin={'l': 60, 'b': 40, 't': 40, 'r': 40}
            ),
        )


@app.callback(
    dash.dependencies.Output('plot_BBE_cont', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def plot_BBE1(fname):

    Onshade='rgba(44, 160, 101, 0.5)'
    On='rgb(44, 160, 101)'
    Off='rgb(44, 100, 160)'
    Offshade='rgba(44, 100, 160, 0.5)'
    F=[0.1, 0.2, 0.3, 0.4, 0.51, 0.63, 0.77, 0.92, 1.08, 1.27, 1.48, 1.72, 2, 2.32, 2.7, 3.15, 3.7, 4.4, 5.3, 6.4, 7.7, 9.5]

    if fname==None:
        return ""
    if os.path.exists(fname):

        BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1, DF1, DDF1, F2, DF2, DDF2=articulation_continuous(fname, False, pitch_method="rapt")
        BBEonavg=np.mean(BBEon,0)
        BBEoffavg=np.mean(BBEoff,0)

        BBEonstd=np.std(BBEon,0)
        BBEoffstd=np.std(BBEoff,0)

    else:
        BBEonavg=np.asarray([])
        BBEoffavg=np.asarray([])
        BBEonstd=np.asarray([])
        BBEoffstd=np.asarray([])
        F=[]

    return go.Figure(data=[go.Scatter(x=F, y=BBEonavg, mode = 'lines', name="Onset Avg.", line=dict(color=On)),
                            go.Scatter(x=F, y=BBEonavg-BBEonstd, mode = 'lines', name="Onset Avg-Std.", line=dict(color=On), fillcolor=Onshade),
                            go.Scatter(x=F, y=BBEonavg+BBEonstd, mode = 'lines', name="Onset Avg+Std.", fill="tonexty", line=dict(color=On), fillcolor=Onshade),
                            go.Scatter(x=F, y=BBEoffavg, mode = 'lines', name="Offset Avg.", line=dict(color=Off)),
                            go.Scatter(x=F, y=BBEoffavg-BBEoffstd, mode = 'lines', name="Offset Avg-Std.", fill="tonexty", line=dict(color=Off), fillcolor=Offshade),
                            go.Scatter(x=F, y=BBEoffavg+BBEoffstd, mode = 'lines', name="Offset Avg+Std.", fill="tonexty", line=dict(color=Off), fillcolor=Offshade)],
                     layout=go.Layout(showlegend=True, legend=dict(x=0.6, y=1), title='Bark band energies in Onset/Offset Transitions',xaxis= dict(title= 'Frequency (Hz)', ticklen= 5, zeroline= False, gridwidth= 2),
                     yaxis= dict(title= 'Amplitude (dB)', ticklen= 5, zeroline= False, gridwidth= 2), margin={'l': 60, 'b': 40, 't': 40, 'r': 40},
                     ))


@app.callback(
    dash.dependencies.Output('plot_formants2', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def plot_formants2(fname):
    if fname==None:
        return ""
    if os.path.exists(fname):

        BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1, DF1, DDF1, F2, DF2, DDF2=articulation_continuous(fname, False, pitch_method="rapt")

        F1avg=np.round(F1.mean(0),2)
        F2avg=np.round(F2.mean(0),2)
        F1std=np.round(F1.std(0),2)
        F2std=np.round(F2.std(0),2)

    else:
        F1=np.asarray([])
        F2=np.asarray([])
        t=np.asarray([])
        F1avg=np.asarray([])
        F1std=np.asarray([])
        F2avg=np.asarray([])
        F2std=np.asarray([])

    return go.Figure(data=[go.Scatter(x=F1, y=F2, mode = 'markers')],
                     layout=go.Layout(title='Formant dispersion',xaxis= dict(title= 'F1 (Hz)', ticklen= 5, zeroline= False, gridwidth= 2, range=[100, 1500]),
                     yaxis= dict(title= 'F2 (Hz)', ticklen= 5, zeroline= False, gridwidth= 2, range=[100, 2500]), margin={'l': 60, 'b': 40, 't': 40, 'r': 40},
                     shapes=[dict(type='circle', layer='below', xref='x', yref='y', x0=F1avg-F1std, y0=F2avg-F2std, x1=F1avg+F1std, y1=F2avg+F2std,  fillcolor='rgba(44, 160, 101, 0.5)',
                     line=dict(color= 'rgb(44, 160, 101)'))]
                     ))



@app.callback(
    dash.dependencies.Output('plot_formants1', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def plot_formants1(fname):
    if fname==None:
        return ""
    if os.path.exists(fname):

        BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1, DF1, DDF1, F2, DF2, DDF2=articulation_continuous(fname, False, pitch_method="rapt")
        fs, signal=read(fname)
        tad=len(signal)/(fs*len(F1))
        t=np.arange(len(F1))*tad

    else:
        F1=[]
        F2=[]
        t=[]

    return go.Figure(data=[go.Scatter(x=t, y=F1, mode = 'lines', name="F1"), go.Scatter(x=t, y=F2, mode = 'lines', name="F2")],
                     layout=go.Layout(showlegend=True, legend=dict(x=0.6, y=1), title='Formants',xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                     yaxis= dict(title= 'Frequency (Hz)', ticklen= 5, zeroline= False, gridwidth= 2), margin={'l': 60, 'b': 40, 't': 40, 'r': 40},
                     ))

@app.callback(
    dash.dependencies.Output('feat_formants', 'children'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def table_feat_glot(fname):
    feat_names=["F1 (Hz)", "F2 (Hz)",  "Delta F1 (Hz/s)", "Delta F2 (Hz/s)"]
    if fname==None:
        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':["","","","",""],
                                    'Value Standard deviation':["","","","",""]
                                    })
        Table=generate_table(dataframe_feat)
        return Table
    if os.path.exists(fname):

        BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1, DF1, DDF1, F2, DF2, DDF2=articulation_continuous(fname, False, pitch_method="rapt")

        F1avg=np.round(F1.mean(0),2)
        F2avg=np.round(F2.mean(0),2)
        DF1avg=np.round(DF1.mean(0),2)
        DF2avg=np.round(DF2.mean(0),2)
        F1std=np.round(F1.std(0),2)
        F2std=np.round(F2.std(0),2)
        DF1std=np.round(DF1.std(0),2)
        DF2std=np.round(DF2.std(0),2)

        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':[F1avg, F2avg, DF1avg, DF2avg],
                                    'Value Standard deviation':[F1std, F2std, DF1std, DF2std]
                                    })
        Table=generate_table(dataframe_feat)
    else:
        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':["","","",""],
                                    'Value Standard deviation':["","","",""]
                                    })
        Table=generate_table(dataframe_feat)
    return Table


@app.callback(
    dash.dependencies.Output('plot_glottal_feat2', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def update_plot_glot_feat2(fname):
    if fname==None:
        return ""
    if os.path.exists(fname):

        varGCIt, avgNAQt, varNAQt, avgQOQt, varQOQt, avgH1H2t, varH1H2t, avgHRFt, varHRFt  = glottal_features(fname, False)
        fs, signal=read(fname)
        tad=len(signal)/(fs*len(avgH1H2t))
        t=np.arange(len(avgH1H2t))*tad

    else:
        avgH1H2t=[]
        t=[]

    return go.Figure(data=[go.Scatter(x=t, y=avgH1H2t, mode = 'lines', name="H1-H2")],
                     layout=go.Layout(showlegend=True, legend=dict(x=0.6, y=1), title='Glottal spectral features',xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                     yaxis= dict(title= 'Frequency (Hz)', ticklen= 5, zeroline= False, gridwidth= 2), margin={'l': 60, 'b': 40, 't': 40, 'r': 40},
                     ))

@app.callback(
    dash.dependencies.Output('plot_glottal_feat1', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def update_plot_glot_feat1(fname):
    if fname==None:
        return ""
    if os.path.exists(fname):

        varGCIt, avgNAQt, varNAQt, avgQOQt, varQOQt, avgH1H2t, varH1H2t, avgHRFt, varHRFt  = glottal_features(fname, False)
        fs, signal=read(fname)
        tad=len(signal)/(fs*len(varNAQt))
        t=np.arange(len(varNAQt))*tad

    else:
        avgNAQt=[]
        avgQOQt=[]
        t=[]

    return go.Figure(data=[go.Scatter(x=t, y=avgNAQt, mode = 'lines', name="NAQ"), go.Scatter(x=t, y=avgQOQt, mode = 'lines', name="QOQ")],
                     layout=go.Layout(showlegend=True, legend=dict(x=0.6, y=1), title='Glottal closing features',xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                     yaxis= dict(title= '', ticklen= 5, zeroline= False, gridwidth= 2), margin={'l': 60, 'b': 40, 't': 40, 'r': 40},
                     ))

@app.callback(
    dash.dependencies.Output('feat_glottal', 'children'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def table_feat_glot(fname):
    feat_names=["GCI variability (ms)", "NAQ",  "QOQ", "H1-H2 (Hz)", "HRF"]
    if fname==None:
        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':["","","","",""],
                                    'Value Standard deviation':["","","","",""]
                                    })
        Table=generate_table(dataframe_feat)
        return Table
    if os.path.exists(fname):

        varGCIt, avgNAQt, varNAQt, avgQOQt, varQOQt, avgH1H2t, varH1H2t, avgHRFt, varHRFt  = glottal_features(fname, False)

        gciavg=np.round(varGCIt.mean(0)*1000,2)
        NAQavg=np.round(avgNAQt.mean(0),2)
        QOQavg=np.round(avgQOQt.mean(0),2)
        QOQavg=np.round(avgQOQt.mean(0),2)
        H1H2avg=np.round(avgH1H2t.mean(0),2)
        HRFavg=np.round(avgHRFt.mean(0),2)

        gcistd=np.round(varGCIt.std(0)*1000,2)
        NAQstd=np.round(avgNAQt.std(0),2)
        QOQstd=np.round(avgQOQt.std(0),2)
        QOQstd=np.round(avgQOQt.std(0),2)
        H1H2std=np.round(avgH1H2t.std(0),2)
        HRFstd=np.round(avgHRFt.std(0),2)



        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':[gciavg, NAQavg, QOQavg, H1H2avg, HRFavg],
                                    'Value Standard deviation':[gcistd, NAQavg, QOQstd, H1H2std, HRFstd]
                                    })
        Table=generate_table(dataframe_feat)
    else:
        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':["","","","",""],
                                    'Value Standard deviation':["","","","",""]
                                    })
        Table=generate_table(dataframe_feat)
    return Table

@app.callback(
    dash.dependencies.Output('glottal_spec', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children'), dash.dependencies.Input('spec_gradio', 'value')])
def compute_specgram(fname, spec_gradio):
    if fname==None:
        return ""
    if os.path.exists(fname):

        fs, data_audio=read(fname)
        data_audio=data_audio-np.mean(data_audio)
        data_audio=data_audio/max(abs(data_audio))
        data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
        f0=pysptk.sptk.rapt(data_audiof, fs, int(0.01*fs), min=20, max=500, voice_bias=-0.2, otype='f0')
        glottal_flow=[]
        Dglottal_flow=[]

        GCI=SE_VQ_varF0(data_audio,fs, f0=f0)
        Dglottal_flow=IAIF(data_audio,fs,GCI)
        Dglottal_flow=Dglottal_flow-np.mean(Dglottal_flow)
        Dglottal_flow=Dglottal_flow/max(abs(Dglottal_flow))

        glottal_flow=np.zeros(len(Dglottal_flow))
        for n in range(len(GCI)-1):
            start=int(GCI[n])
            stop=int(GCI[n+1])
            frame_int=Dglottal_flow[start:stop]
            tin=np.arange(len(frame_int))/fs
            glottal_flow_i=cumtrapz(frame_int, tin)
            glottal_flow[start+1:stop]=glottal_flow_i

        #glottal_flow_i=cumtrapz(Dglottal_flow)
        glottal_flow=glottal_flow-np.mean(glottal_flow)
        glottal_flow=glottal_flow/max(abs(glottal_flow))

        glottal_flow=glottal_flow[int(0.02*fs):int(len(glottal_flow)-0.1*fs)]
        Dglottal_flow=Dglottal_flow[int(0.02*fs):int(len(Dglottal_flow)-0.1*fs)]


        if spec_gradio=="GF":
            y, x, z=sp.spectrogram(glottal_flow, fs, "hamming", nfft=2048, mode="magnitude", scaling="spectrum")
            titles="Glottal flow"
        else:
            y, x, z=sp.spectrogram(Dglottal_flow, fs, "hamming", nfft=2048, mode="magnitude", scaling="spectrum")
            titles="Glottal flow derivative"
        z=20*np.log10(z)+20
        y=y/1000
    else:
        x=[]
        y=[]
        z=[]
        titles=""
    return go.Figure(data=[go.Heatmap(x=x, y=y, z=z, colorscale='Viridis')],
                     layout=go.Layout(title=titles,xaxis= dict(title= 'Time (s)', ticklen= 5), yaxis= dict(title= 'Frequency (kHz)', ticklen= 5), margin={'l': 60, 'b': 40, 't': 40, 'r': 40},
            ))

@app.callback(
    dash.dependencies.Output('plot_glottalfft', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def update_plot_glottalfft(fname):
    if fname==None:
        return ""
    if os.path.exists(fname):

        fs, data_audio=read(fname)
        data_audio=data_audio-np.mean(data_audio)
        data_audio=data_audio/max(abs(data_audio))
        data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
        f0=pysptk.sptk.rapt(data_audiof, fs, int(0.01*fs), min=20, max=500, voice_bias=-0.2, otype='f0')
        glottal_flow=[]
        Dglottal_flow=[]

        GCI=SE_VQ_varF0(data_audio,fs, f0=f0)
        Dglottal_flow=IAIF(data_audio,fs,GCI)
        Dglottal_flow=Dglottal_flow-np.mean(Dglottal_flow)
        Dglottal_flow=Dglottal_flow/max(abs(Dglottal_flow))

        glottal_flow=np.zeros(len(Dglottal_flow))
        for n in range(len(GCI)-1):
            start=int(GCI[n])
            stop=int(GCI[n+1])
            frame_int=Dglottal_flow[start:stop]
            tin=np.arange(len(frame_int))/fs
            glottal_flow_i=cumtrapz(frame_int, tin)
            glottal_flow[start+1:stop]=glottal_flow_i

        #glottal_flow_i=cumtrapz(Dglottal_flow)
        glottal_flow=glottal_flow-np.mean(glottal_flow)
        glottal_flow=glottal_flow/max(abs(glottal_flow))


        GFFT=10*np.log10(np.abs(np.fft.fft(glottal_flow, n=512)))
        DGFFT=10*np.log10(np.abs(np.fft.fft(Dglottal_flow, n=512)))
        F=np.fft.fftfreq(512, 1/fs)
        Fshift=np.fft.fftshift(F)

    else:

        Fshift=[]
        GFFT=[]
        DGFFT=[]

    return go.Figure(data=[go.Scatter(x=Fshift[256:], y=GFFT[0:255], mode = 'lines', name="Glottal flow"),
                           go.Scatter(x=Fshift[256:], y=DGFFT[0:255], mode = 'lines', name="Glottal flow derivative"),],
                     layout=go.Layout(title='Glottal flow spectrum', legend=dict(x=0.6, y=1),
                     xaxis= dict(title= 'Frequency (Hz)', ticklen= 5, zeroline= False, gridwidth= 2),
                     yaxis= dict(title= 'Amplitude (dB)', ticklen= 5, zeroline= False, gridwidth= 2),
                     margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                     ))

@app.callback(
    dash.dependencies.Output('plot_pert_feat', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def update_plot_pert_feat(fname):
    if fname==None:
        return ""
    if os.path.exists(fname):
        fs, signal=read(fname)
        F0, DF0, DDF0, F0semi, Jitter, Shimmer, apq, ppq, logE, degreeU=phonationVowels(fname, False, pitch_method="rapt", voice_bias=-0.3, size_step=0.01)
        tad=len(signal)/(fs*len(Jitter))
        t=np.arange(len(Jitter))*tad
        Jitter=Jitter[12:]
        Shimmer=Shimmer[12:]
        ppq=ppq[6:]
        t=t[12:]

    else:
        Jitter=[]
        Shimmer=[]
        apq=[]
        ppq=[]
        t=[]

    return go.Figure(data=[go.Scatter(x=t, y=Jitter, mode = 'lines', name="Jitter"), go.Scatter(x=t, y=Shimmer, mode = 'lines', name="Shimmer"),
                           go.Scatter(x=t, y=apq, mode = 'lines', name="APQ"), go.Scatter(x=t, y=ppq, mode = 'lines', name="PPQ")],
                     layout=go.Layout(showlegend=True, legend=dict(x=0.6, y=1), title='Perturbation features',xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                     yaxis= dict(title= 'Perturbation (%)', ticklen= 5, zeroline= False, gridwidth= 2), margin={'l': 60, 'b': 40, 't': 40, 'r': 40},
                     ))


@app.callback(
    dash.dependencies.Output('plot_glottal', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def update_plot_glottal(fname):
    if fname==None:
        return ""
    if os.path.exists(fname):
        size_frame=0.2
        size_step=0.2
        fs, data_audio=read(fname)
        data_audio=data_audio-np.mean(data_audio)
        data_audio=data_audio/max(abs(data_audio))
        size_frameS=size_frame*float(fs)
        size_stepS=size_step*float(fs)
        overlap=size_stepS/size_frameS
        nF=int((len(data_audio)/size_frameS/overlap))-1
        data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
        f0=pysptk.sptk.rapt(data_audiof, fs, int(0.01*fs), min=20, max=500, voice_bias=-0.2, otype='f0')
        sizef0=int(size_frame/0.01)
        stepf0=int(size_step/0.01)
        startf0=0
        stopf0=sizef0
        glottal_flow=[]
        Dglottal_flow=[]
        GCI=[]
        amGCI=[]

        GCI=SE_VQ_varF0(data_audio,fs, f0=f0)
        Dglottal_flow=IAIF(data_audio,fs,GCI)
        Dglottal_flow=Dglottal_flow-np.mean(Dglottal_flow)
        Dglottal_flow=Dglottal_flow/max(abs(Dglottal_flow))


        glottal_flow=np.zeros(len(Dglottal_flow))
        for n in range(len(GCI)-1):
            start=int(GCI[n])
            stop=int(GCI[n+1])
            frame_int=Dglottal_flow[start:stop]
            tin=np.arange(len(frame_int))/fs
            glottal_flow_i=cumtrapz(frame_int, tin)
            glottal_flow[start+1:stop]=glottal_flow_i

        #glottal_flow_i=cumtrapz(Dglottal_flow)
        glottal_flow=glottal_flow-np.mean(glottal_flow)
        glottal_flow=glottal_flow/max(abs(glottal_flow))

        amGCI=[glottal_flow[int(k-2)] for k in GCI]
        GCI=GCI/fs
        amGCI=np.hstack(amGCI)
        t=np.arange(len(data_audio))/fs
    else:
        t=[]
        tg=[]
        tgd=[]
        data_audio=[]
        GCI=[]
        amGCI=[]
        glottal_flow=[]
        Dglottal_flow=[]

    marker = dict(
        size = 10,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2,
        )
    )

    return go.Figure(data=[go.Scatter(x=t[0:-1], y=glottal_flow, mode = 'lines', name="Glottal flow"),
                           go.Scatter(x=t, y=Dglottal_flow, mode = 'lines', name="Glottal flow derivative"),
                           go.Scatter(x=t, y=data_audio, mode = 'lines', name="Speech signal"),
                           go.Scatter(x=GCI, y=amGCI, name="GCI", mode = 'markers', marker=marker)],
                     layout=go.Layout(title='Glottal flow', legend=dict(x=0.6, y=1), xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2), margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                     ))

@app.callback(
    dash.dependencies.Output('feat_pert', 'children'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def table_feat_pert(fname):
    if fname==None:
        feat_names=["Jitter (%)", "Shimmer (%)", "APQ (%)", "PPQ (%)", "F0 variability (Hz)"]
        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':["","","","",""],
                                    'Value Standard deviation':["","","","",""]
                                    })
        Table=generate_table(dataframe_feat)
        return Table
    if os.path.exists(fname):
        fs, signal=read(fname)
        F0, DF0, DDF0, F0semi, Jitter, Shimmer, apq, ppq, logE, degreeU=phonationVowels(fname, False, pitch_method="rapt", voice_bias=-0.3, size_step=0.01)
        tad=len(signal)/(fs*len(F0))
        t=np.arange(len(F0))*tad
        Jitterstr=str(np.round(np.mean(Jitter),2))
        Shimmerstr=str(np.round(np.mean(Shimmer),2))
        APQstr=str(np.round(np.mean(apq),2))
        PPQstr=str(np.round(np.mean(ppq),2))
        F0nz=F0[np.where(F0>0)[0]]
        F0varstr=str(np.round(np.std(F0nz),2))

        Jitterstr2=str(np.round(np.std(Jitter),2))
        Shimmerstr2=str(np.round(np.std(Shimmer),2))
        APQstr2=str(np.round(np.std(apq),2))
        PPQstr2=str(np.round(np.std(ppq),2))


        feat_names=["Jitter (%)", "Shimmer (%)", "APQ (%)", "PPQ (%)", "F0 variability (Hz)"]

        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':[Jitterstr, Shimmerstr, APQstr, PPQstr, F0varstr],
                                    'Value Standard deviation':[Jitterstr2, Shimmerstr2, APQstr2, PPQstr2, "-"]
                                    })
        Table=generate_table(dataframe_feat)
    else:
        feat_names=["Jitter (%)", "Shimmer (%)", "APQ (%)", "PPQ (%)", "F0 variability (Hz)"]
        dataframe_feat=pd.DataFrame({'Feature': feat_names,
                                    'Value Average':["","","","",""],
                                    'Value Standard deviation':["","","","",""]
                                    })
        Table=generate_table(dataframe_feat)
    return Table


@app.callback(
    dash.dependencies.Output('plot_F0', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def update_plot_f0(fname):
    if fname==None:
        return ""
    if os.path.exists(fname):
        fs, signal=read(fname)
        F0, DF0, DDF0, F0semi, Jitter, Shimmer, apq, ppq, logE, degreeU=phonationVowels(fname, False, pitch_method="rapt", voice_bias=-0.3, size_step=0.01)
        tad=len(signal)/(fs*len(F0))
        t=np.arange(len(F0))*tad
    else:
        F0=[]
        t=[]
    return go.Figure(data=[go.Scatter(x=t, y=F0, mode = 'lines')],
                     layout=go.Layout(title='Fundamental Frequency',xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2),
                     yaxis= dict(title= 'Frequency (Hz)', ticklen= 5, zeroline= False, gridwidth= 2), margin={'l': 60, 'b': 40, 't': 40, 'r': 40},
                     ))


@app.callback(
    dash.dependencies.Output('file_audio_disp', 'children'),
    [dash.dependencies.Input('bt_load', 'n_clicks')]
)
def load_audio(n_clicks):
    if n_clicks==None:
        return ""
    path = easygui.fileopenbox(msg="select an audio file (*.wav)", filetypes="/*.wav")
    return path

@app.callback(
    dash.dependencies.Output('play_text', 'children'),
    [dash.dependencies.Input('bt_play', 'n_clicks'),
    dash.dependencies.Input('file_audio_disp', 'children')]
)
def play_audio(n_clicks, fname):
    print(n_clicks)
    if n_clicks==None or fname==None:
        return ""
    if os.path.exists(fname):
        fs,x=read(fname)
        sounddevice.play(x,fs)
        return ""
    else:
        return ""


@app.callback(
    dash.dependencies.Output('plot_spec', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def compute_specgram(fname):
    if fname==None:
        return ""
    if os.path.exists(fname):
        fs, signal=read(fname)
        signal=signal-np.mean(signal)
        signal=signal/max(abs(signal))
        y, x, z=sp.spectrogram(signal, fs, "hamming", nfft=2048, mode="magnitude", scaling="spectrum")
        z=20*np.log10(z)+20
        y=y/1000
    else:
        x=[]
        y=[]
        z=[]
    return go.Figure(data=[go.Heatmap(x=x, y=y, z=z, colorscale='Viridis')],
                     layout=go.Layout(title='Spectrogram',xaxis= dict(title= 'Time (s)', ticklen= 5), yaxis= dict(title= 'Frequency (kHz)', ticklen= 5), margin={'l': 60, 'b': 40, 't': 40, 'r': 40},
            ))

@app.callback(
    dash.dependencies.Output('plot_speech', 'figure'),
    [dash.dependencies.Input('file_audio_disp', 'children')])
def update_plot(fname):
    global signal, fs
    if fname==None:
        return ""
    print(fname)
    if os.path.exists(fname):
        fs, signal=read(fname)
        signal=signal-np.mean(signal)
        signal=signal/max(abs(signal))
        xd=np.arange(len(signal))/fs
    else:
        signal=[]
        xd=[]
    return go.Figure(data=[go.Scatter(x=xd, y=signal, mode = 'lines')],
                     layout=go.Layout(title='Speech signal',xaxis= dict(title= 'Time (s)', ticklen= 5, zeroline= False, gridwidth= 2), margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                     ))


if __name__ == '__main__':
    app.run_server(debug=True)
