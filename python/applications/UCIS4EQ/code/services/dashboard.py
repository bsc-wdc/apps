import os
from urllib.request import urlopen
import json
from bson.objectid import ObjectId
import dash
import base64
import uuid
import subprocess
import datetime
from dash import dcc
from dash import html
from dash import dash_table
import dash_cytoscape as dcyto
import dash_leaflet as dl
import requests
import tarfile
import reverse_geocoder as rg
import glob
import time
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from obspy.imaging.beachball import beachball
import ucis4eq.dal as dal
from ucis4eq.dal import staticDataAccess

from ucis4eq.scc.event import EventCountry

import pandas as pd

GEOTIFF_ID = "geotiff-id"
GEOTIFF_MARKER_ID = "geotiff-marker-id"
COORDINATE_CLICK_ID = "coordinate-click-id"
HPC_RUN_PYCOMPSS=os.environ.get("HPC_RUN_PYCOMPSS", "False").lower()
progressValues = None
trialProgressValues = None
if HPC_RUN_PYCOMPSS == 'true':
    print("****** HPC_RUN_PYCOMPSS *******")
    progressValues = {
       'EventSetup': { 'value': 1, 'init':0 },
       'EventRegion': { 'value': 1, 'init':1 },
       'ComputeResources': { 'value': 1 , 'init': 2},
       'SlipGenGPSetup': { 'value': 1, 'init': 3},
       'CMTInputs': { 'value': 1, 'init': 4},
       'CMTCalculation': { 'value': 1, 'init': 5},
       'InputParametersBuilder': { 'value': 1, 'init': 6},
       'SimulationRun': { 'value': 80, 'init': 7},
       'SimulationPostSwarm': { 'value': 13, 'init': 87}
    }

    trialProgressValues = {
       'InputParametersBuilder': { 'value': 5, 'init': 0},
       'SimulationRun': { 'value': 95, 'init': 5}
    }

else :
    print("****** NO HPC_RUN_PYCOMPSS *******")
    progressValues = {
       'EventSetup': { 'value': 1, 'init':0 },
       'EventRegion': { 'value': 1, 'init':1 },
       'ComputeResources': { 'value': 1 , 'init': 2},
       'CMTInputs': { 'value': 1, 'init': 3},
       'CMTCalculation': { 'value': 1, 'init': 4},
       'SourceType': { 'value': 1, 'init': 5},
       'SlipGenGPSetup': { 'value': 1, 'init': 6},
       'SlipGenGP': { 'value': 3, 'init': 7},
       'InputParametersBuilder': { 'value': 1, 'init': 10},
       'SalvusPrepare': { 'value': 7, 'init': 11},
       'SalvusRun': { 'value':  65, 'init': 18},
       'SalvusPost': { 'value': 5, 'init': 83},
       'SalvusPostSwarm': { 'value': 5, 'init': 88},
       'SalvusPlots': { 'value': 7, 'init': 93},
    }
    
    trialProgressValues = {
       'SlipGenGP': { 'value': 5, 'init': 0},
       'InputParametersBuilder': { 'value': 5, 'init': 5},
       'SalvusPrepare': { 'value': 5, 'init': 10},
       'SalvusRun': { 'value':  80, 'init': 15},
       'SalvusPost': { 'value': 5, 'init': 95},
    }

doneSelectedRow = None
regionID = None

figureOptions = {
  "PGA": {
    "Components": {
      "pattern": "[ENZ]",
      "units": {
        "cm/s²": "cm_s_s",
        "%g": "percent"
      }
    },
    "Maximum": {
      "pattern": "max",
      "units": {
        "cm/s²": "cm_s_s",
        "%g": "percent"
      }
    }
  },
  "PGV": {
    "Components": {
      "pattern": "[ENZ]",
      "units": {
        "cm/s": "cm_s"
      }
    },
    "Maximum": {
      "pattern": "max",
      "units": {
        "cm/s": "cm_s"
      }
    }
  },
  "SA": {
    "Components": {
      "pattern": "[ENZ]",
      "units": {
        "cm/s": "cm_s"
      }
    },
    "Maximum": {
      "pattern": "max",
      "units": {
        "cm/s": "cm_s"
      }
    }
  },
  "Arias": {
    "Components": {
      "pattern": "[ENZ]",
      "units": {
        "cm/s": "cm_s"
      }
    },
    "Maximum": {
      "pattern": "max",
      "units": {
        "cm/s": "cm_s"
      }
    }
  },
  "duration_AI": {
    "Components": {
      "pattern": "[ENZ]",
      "units": {
        "s": "_s"
      }
    },
    "Maximum": {
      "pattern": "max",
      "units": {
        "s": "_s"
      }
    }
  },  
  "CAV": {
    "Components": {
      "pattern": "[ENZ]",
      "units": {
        "cm/s": "cm_s"
      }
    },
    "Maximum": {
      "pattern": "max",
      "units": {
        "cm/s": "cm_s"
      }
    }
  }
}

external_stylesheets = [
    {
        "href": "/opt/dashboard/style.css"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "/opt/dashboard/style.css"])

cheeseLogo = base64.b64encode(open("/root/services/assets/Cheese_Logo.png", 'rb').read()).decode('ascii')
eflowsLogo = base64.b64encode(open("/root/services/assets/eFlows_Logo.png", 'rb').read()).decode('ascii')


app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="", className="header-emoji"),
                html.H1(
                    children="UCIS4EQ Portal", className="header-title"
                ),
                html.P(
                    children="Monitor dashboard provides real-time information"
                             " about urgent computing EQ simulations",
                    className="header-description",
                )
            ],
            className="header",
        ),
        html.Div(
            dcc.Tabs(
                id="tabs-with-classes",
                parent_className='custom-tabs',
                className='custom-tabs-container',
                children=[
                    dcc.Tab(
                        label='Alerts',
                        value='tab-1',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='User Event',
                        value='tab-2',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='System Monitor',
                        value='tab-3',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='Results',
                        value='tab-4', className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),                  
                ]),
        ),

        html.Div(id='tabs-content-classes',
                 children=[
                ],
                className='wrapper'),
                
        html.Div(
            [
                dbc.Button("Acknowledgement", id="button-logos", className='modal-button', n_clicks=0),
                dbc.Modal(
                    [
                        #dbc.ModalHeader(),
                        dbc.ModalBody([
                            dbc.Card(dbc.CardImg(src='data:image/png;base64,{}'.format(cheeseLogo)), className="card-logo-cheese"),
                            dbc.Card(dbc.CardImg(src='data:image/png;base64,{}'.format(eflowsLogo)), className="card-logo-eflows")
                            ],
                        ),
                    ],
                    id="modal-logos",
                    size="sm",
                    is_open=False,
                ),
            ],
            style={"position": "absolute", "top": 10, "right": 10},
        ),
        #html.Div(id="logos",
        #         children=[
        #                   dbc.Card(dbc.CardImg(src='data:image/png;base64,{}'.format(cheeseLogo)), className="card-logo-cheese"),
        #                   dbc.Card(dbc.CardImg(src='data:image/png;base64,{}'.format(eflowsLogo)), className="card-logo-eflows")
        #                  ],
        #         style={"position": "absolute", "top": 250, "left": 15, "width": 340},
        #        ),
    ]
)

@app.callback(
    Output("modal-logos", "is_open"),
    [Input("button-logos", "n_clicks")],
    [State("modal-logos", "is_open")],
)
def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(Output('tabs-content-classes', 'children'),
              Input('tabs-with-classes', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        map_layers = mapLayers()

        content = html.Div(id='tab-content-1')

        alert = dcc.Interval(id="progress-interval-alerts", n_intervals=0, interval=10000)

        return html.Div(children = [alert, content, map_layers])

    elif tab == 'tab-2':

        submit = triggerEvent()
        new = newEvent()

        #beachball = html.Div(id='tab-beachball-2')
        map_layers = mapLayers()

        content = html.Div(id='tab-content-2')


        evetAndBB = html.Div(children = [dbc.Row([
                                            dbc.Col(new),
                                            dbc.Col(content)],
                             style = {"width": "100%"} )], className='card')

        return html.Div(children = [submit, evetAndBB, map_layers])

    elif tab == 'tab-3':
        content = html.Div(id='tab-table-3')
        eventsProgress = html.Div(id='tab-progress-3')
        eventsDetails = html.Div(id='tab-details-3')        
        eventsTrials = html.Div(id='tab-trials-3')
        eventsSnap = html.Div(id='tab-snaps-3')

        eventsInfo = html.Div(children = [dbc.Row([
                                            dbc.Col([eventsProgress, eventsDetails, eventsTrials], width=8),
                                            dbc.Col(eventsSnap, width=4)])
                     ])

        return html.Div(children = [
                            content,
                            getLoading('tab-table-3'),
                            eventsInfo
                            ])

    elif tab == 'tab-4':
        content = html.Div(id='tab-table-4')
        plots = html.Div(id='tab-plots-4')

        modal =  html.Div([dbc.Modal([
                        dbc.ModalHeader(),
                        dbc.ModalBody(id="modal-plots-body"),
                        #dbc.ModalFooter(dbc.Button("Close", id="close-modal-plots")),
                        ],
                        id="modal-plots",
                        size="xl",
                        is_open=False,
                        keyboard=False,
                        backdrop="static",                        
                    )])

        return html.Div(children = [
                            content,
                            getLoading('tab-table-4'),
                            modal,
                            getLoading('tab-plots-4'),
                            ])

def getMarkers(data):
    markers = []

    for k in data.keys():
        if 'Color' in data[k].keys():
            color = data[k]['Color']
        else:
            color = "blue"

        markers.append(
            dl.Marker(
                title=data[k]['Origin'],
                icon={"iconUrl": "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-" + color + ".png"},
                position=(data[k]['Latitude'], data[k]['Longitude']),
                children=[
                    dl.Tooltip(data[k]['Origin']),
                    dl.Popup(data[k]['Magnitude']),
                ],
            )
        )

    cluster = dl.MarkerClusterGroup(id="markers", children=markers)
    return cluster

def getLoading(id):
    lid = "loading-" + id
    return  dcc.Loading(
                id=id,
                type="default",
                children=[html.Div(id=lid)],
                )

@app.callback(Output('tab-content-1', 'children'),
              Input("progress-interval-alerts", "n_intervals"),
              Input('layer-selection', 'value'))
def queryEvents(n, layer):

    #data_url = 'https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv'
    #df = pd.read_csv(data_url)

    # Make the query
    fieldsEQ = {}
    i = 0
    col = dal.database["Requests"]
    cursor= col.find({})
    for event in cursor:
        for a in event['alerts']:

            elapsed = str(round(a['elapsedtime']/3600, 2))
            #if a['elapsedtime'] < 86400:
            #    elapsed = str(round(a['elapsedtime']/3600, 2)) + " hours"
            #else:
            #    elapsed = str(round(a['elapsedtime']/86400, 2)) + " days"

            fieldsEQ[str(i)] = {
                'Origin': a['description'],
                'Source': a['agency'],
                'Magnitude': a['magnitude'],
                'Latitude': a['latitude'],
                'Longitude': a['longitude'],
                'Depth (m)': a['depth'],
                'Time (UTC)': a['time'],
                'Elapsed Time (hours)': elapsed
            }
            i = i + 1;

    columns = ['Origin', 'Source', 'Magnitude', 'Latitude', 'Longitude', 'Depth (m)', 'Time (UTC)', 'Elapsed Time (hours)']

    df = pd.DataFrame.from_dict(fieldsEQ, orient='index', columns=columns)

    table = html.Div([
                html.Div(
                    className = "card",
                    children = [
                        # Draw table
                        dash_table.DataTable(
                            id='table',
                            columns=[{"name": i, "id": i}
                                     for i in columns],
                            data=df.to_dict('records'),
                            page_size= 5,
                            sort_action="native",
                            sort_mode='multi',
                            sort_by=[{"column_id": "Elapsed Time (hours)", "direction": "asc"}],
                            style_data={
                                'color': 'black',
                                'backgroundColor': 'white',
                                'textAlign': 'center',
                                'border': 'none',
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(247, 247, 240)',
                                }
                            ],
                            style_header={
                                'backgroundColor': 'rgb(227, 98, 9)',
                                'color': 'black',
                                'fontWeight': 'bold',
                                'textAlign': 'center'
                            }
                        ),
                    ]
                ),
                html.Div(
                    children = [
                    # Draw Map
                    dl.Map(style={'width': '100%', 'height': '40em'},
                           center=[25.67492, 28.63586],
                           zoom=2,
                           children=[
                               #dl.TileLayer(url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_nolabels/{z}/{x}/{y}.png"),
                               dl.TileLayer(url="http://www.google.es/maps/vt?lyrs="+layer+"@189&gl=cn&x={x}&y={y}&z={z}"),
                               dl.GeoTIFFOverlay(id=GEOTIFF_ID, interactive=True),

                               getMarkers(fieldsEQ),

                               html.Div(id=GEOTIFF_MARKER_ID)

                           ], className="card")
                    ]
                ),
    ])

    return table

@app.callback(Output('tab-beachball-2', 'children'),
              [Input('cmt-strike', 'value'),
               Input('cmt-dip', 'value'),
               Input('cmt-rake', 'value')])
def generateBeachBall(stk, dip, rake):

  beachball([stk, dip, rake], width=400,
            linewidth=1, facecolor='blue',outfile = "beachball.png")


  bbFigure = base64.b64encode(open("beachball.png", 'rb').read()).decode('ascii')

  bb = dbc.Card(
    [dbc.CardImg(src='data:image/png;base64,{}'.format(bbFigure), top=True)],
    className="card-bb")

  return bb

@app.callback(Output('tab-content-2', 'children'),
              [Input('layer-selection', 'value'),
               Input('latitude', 'value'),
               Input('longitude', 'value')])
def showEvent(layer, latitude, longitude):

    fieldsEQ = {'event location': {
                    'Origin': 'EQ Epicenter',
                    'Latitude': float(latitude),
                    'Longitude': float(longitude),
                    'Magnitude': 'EarthQuake Event'
                    }
               }

    regionPipeline = [
       {
         "$project" : {
            "_id" : {
                "$toString": "$_id"
            },
            "id": 1,
            "region": 1,
            "mlat" : "$min_latitude",
            "Mlat" : "$max_latitude",
            "mlon" : "$min_longitude",
            "Mlon" : "$max_longitude"
         }
       },
       {
         "$match" : {
           "$and" : [
            {"mlat": {"$lte": float(latitude)}},
            {"Mlat": {"$gte": float(latitude)}},
            {"mlon": {"$lte": float(longitude)}},
            {"Mlon": {"$gte": float(longitude)}}
           ]
         }
       }
    ]

    #for region in self.db['Regions'].aggregate(regionPipeline):

    col = dal.database["Regions"]
    region = list(col.aggregate(regionPipeline))
    sitesMarkers = {}
    if region:
        col = dal.database["Receivers"]
        sitesTypes = col.find_one({"id": region['id']})
        for siteType in sitesTypes:
            if not isinstance(sitesTypes[siteType], dict):
                continue

            if siteType == 'towns':
                color = "yellow"
                type = "Town"
            elif siteType == 'seismic_stations':
                color = "red"
                type = "Seismic station"
            elif siteType == 'accelerometers':
                color = "green"
                type = "Accelerometer"

            siteGroup = sitesTypes[siteType]
            for siteK in siteGroup:
                site = siteGroup[siteK]
                sitesMarkers[siteK] = {
                    'Origin': type,
                    'Latitude': float(site['latitude']),
                    'Longitude': float(site['longitude']),
                    'Magnitude': siteK,
                    'Color': color,
                }

    # Draw Map
    eventMap = dl.Map(style={'width': '100%', 'height': '42em'},
                   center=[float(latitude), float(longitude)],
                   zoom=8,
                   children=[
                       #dl.TileLayer(url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_nolabels/{z}/{x}/{y}.png"),
                       dl.TileLayer(url="http://www.google.es/maps/vt?lyrs="+layer+"@189&gl=cn&x={x}&y={y}&z={z}"),
                       dl.GeoTIFFOverlay(id=GEOTIFF_ID, interactive=True),

                       getMarkers(fieldsEQ),
                       getMarkers(sitesMarkers),

                       html.Div(id=GEOTIFF_MARKER_ID)

               ])

    return eventMap

def newEvent():
    event = html.Div([
                html.Div([
                        html.H5('Event Info', style={'font-weight': 'bold'}),
                        dbc.Row([
                            dbc.Label('Name: ', html_for="location-name", width=2),
                            dbc.Col(
                                dbc.Input(id='location-name', value='Samos EQ Example', type='text'),
                                width=10,
                                ),
                            ],
                            className="mb-2"),

                        dbc.Row([
                            dbc.Label('Magnitude: ', html_for="magnitude", width=2),
                            dbc.Col(
                                dcc.Slider(id='magnitude', min=2, max=10, value=7, marks=None,
                                step=0.1, tooltip={"placement": "bottom", "always_visible": True},
                                className="slider"),
                                width=10
                                ),
                            ],
                        className="mb-1"),
                    ]),

                html.Div([
                        dbc.Row([
                            dbc.Label('Latitude: ', html_for="latitude", width=2),
                            dbc.Col(
                                dbc.Input(id='latitude', value='37.918', type='number'),
                                width=2,
                                ),
                            dbc.Label('Longitude: ', html_for="longitude", width=2),
                            dbc.Col(
                                dbc.Input(id='longitude', value='26.79', type='number'),
                                width=2,
                                ),
                            dbc.Label('Depth: ', html_for="depth", width=2),
                            dbc.Col(
                                dbc.Input(id='depth', value='21000', type='number'),
                                width=2,
                                ),                                
                            ],
                            className="mb-1"),


                    ]),
                html.Div([
                        html.H5('Central Moment Tensor', style={'font-weight': 'bold', 'padding-top': '10px'}),
                        dbc.Row([
                            dbc.Col([
                                dbc.Row([
                                    dbc.Label('Strike: ', html_for="cmt-strike", width=3),
                                    dbc.Col(
                                        dcc.Slider(id='cmt-strike', min=0, max=360, value=270, marks=None,
                                               step=0.5, tooltip={"placement": "bottom", "always_visible": True},
                                               className="slider"),
                                        width=9
                                        )
                                    ],
                                className="mb-1"),

                                dbc.Row([
                                    dbc.Label('Dip: ', html_for="cmt-dip", width=3),
                                    dbc.Col(
                                        dcc.Slider(id='cmt-dip', min=0, max=90, value=37, marks=None,
                                               step=0.5, tooltip={"placement": "bottom", "always_visible": True},
                                               className="slider"),
                                        width=9
                                        )
                                    ],
                                className="mb-1"),

                                dbc.Row([
                                    dbc.Label('Rake: ', html_for="cmt-rake", width=3),
                                    dbc.Col(
                                        dcc.Slider(id='cmt-rake',  min=-180, max=180, value=-86, marks=None,
                                               step=0.5, tooltip={"placement": "bottom", "always_visible": True},
                                               className="slider"),
                                        width=9
                                        )
                                    ],
                                className="mb-1"),
                            ], width=8),
                            dbc.Col([
                                html.Div(id='tab-beachball-2', className='card-bb')
                            ], width=4),
                        ], style = {"width": "100%"}),
                    ]),
                html.Div([
                        html.H5('Graves-Pitarka (Rupture Generator)', style={'font-weight': 'bold'}),
                        dbc.Row([
                            dbc.Col([
                                dbc.Row([
                                    dbc.Label('Seed: ', html_for="cmt-strike", width=2),
                                    dbc.Col(
                                        dcc.Slider(id='gp-seed', min=1, max=9999999, value=2109996, marks=None,
                                               step=1, tooltip={"placement": "bottom", "always_visible": True},
                                               className="slider"),
                                        width=10
                                        ),
                                    ],
                                className="mb-3"),
                            ], style = {"width": "70%"}),
                        ], style = {"width": "100%"}),
                    ]),
            ])

    return event

def mapLayers():
    content= html.Div(
                    children=[

                    html.H3('Map layer selection'),
                    dcc.RadioItems(
                        id="layer-selection",
                        options=[
                            {'label': 'Hybrid', 'value' : 's,h'},
                            {'label': 'Satellite', 'value' : 's'},
                            {'label': 'Streets', 'value' : 'm'},
                            {'label': 'Terrain', 'value' : 'p'}
                        ],
                        value="s,h",
                    ),
                    ],
                    className='card'
                )

    return content

def dropDownMenus():

    options = []
    for key in figureOptions.keys():
        options.append({'label': key, 'value': key})

    first = dcc.Dropdown(
            id='first-dropdown',
            options=options,
            value=None,
            placeholder="Select plot type"
        )

    second = dcc.Dropdown(
            id='second-dropdown',
            options=[],
            value=None,
            placeholder="Select plot sub-type"
        )
    third = dcc.Dropdown(
            id='third-dropdown',
            options=[],
            value=None,
            placeholder="Select metric"
        )
    
    content = html.Div(children=[dbc.Row([dbc.Col(first, width=4), 
                                          dbc.Col(second, width=4),
                                          dbc.Col(third, width=4)], 
             style = {"width": "100%"} )], className = "card")
    return content


@app.callback(
    Output("second-dropdown", "options"),
    [Input("first-dropdown", "value")],
)
def updateFirstDropdown(value1):

    if not value1:
        return dash.no_update

    options = []
    for type in figureOptions[value1].keys():
        options.append({"label": type, "value": type})

    return options

@app.callback(
    Output("third-dropdown", "options"),
    [Input("first-dropdown", "value"),
     Input("second-dropdown", "value")],
)
def updateSecondDropdown(value1, value2):

    if not (value1 and value2):
        return dash.no_update

    options = []
    for units in figureOptions[value1][value2]['units'].keys():
        options.append({"label": units, "value": units})

    return options  

@app.callback(
    Output("plotFigures", "children"),
    [Input("first-dropdown", "value"),
     Input("second-dropdown", "value"),
     Input("third-dropdown", "value")],
)
def buildpattern(value1, value2, value3):

    if not (doneSelectedRow and regionID
            and value1 and value2 and value3):
        return dash.no_update
        
    if value2 in figureOptions[value1].keys() and value3 in figureOptions[value1][value2]['units'].keys():

        # Pattern (    # PGV.*(E|N|Z).*cm_s.*)
        pattern = value1 + "*" + figureOptions[value1][value2]['pattern'] +\
         "*" + figureOptions[value1][value2]['units'][value3] + "*"
         
        # Make the query
        fieldsEQ = {}
        col = dal.database["Requests"]
        request = list(col.find({"_id": ObjectId(doneSelectedRow['Run'])}))[0]

        path = "event_" + request['uuid'] + "_" + regionID + "/"
        filenames = "/*/" + pattern
        # Create the directory for the current execution
        workSpace = "/workspace/runs/" + path
        print("files: "+ workSpace + "/" + filenames)
        os.makedirs(workSpace, exist_ok=True)
        
        files = glob.glob(workSpace + "/" + filenames)

        images = []
        for file in files:
            print("Eval: " + str(file))
            figure = base64.b64encode(open(file, 'rb').read()).decode('ascii')

            item = dbc.Col(dbc.CardImg(src='data:image/png;base64,{}'.format(figure), top=True))

            images.append(item)

        if images:
            content = dbc.Row(images)
        else:
            content = html.H3('No results found for that inputs combination')

    else:
        content = html.H3('Choose a combination for plotting results')

    return content

#@app.callback(Output("accordion-contents", "children"), [Input("accordion", "active_item")])
#def showSecondRadios(value):

#    accordion = []
#    for key in figureOptions.keys():
#        item = dbc.AccordionItem([
#                    html.Div(id="code")
#                ],
#                title=key,
#                item_id=key,
#            )
#
#        accordion.append(item)
#
#    content =  html.Div(
#                id="accordion-contents", children = [
#                dbc.Accordion(accordion, id="accordion"),
#              ])
#
#
#    return content

#@app.callback(
#    Output("accordion-contents", "children"),
#    [Input("accordion", "active_item")],
#)
#def change_item(item):
#    return f"Item selected: {item}"

#@app.callback(Output("accordion-contents", "children"), [Input("accordion", "active_item")])
#def showSecondRadios(value):
#    buttons = []
#    for type in figureOptions[value]:
#        #item = dbc.Button(type['name'], id=type['name'], value=type['pattern'], n_clicks=0)
#        buttons.append({"label": type['name'], "value": type['pattern']})
#
#
#
#    firstRadio= dbc.RadioItems(
#                    id="first-radios",
#                    className="btn-group",
#                    inputClassName="btn-check",
#                    labelClassName="btn btn-outline-primary",
#                    labelCheckedClassName="active",
#                    options= buttons,
#                    value=""
#                )
#
#    return firstRadio

def triggerEvent():
    content= html.Div(
                    children=[
                        html.Button('Trigger this event', id='submit-job', n_clicks=0,
                        style={'width': '100%', 'height': '100%',
                              'font-family': 'system-ui', 'font-size': '25px'})
                    ],
                    className='card'
                )
    return content

@app.callback(
    Output('submit-job', 'children'),
    Output('submit-job', 'disabled'),
    [Input('submit-job', 'n_clicks'),
     Input('location-name', 'value'),
     Input('magnitude', 'value'),
     Input('latitude', 'value'),
     Input('longitude', 'value'),
     Input('depth', 'value'),
     Input('cmt-strike', 'value'),
     Input('cmt-dip', 'value'),
     Input('cmt-rake', 'value'),
     Input('gp-seed', 'value')]
)
def submitJob(n_clicks, ename, mw, lat, lon, depth, stk, dip, rake, seed):
    if n_clicks > 0:
        # Preparing job info
        job = {}

        euuid = str(uuid.uuid1())

        agency = 'USER_EVENT'

        job['sources'] = {agency: {'data': '', 'timestamp': '', 'query': ''}}
        job['uuid'] = euuid

        alert = {}
        alert['agency'] = agency
        alert['description'] = ename
        alert['magnitude'] = float(mw)
        alert['latitude'] = float(lat)
        alert['longitude'] = float(lon)
        alert['depth'] = float(depth)
        alert['time'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y/%m/%d, %H:%M:%S')
        #alert['time'] = datetime.datetime.now(datetime.timezone.utc).strftime('%d/%m/%Y, %H:%M:%S')
        alert['elapsedtime'] = 0
        alert['cmt'] = {agency: {
                                      'strike': stk,
                                      'dip': dip,
                                      'rake': rake
                                   }
                         }
        alert['seed'] = seed;
        job['alerts'] = [alert]

        # Create event directory and safe it
        os.makedirs(euuid, exist_ok=True)

        file = euuid + "/job.json"
        with open(euuid + "/job.json", "w") as f:
            json.dump(job, f, indent=4)

        # Start running the triggering system
        subprocess.Popen(["curl", "-X", "POST", "-H", "Content-Type: application/json", "-d", "@"+file, "http://127.0.0.1:5001/PyCOMPSsWM"])
        #cmd ="curl -X POST -H 'Content-Type: application/json' -d @%s http://127.0.0.1:5001/WMEmulator"
        #cmd = cmd.replace("%s", file)
        #os.system(cmd.replace("%s", file))

        return "Event launched", True
    else:
        return dash.no_update, dash.no_update

@app.callback(Output('tab-table-3', 'children'),
              Input('tabs-with-classes', 'value'))
def queryJobs(value):

    eventRequests = [
        {
          "$unwind": "$alerts"
        },
        {
          "$group": {
            "_id": "$_id",
            "state": { "$first": "$state"},
            "description": { "$first": "$description"},
            "uuid": { "$first": "$uuid"},
            "latitude": { "$avg": "$alerts.latitude" },
            "longitude": { "$avg": "$alerts.longitude" },
            "magnitudemin": { "$min": "$alerts.magnitude" },
            "magnitudemax": { "$max": "$alerts.magnitude" },
            "depthmin": { "$min": "$alerts.depth" },
            "depthmax": { "$max": "$alerts.depth" },
            "alerts": { "$sum": 1}
          }
        }
    ]

    # Make the query
    fieldsEQ = {}
    col = dal.database["Requests"]
    cursor= col.aggregate(eventRequests)
    for event in cursor:

            lat = round(event['latitude'], 2)
            lon = round(event['longitude'], 2)

            id = str(ObjectId(event['_id']))
            place = rg.search((lat, lon))[0]
            fieldsEQ[id] = {
                'Status': event['state'],
                'Origin': place['admin1'],
                'Site': place['admin2'],
                'Latitude': lat,
                'Longitude': lon,
                'Min. Mw': event['magnitudemin'],
                'Max. Mw': event['magnitudemax'],
                'Min. Depth': event['depthmin'],
                'Max. Depth': event['depthmax'],
                '# Alerts': event['alerts'],
                #'UUID': event['uuid']
                'Run': id
            }

    columns = ['Status', 'Origin', 'Site', 'Latitude', 'Longitude', 'Min. Mw', 'Max. Mw', 'Min. Depth', 'Max. Depth', '# Alerts', 'Run']


    df = pd.DataFrame.from_dict(fieldsEQ, orient='index', columns=columns)

    table = html.Div(
        className="card",
        children=[
            dcc.Interval(id="progress-interval", n_intervals=0, interval=30000),
            dcc.Interval(id="progress-interval-snaps", n_intervals=0, interval=90000),
            # Draw table
            dash_table.DataTable(
                id='tableRunning',
                columns=[{"name": i, "id": i}
                         for i in columns],
                data=df.to_dict('records'),
                page_size= 5,
                sort_action="native",
                sort_mode='multi',
                row_selectable='single',
                style_data={
                    'color': 'black',
                    'backgroundColor': 'white',
                    'textAlign': 'center',
                    'border': 'none',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(247, 247, 240)',
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(227, 98, 9)',
                    'color': 'black',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                }

        )],
    )

    return table

@app.callback(Output('tab-progress-3', 'children'),
              Input("progress-interval", "n_intervals"),
              Input('tableRunning', "derived_viewport_data"),
              Input('tableRunning', "derived_virtual_selected_rows"))
def executionTree(n, data, idx):

    if len(idx):
        progressBar = html.Div(
                        className = "card",
                        children = [dbc.Row([
                                      dbc.Col(dbc.Progress(id="progress", striped=True, animated="True", style={"height": "50px", "font-size": "40px"}), width=11),
                                      dbc.Col(dbc.Button("+", id="progress-button", className='progress-button', n_clicks=0, style = {"width": "100%"}), width=1)
                                    ],
                                    style = {"width": "100%"} )]
                     )
    else:
        progressBar = html.Div()

    return progressBar

@app.callback(Output('tab-snaps-3', 'children'),
              Input("progress-interval-snaps", "n_intervals"),
              Input('tableRunning', "derived_viewport_data"),
              Input('tableRunning', "derived_virtual_selected_rows"))
def executionSnapshot(n, data, idx):

    fieldsEQ = {}

    if not idx or len(idx) == 0:
        return html.Div()

    col = dal.database["ServiceRuns"]
    cursor= col.find({"requestId": data[idx[0]]['Run']})
    service = list(cursor)[-1]
    map = True
    if service['serviceName'] == "SalvusRun" or service['serviceName'] == "SimulationRun":

        # Creating the repository instance for data transfer
        # TODO: Select the repository from the DB 'Resources' document
        #dataRepo = dal.repositories.create(service['machine']['repository'], **dal.config)
        dataRepo = staticDataAccess.repositories.create(service['machine']['repository'], **dal.config)

        # Create the directory for the current execution
        workSpace = "/workspace/runs/" + service['inputs']['trial'] + "/"
        os.makedirs(workSpace, exist_ok=True)

        # Create the remote working directory
        filename = "current.png"
        rfile = service['inputs']['trial'] + "/salvus/snapshots/" +  filename
        lfile = workSpace + filename

        # Download results from HPC machine
        #dataRepo.downloadFile(rfile, lfile, trials=0)
        
        try:
            currentSnapshot = base64.b64encode(open(lfile, 'rb').read()).decode('ascii')
            snapshot = dbc.Card(
                                [dbc.CardImg(src='data:image/png;base64,{}'.format(currentSnapshot)),
                                 html.H4(
                                   "A preview of the magnitude of the (unfiltered) velocity field.",
                                   className="card-text",
                                 )
                                ]
                                )
            map = False
        except:
            pass
    

    if map:
        latitude = float(data[idx[0]]['Latitude'])
        longitude = float(data[idx[0]]['Longitude'])

        fieldsEQ = {'event location': {
                        'Origin': 'EQ Epicenter',
                        'Latitude': latitude,
                        'Longitude': longitude,
                        'Magnitude': 'EarthQuake Event'
                        }
                   }

        snapshot = dl.Map(style={'width': '100%', 'height': '480px'},
                       center=[latitude, longitude],
                       zoom=6,
                       children=[
                           #dl.TileLayer(url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_nolabels/{z}/{x}/{y}.png"),
                           dl.TileLayer(url="http://www.google.es/maps/vt?lyrs=s,h@189&gl=cn&x={x}&y={y}&z={z}"),
                           dl.GeoTIFFOverlay(id=GEOTIFF_ID, interactive=True),

                           getMarkers(fieldsEQ),
                           html.Div(id=GEOTIFF_MARKER_ID)

                   ], className = "card")

    return html.Div(children=[snapshot])


@app.callback(Output('tab-trials-3', 'children'),
              Input("progress-interval", "n_intervals"),
              Input('tableRunning', "derived_viewport_data"),
              Input('tableRunning', "derived_virtual_selected_rows"))
def executionDetails(n, data, idx):
    
    if not len(idx):
        return dash.no_update  
        
    request = [
        {
            "$match": {
                "requestId": data[idx[0]]['Run'],
                 "inputs.trial": {"$ne" : None} 
                }
        },
        {
            "$group": { 
                "_id": { "trial": "$inputs.trial"},
                } 
        },
        {
        "$sort": {
            "_id.trial": 1  # Sort by inputs.trial in ascending order
        }
    }
    ] 
        
    col = dal.database["ServiceRuns"]
    cursor= col.aggregate(request)    
    
    pbars = []
    
    for trial in cursor:    
        name = trial["_id"]["trial"]
        cursor2 = col.find({"inputs.trial": name})
        service=list(cursor2)[-1]
        if name == "event_cabc0e50-a0ec-11ea-9ec0-test/trial_WESTERN_TURKEY_3D.GCMT.slip2":
            show = True
        else:
            show = False
        
        progress, label, color = calculateProgress(service, trialProgressValues, show)
            
        elem = dbc.Row([
                      dbc.Col(html.H6(name.split("/")[-1]), width=7),
                      dbc.Col(dbc.Progress(id=f"progressBar_{name}", 
                                           striped=True, 
                                           animated="True", 
                                           value=progress,
                                           label=label,
                                           color=color,
                                           style={"height": "25px", 
                                                  "font-size": "20px",
                                                  "text-align": "center"}
                                          ), 
                                           width=5),
                        dbc.Tooltip(
                                    f"Stage: {service['serviceName']}",
                                    target=f"progressBar_{name}",
                                    placement="left"
                                )                   
                      
                    ],
                    style = {"width": "100%", "margin": "5px"} )            
            
        pbars.append(elem)
            
        
    content = html.Div(
                    className="card",
                    children = pbars
                )
    return content
    
@app.callback(Output('tab-details-3', 'children'),
              Input("progress-interval", "n_intervals"),
              Input('tableRunning', "derived_viewport_data"),
              Input('tableRunning', "derived_virtual_selected_rows"))
def executionDetails(n, data, idx):

    fieldsEQ = {}

    if not idx or len(idx) == 0:
        return html.Div()

    col = dal.database["ServiceRuns"]
    cursor= col.find({"requestId": data[idx[0]]['Run']})

    i = 0
    for event in cursor:
        if event['machine'] == 'N/A':
            machine = "Local"
        else:
            machine = event['machine']['id'] + ' (' + event['machine']['organization'] + ')'
            
        fieldsEQ[str(i)] = {
            'Service': event['serviceName'],
            'Status':  event['status'],
            'InitTime': event['initTime'],
            'EndTime': event['endTime'] if 'endTime' in event.keys() else '',
            'Machine': machine
        }
        i = i + 1;

    columns = ['Service', 'Status', 'InitTime', 'EndTime', 'Machine']


    df = pd.DataFrame.from_dict(fieldsEQ, orient='index', columns=columns)

    dataTable = dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i}
                             for i in columns],
                    data=df.to_dict('records'),
                    sort_action="native",
                    sort_mode='multi',
                    style_data={
                        'color': 'black',
                        'backgroundColor': 'white',
                        'textAlign': 'center',
                        'border': 'none',
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(247, 247, 240)',
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(227, 98, 9)',
                        'color': 'black',
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_table={'overflowY': 'auto'}
            )

    content = html.Div(
                    id="generalProgress",
                    children = [
                        dbc.Modal([
                                dbc.ModalHeader(),
                                dbc.ModalBody(),
                                   dataTable
                                ],
                                size="xl",
                                is_open=False,
                                keyboard=False,
                                backdrop="static",  
                                scrollable=True,
                                id="modal-details"
                            )                   
                    ]
                )
    return content
    
@app.callback(
    Output("modal-details", "is_open"),
    [Input("progress-button", "n_clicks")],
    [State("modal-details", "is_open")],
)
def toggle_modal_progress(n1, is_open):
    if n1:
        return not is_open
    return is_open    

@app.callback(
    [Output("progress", "value"), Output("progress", "label"), Output("progress", "color")],
    [Input("progress-interval", "n_intervals"),
     Input('tableRunning', "derived_viewport_data"),
     Input('tableRunning', "derived_virtual_selected_rows")]
)
def update_progress(n, data, idx):

    if not data or not idx:
        return dash.no_update, dash.no_update, dash.no_update


    col = dal.database["ServiceRuns"]
    cursor= col.find({"requestId": data[idx[0]]['Run']})

    # Calculate progress for main progress bar        
    progress,label, color = calculateProgress(list(cursor)[-1], progressValues)

    if data[idx[0]]['Status'] == "REJECTED":
       color = "warning"
    elif data[idx[0]]['Status'] == "SUCCESS":
        color = "success"

    # check progress of some background process, in this example we'll just
    # use n_intervals constrained to be in 0-100
    #progress = min(n % 110, 100)
    # only add text after 5% progress to ensure text isn't squashed too much
    return progress, label, color

def calculateProgress(service, progressValues, show=False):

    color = "#2b6777"
    value = progressValues[service['serviceName']]['value']

    if service['serviceName'] == "SalvusRun":
        r = requests.post("http://127.0.0.1:5003/SalvusPing", json=service['inputs'])
        if r.json()['response'] == 501 or not r.json()['result']:
            #raise Exception(r.json()['result'])
            value = 0
        else:
            result = r.json()['result']
            if len(result['tasks']) == 2:
                task = result['tasks'][1]
                if 'current' in task.keys():
                    value = int(value * (task['current'] / task['total']))
            if show:
                print(service['serviceName'] + "    ---> " + str(value))
    elif service['serviceName'] == "SimulationRun":
        if service['status'] == "RUNNING":
            time.sleep(1)
            r = requests.post("http://127.0.0.1:5004/SimulationPing", json=service['inputs'])
            if r.json()['response'] == 501 or not r.json()['result']:
                #raise Exception(r.json()['result'])
                value = 0
            else:
                result = r.json()['result']
                if len(result['tasks']) == 2:
                    task = result['tasks'][1]
                    if 'current' in task.keys():
                        value = int(value * (task['current'] / task['total']))
                if show:
                    print(service['serviceName'] + "    ---> " + str(value))
        else:
            value = 0
        

    if service['status'] == "SUCCESS":
        progress = progressValues[service['serviceName']]['init'] +\
                   progressValues[service['serviceName']]['value']
    else:
        progress = progressValues[service['serviceName']]['init'] + value
        if service['status'] == "FAILED":
            color = "danger"
        elif service['status'] == "CANCELED":
            color = "warning"
    
    return progress, f"{progress} %" if progress >= 8 else "", color

@app.callback(Output('tab-table-4', 'children'),
              Input('tabs-with-classes', 'value'))
def queryDoneJobs(value):

    eventRequests = [
        {
          "$match" : {
            "state": "SUCCESS"
          }
        },
        {
          "$unwind": "$alerts"
        },
        {
          "$group": {
            "_id": "$_id",
            "description": { "$first": "$description"},
            "uuid": { "$first": "$uuid"},
            "latitude": { "$avg": "$alerts.latitude" },
            "longitude": { "$avg": "$alerts.longitude" },
            "magnitudemin": { "$min": "$alerts.magnitude" },
            "magnitudemax": { "$max": "$alerts.magnitude" },
            "depthmin": { "$min": "$alerts.depth" },
            "depthmax": { "$max": "$alerts.depth" },
            "alerts": { "$sum": 1}
          }
        }
    ]

    # Make the query
    fieldsEQ = {}
    col = dal.database["Requests"]
    cursor= col.aggregate(eventRequests)
    for event in cursor:

            lat = round(event['latitude'], 2)
            lon = round(event['longitude'], 2)

            id = str(ObjectId(event['_id']))
            place = rg.search((lat, lon))[0]

            fieldsEQ[id] = {
                'Origin': place['admin1'],
                'Site': place['admin2'],
                'Latitude': lat,
                'Longitude': lon,
                'Min. Mw': event['magnitudemin'],
                'Max. Mw': event['magnitudemax'],
                'Min. Depth': event['depthmin'],
                'Max. Depth': event['depthmax'],
                '# Alerts': event['alerts'],
                #'UUID': event['uuid']
                'Run': id
            }

    columns = ['Origin', 'Site', 'Latitude', 'Longitude', 'Min. Mw', 'Max. Mw', 'Min. Depth', 'Max. Depth', '# Alerts', 'Run']


    df = pd.DataFrame.from_dict(fieldsEQ, orient='index', columns=columns)

    table = html.Div(
        className="card",
        children=[
            # Draw table
            dash_table.DataTable(
                id='tableDone',
                columns=[{"name": i, "id": i}
                         for i in columns],
                data=df.to_dict('records'),
                page_size= 5,
                sort_action="native",
                sort_mode='multi',
                row_selectable='single',
                style_data={
                    'color': 'black',
                    'backgroundColor': 'white',
                    'textAlign': 'center',
                    'border': 'none',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(247, 247, 240)',
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(227, 98, 9)',
                    'color': 'black',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                }

        )],
    )

    return table

@app.callback(Output('tab-domains-4', 'children'),
              Input('tableDone', "derived_viewport_data"),
              Input('tableDone', "derived_virtual_selected_rows"))
def queryDomains(data, idx):



    columns = ['Region']

    #df = pd.DataFrame.from_dict(fieldsEQ, orient='index', columns=columns)

    content= html.Div(
                    children=[
                    ]
                )
    return content

@app.callback(Output('modal-plots-body', 'children'),
              Input('tableDone', "derived_viewport_data"),
              Input('tableDone', "derived_virtual_selected_rows"))
def plotResults(data, idx):

    fieldsEQ = {}

    if not idx or len(idx) == 0:
        return html.Div()

    global doneSelectedRow
    doneSelectedRow = data[idx[0]]

    regionPipeline = [
       {
         "$project" : {
            "_id" : {
                "$toString": "$_id"
            },
            "id": 1,
            "region": 1,
            "mlat" : "$min_latitude",
            "Mlat" : "$max_latitude",
            "mlon" : "$min_longitude",
            "Mlon" : "$max_longitude",
            "Depth" : "$depth_in_m",
         }
       },
       {
         "$match" : {
           "$and" : [
            {"mlat": {"$lte": float(data[idx[0]]['Latitude'])}},
            {"Mlat": {"$gte": float(data[idx[0]]['Latitude'])}},
            {"mlon": {"$lte": float(data[idx[0]]['Longitude'])}},
            {"Mlon": {"$gte": float(data[idx[0]]['Longitude'])}}
           ]
         }
       }
    ]

    #for region in self.db['Regions'].aggregate(regionPipeline):

    col = dal.database["Regions"]
    region = list(col.aggregate(regionPipeline))[0]

    # Make the query
    fieldsEQ = {}
    col = dal.database["Requests"]
    request = list(col.find({"_id": ObjectId(doneSelectedRow['Run'])}))[0]

    global regionID
    regionID = region['id']
    path = "event_" + request['uuid'] + "_" + region['id'] + "/"
    filename = "event_" + request['uuid'] + ".tar.gz"

    # Creating the repository instance for data transfer
    # TODO: Select the repository from the DB 'Resources' document
    dataRepo = staticDataAccess.repositories.create(dal.repository, **dal.config)

    # Create the directory for the current execution
    workSpace = "/workspace/runs/" + path
    os.makedirs(workSpace, exist_ok=True)

    rpath = "UCIS4EQ/Runs/" + path + filename
    lpath = workSpace + filename

    # Download results from HPC machine
    if not os.path.isfile(lpath):
        dataRepo.downloadFile(rpath, lpath)

    # Extract files from tar ball
    tar = tarfile.open(lpath, "r:gz")
    tar.extractall(path=workSpace)
    tar.close()
    print("FIle Extracted in "+ workSpace)
    
    figures = html.Div(id="plotFigures", className="card",
                       children=[html.H3('Figures will appear here')])          
          
    return html.Div(children = [dropDownMenus(), figures])
    
@app.callback(
    Output("modal-plots", "is_open"),
    [Input('tableDone', "derived_virtual_selected_rows")],
    [State("modal-plots", "is_open")],
)
def toggle_modal_plots(n1, is_open):
    if n1:
        return not is_open
    return is_open
    
    
if __name__ == "__main__":
    app.title = "UCIS4EQ Monitor"
    app.config['suppress_callback_exceptions'] = True
    app.run_server(host="0.0.0.0", debug=True)
