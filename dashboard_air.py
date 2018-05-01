# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import glob
import os
import yaml
import matplotlib
from matplotlib import cm
import base64
from scipy.integrate import cumtrapz

def define_app():
    app = dash.Dash()
    
    # Dash CSS
    app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
    
    # Loading screen CSS
    app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})
    
    # Logo
    encoded_image = base64.b64encode(open('logo.png', 'rb').read())
    
    
    folders = glob.glob('./Data/*')
    folder = 'hale_2017_11_16_10_15_17 - Single Orbit'
    file = glob.glob('./Data/'+folder+'/Intermediates/*.xlsx')[-1]
    df = pd.read_excel(file)
    df = fix_units(df)
    units = load_units()
    labels = label_list(df.columns)
#    times = np.arange(df.time.min()/3600,df.time.max()/3600,1)
    
    # Get colormap
    plasma_cmap = matplotlib.cm.get_cmap('plasma')
    plasma_rgb = []
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
    for i in range(0, 255):
       k = matplotlib.colors.colorConverter.to_rgb(plasma_cmap(norm(i)))
       plasma_rgb.append(k)
    plasma = matplotlib_to_plotly(plasma_cmap, 255)
    
    app.layout = html.Div(children=[
            
        # Title
        html.Div([
        html.H1(children='Optimization Results',className='ten columns'),
        html.A(html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),href='https://apm.byu.edu/prism/index.php',className='two columns'),
        ],className='row'),
        
        # Data Selection Dropdown Row
        html.Div([
            # Data Selection
            html.Div([
            html.Div(children='Select Data',className='twelve columns'),
            dcc.Dropdown(
                    id='select_data',
                    options=[{'label': os.path.basename(i), 'value': i} for i in folders],
                    value=folders[-1],
                    className='five columns'
                    ),
            html.Button('Refresh Folder List',id='refresh_button',className='two columns'),
            html.Div('Description text',id='description',className='three columns')
                ]),
        ],className='row',style={'padding-bottom':'20px','padding-left':'10px'}),
            
        
        # Variable Selection Dropdown Row
        html.Div([
            # Variable Selection 1
            html.Div([
            html.Div('Select Variable'),
            dcc.Dropdown(
                    id='variable',
                    options=[{'label': i, 'value': i} for i in df.columns.sort_values()],
                    value='h'
                )
            ],className='four columns',style={'vertical-align': 'top'}),
            
            # Variable Selection 2
            html.Div([
            html.Div('Select Variable'),
            dcc.Dropdown(
                    id='variable2',
                    options=[{'label': i, 'value': i} for i in df.columns.sort_values()],
                    value='te_kwh'
                )
            ],className='four columns',style={'vertical-align': 'top'}),
            
            # Variable Selection 3
            html.Div([
            html.Div('Select Variable'),
            dcc.Dropdown(
                    id='variable3',
                    options=[{'label': i, 'value': i} for i in df.columns.sort_values()],
                    value='e_batt_kwh'
                )
            ],className='four columns',style={'vertical-align': 'top'}),
            
        ],className='row',style={'padding-bottom':'10px','padding-left':'10px'}),
            
        # Radio Row
        html.Div([
                html.Div([
                dcc.RadioItems(
                        id='dimension',
                        options=[{'label': i, 'value': i} for i in ['2D', '3D']],
                        value='2D'
                    )
                ],className='one column'),
                html.Div([
                dcc.RadioItems(
                        id='showwind',
                        options=[{'label': i, 'value': i} for i in ['Hide Wind', 'Show Wind']],
                        value='Hide Wind'
                    )
                ],className='two columns'),
                html.Div([
                dcc.RadioItems(
                        id='showsun',
                        options=[{'label': i, 'value': i} for i in ['Hide Sun', 'Show Sun']],
                        value='Hide Sun'
                    )
                ],className='two columns'),
#                html.Div([
#                html.Div('Select Variable'),
#                dcc.Dropdown(
#                        id='variable4',
#                        options=[{'label': i, 'value': i} for i in df.columns.sort_values()],
#                        value='e_batt_kwh'
#                    )
#                ],className='four columns',style={'vertical-align': 'top'})
        ],className='row',style={'padding-bottom':'10px','padding-left':'10px'}),
    
        # Graph Row
        html.Div([
            # Graph 1
            html.Div([
            dcc.Graph(id='2D-plot')],className='four columns'),
            
            # Graph 2
            html.Div([
            dcc.Graph(id='TE-plot')],className='four columns'),
    
            # Graph 3
            html.Div([
            dcc.Graph(id='plot3')],className='four columns'),
    
        ],className='row',style={'padding-left':'10px'}),
        
        # Slider Row
        html.Div([
            # Slider
            html.Div([dcc.RangeSlider(
        		id='hours',
               #marks={i: 'Hour {}'.format(int(i)) if int(i) == 0 else str(int(i)) for i in np.arange(df.time.min()/3600,df.time.max()/3600,1)},
               marks={int(i): str(i) for i in np.arange(df.time.min()/3600,max(df.time.max()/3600,2),1)},
        		count=1,
        		min=np.arange(df.time.min()/3600,df.time.max()/3600,1)[0],
        		max=max(np.arange(df.time.min()/3600,df.time.max()/3600,1)[-1],1),
        		step=0.05,
        		value=[0, max(np.arange(df.time.min()/3600,df.time.max()/3600,1)[-1],1)],
#                value=[2,3],
              allowCross=False,
              pushable=0.05
        		)],
                className='twelve columns'
            ),
        ],className='row',style={'padding-left':'15px','padding-right':'15px'}),
        
        # Hidden data div
        html.Div(id='intermediate-value', style={'display': 'none'})
    ])
        
    @app.callback(
    dash.dependencies.Output('2D-plot', 'figure'),
    [dash.dependencies.Input('variable', 'value'),
    dash.dependencies.Input('hours', 'value'),
    dash.dependencies.Input('dimension', 'value'),
    dash.dependencies.Input('showsun', 'value'),
    dash.dependencies.Input('showwind', 'value'),
    dash.dependencies.Input('intermediate-value', 'children')])
    def update_graph(variable,hours,dimension,sun,wind,data):
        
        df = pd.read_json(data)
        df = df.sort_values('time')
        
#        df_filtered = df.iloc[int(hours[0]*3600/10):int(hours[1]*3600/10),:]
        df_filtered = df.loc[(df.time_hr >= hours[0]) & (df.time_hr <= hours[1])]
        
        colors = df_filtered[variable]
        
        # Try to get pretty colorbar title
        if variable in units:
            colortitle = units[variable]
        else:
            colortitle = variable
    	
        if(dimension=='2D'):
            points = go.Scatter(
            						x=df_filtered['y_km'],
            						y=df_filtered['x_km'],
            						mode='lines+markers',
            						marker=dict(
            							size='7',
            							color = colors, #set color equal to a variable
                                colorbar = go.ColorBar(title=colortitle,titleside = 'right'),
            							colorscale='Jet',
            							showscale=True
            						),
                               line=dict(
                                       color='gray'
                                       ),
                            text = colors,
            						opacity=0.7
            					)
            layout = go.Layout(
            					title = 'Flight Path',
            					xaxis={'title': 'East (km)'},
            					yaxis={'title': 'North (km)','scaleanchor':'x','scaleratio':1},
            					hovermode='closest',
                         shapes=[{
                                'type': 'circle',
                                'xref': 'x',
                                'yref': 'y',
                                'x0': -3,
                                'y0': -3,
                                'x1': 3,
                                'y1': 3,
                                'layer':'below',
                                'line': {
                                    'color': 'orange',
                                    'dash': 'dash',
                                    'width': 4
                                }
                            }]
            				)
            # Plot sun and wind
            if(sun=='Show Sun' and wind=='Show Wind' and 'w_n' in df.columns):
                df_sun = df_filtered.iloc[::90,:]
                sun_n = df_sun.sn2*5
                sun_e = df_sun.sn1*5
                figure = ff.create_quiver(sun_e,sun_n,-df_sun.sn1*3,-df_sun.sn2*3, scale=0.2, name='Sun',line={'color':'rgb(238,180,34,0.8)'})
                figure['data'].append(points)
                wind_x,wind_y = np.meshgrid(np.arange(-4, 5, 1),
                                            np.arange(-4, 5, 1))
                wind_u = np.ones(wind_x.shape)*df['w_e'][0]
                wind_v = np.ones(wind_x.shape)*df['w_n'][0]
                figure_wind = ff.create_quiver(wind_x,wind_y,wind_u,wind_v, scale=0.05, name='Wind',line={'color':'rgb(220,220,220,0.5)'})
                figure['data'].append(figure_wind['data'][0])
                figure['layout'] = (layout)
            # Plot sun vectors if enabled
            elif(sun=='Show Sun'):
                df_sun = df_filtered.iloc[::90,:]
                sun_n = df_sun.sn2*5
                sun_e = df_sun.sn1*5
                figure = ff.create_quiver(sun_e,sun_n,-df_sun.sn1*3,-df_sun.sn2*3, scale=0.2, name='Sun',line={'color':'rgb(238,180,34,0.8)'})
                figure['data'].append(points)
                figure['layout'] = (layout)
            elif(wind=='Show Wind' and 'w_n' in df.columns):
                wind_x,wind_y = np.meshgrid(np.arange(-4, 5, 1),
                                            np.arange(-4, 5, 1))
                wind_u = np.ones(wind_x.shape)*df['w_e'][0]
                wind_v = np.ones(wind_x.shape)*df['w_n'][0]
                figure = ff.create_quiver(wind_x,wind_y,wind_u,wind_v, scale=0.05, name='Wind',line={'color':'rgb(220,220,220,0.5)'})
                figure['data'].append(points)
                figure['layout'] = (layout)  
            else:
                figure =  {
            			'data': [
            					go.Scatter(
            						x=df_filtered['y_km'],
            						y=df_filtered['x_km'],
            						mode='lines+markers',
            						marker=dict(
            							size='7',
            							color = colors, #set color equal to a variable
                                colorbar = go.ColorBar(title=colortitle,titleside = 'right'),
            							colorscale='Jet',
            							showscale=True
            						),
                                line=dict(
                                   color='gray'
                                   ),
                            text = colors,
            						opacity=0.7
            					)
            				],
            				'layout': go.Layout(
            					title = 'Flight Path',
            					xaxis={'title': 'East (km)'},
            					yaxis={'title': 'North (km)','scaleanchor':'x','scaleratio':1},
            					hovermode='closest',
                         shapes=[{
                                'type': 'circle',
                                'xref': 'x',
                                'yref': 'y',
                                'x0': -3,
                                'y0': -3,
                                'x1': 3,
                                'y1': 3,
                                'layer':'below',
                                'line': {
                                    'color': 'orange',
                                    'dash': 'dash',
                                    'width': 4
                                }
                            }]
            				)
            		}
    
        else:
            figure =  {
            'data': [
                    go.Scatter3d(
                        x=df_filtered['y_km'],
                        y=df_filtered['x_km'],
                        z=df_filtered['h_km'],
                        mode='lines+markers',
                        marker=dict(
                            size='3',
                            color = colors, #set color equal to a variable
                            colorbar = go.ColorBar(title=colortitle,titleside = 'right'),
                            colorscale='Jet',
                            showscale=True
                        ),
                        opacity=0.7,
                        text = colors
                    )
                ],
                'layout': go.Layout(
                    scene = go.Scene(
                            xaxis=go.XAxis(title='East (km)'),
                            yaxis=go.YAxis(title='North (km)'),
                            zaxis=go.ZAxis(title='Altitude (km)',range=[df_filtered.h_km.min()-1,df_filtered.h_km.max()+1])
                            ),
                    margin=dict(
                    r=10, l=10,
                    b=10, t=10)
                )
        }
        return figure
        
    @app.callback(
    dash.dependencies.Output('TE-plot', 'figure'),
    [dash.dependencies.Input('variable2', 'value'),
#     dash.dependencies.Input('variable4', 'value'),
    dash.dependencies.Input('hours', 'value'),
    dash.dependencies.Input('intermediate-value', 'children')])
    def update_graph2(variable2,hours,data):
        
        df = pd.read_json(data)
        df = df.sort_values('time')
        
#        df_filtered = df.iloc[int(hours[0]*3600/10):int(hours[1]*3600/10),:]
        df_filtered = df.loc[(df.time_hr >= hours[0]) & (df.time_hr <= hours[1])]
        
        values = df_filtered[variable2]
#        values2 = df_filtered[variable4]
        
        # Try to get pretty axis title
        if variable2 in units:
            axistitle = units[variable2]
        else:
            axistitle = variable2
            
#        # Try to get pretty axis title
#        if variable4 in units:
#            axistitle2 = units[variable4]
#        else:
#            axistitle2 = variable4
            
        figure = {
            'data': [
                    go.Scatter(
                        x=df_filtered['time_hr'],
                        y=values,
                        mode='lines',
                        marker=dict(
                            size='3'
                        ),
                        opacity=0.7,
                        name=axistitle
                    ),
#                    go.Scatter(
#                        x=df_filtered['time_hr'],
#                        y=values2,
#                        mode='lines',
#                        marker=dict(
#                            size='3'
#                        ),
#                        opacity=0.7,
#                        yaxis='y2',
#                        name=axistitle2
#                    ),
                ],
                'layout': go.Layout(
                    title = axistitle,
                    xaxis={'title': 'Time (hr)'},
        				yaxis={'title': axistitle},
#                    yaxis2={'title':axistitle2,'overlaying':'y','side':'right'},
                    hovermode='closest'
                )
        }
    
        if(variable2=='te_kwh' or variable2=='te'):
            figure['layout'] = go.Layout(
                    title = axistitle,
                    xaxis={'title': 'Time (hr)'},
    				yaxis={'title': axistitle,'range':[0,values.max()]},
                    hovermode='closest'
                    )
        
        return figure
    
    @app.callback(
    dash.dependencies.Output('plot3', 'figure'),
    [dash.dependencies.Input('variable3', 'value'),
    dash.dependencies.Input('hours', 'value'),
    dash.dependencies.Input('intermediate-value', 'children')])
    def update_graph3(variable3,hours,data):
        
        df = pd.read_json(data)
        df = df.sort_values('time')
        
#        df_filtered = df.iloc[int(hours[0]*3600/10):int(hours[1]*3600/10),:]
        df_filtered = df.loc[(df.time_hr >= hours[0]) & (df.time_hr <= hours[1])]
        
        values = df_filtered[variable3]
        
        # Try to get pretty axis title
        if variable3 in units:
            axistitle = units[variable3]
        else:
            axistitle = variable3
        
        return {
            'data': [
                    go.Scatter(
                        x=df_filtered['time_hr'],
                        y=values,
                        mode='lines',
                        marker=dict(
                            size='3'
                        ),
                        opacity=0.7
                    )
                ],
                'layout': go.Layout(
                    title = axistitle,
                    xaxis={'title': 'Time (hr)'},
    				yaxis={'title': axistitle},
                    hovermode='closest'
                )
        }
            
    @app.callback(dash.dependencies.Output('intermediate-value', 'children'), [dash.dependencies.Input('select_data', 'value')])
    def load_data(folder):
         # Load data
         file = glob.glob(folder+'/Intermediates/*.xlsx')
         if(len(file)==0):
             file = glob.glob(folder+'/ss*.xlsx')
         new_df = pd.read_excel(file[-1])
         new_df = fix_units(new_df)
         return new_df.to_json() # or, more generally, json.dumps(cleaned_df)
     
    @app.callback(dash.dependencies.Output('description', 'children'), [dash.dependencies.Input('select_data', 'value')])
    def load_description(folder):
         # Load description
         file = glob.glob(folder+'/*.yml')
         if(len(file)>0):
             with open(file[0], 'r') as ifile:
                 config_file = yaml.load(ifile)
             description = 'Description: ' + config_file['file']['description_full']
         else:
             description = 'No description entered.'
         return description
     
    @app.callback(
    dash.dependencies.Output('hours', 'max'),
    [dash.dependencies.Input('intermediate-value', 'children')])
    def set_slider_max(data):
        df = pd.read_json(data)
        return max(np.arange(df.time.min()/3600,df.time.max()/3600,1)[-1],1)
    
    @app.callback(
    dash.dependencies.Output('hours', 'min'),
    [dash.dependencies.Input('intermediate-value', 'children')])
    def set_slider_min(data):
        df = pd.read_json(data)
        return np.arange(df.time.min()/3600,df.time.max()/3600,1)[0]
    
    @app.callback(
    dash.dependencies.Output('hours', 'marks'),
    [dash.dependencies.Input('intermediate-value', 'children')])
    def set_slider_marks(data):
        df = pd.read_json(data)
        return {int(i): str(i) for i in np.arange(df.time.min()/3600,max(df.time.max()/3600,2),1)}
    
    @app.callback(
    dash.dependencies.Output('hours', 'value'),
    [dash.dependencies.Input('intermediate-value', 'children')])
    def set_slider_value(data):
        df = pd.read_json(data)
        times = np.arange(df.time.min()/3600,df.time.max()/3600,1)
        values = [times.min(),max(times.max(),1)]
        return values
    
    @app.callback(
    dash.dependencies.Output('variable', 'options'),
    [dash.dependencies.Input('intermediate-value', 'children')])
    def update_dropdown1(data):
        df = pd.read_json(data)
        labels = label_list(df.columns)
        values = dict((y,x) for x,y in labels.items())
        return [{'label': i, 'value': values[i]} for i in sorted(values)]
    
    @app.callback(
    dash.dependencies.Output('variable2', 'options'),
    [dash.dependencies.Input('intermediate-value', 'children')])
    def update_dropdown2(data):
        df = pd.read_json(data)
        labels = label_list(df.columns)
        values = dict((y,x) for x,y in labels.items())
        return [{'label': i, 'value': values[i]} for i in sorted(values)]
    
    @app.callback(
    dash.dependencies.Output('variable3', 'options'),
    [dash.dependencies.Input('intermediate-value', 'children')])
    def update_dropdown3(data):
        df = pd.read_json(data)
        labels = label_list(df.columns)
        values = dict((y,x) for x,y in labels.items())
        return [{'label': i, 'value': values[i]} for i in sorted(values)]
    
#    @app.callback(
#    dash.dependencies.Output('variable4', 'options'),
#    [dash.dependencies.Input('intermediate-value', 'children')])
#    def update_dropdown4(data):
#        df = pd.read_json(data)
#        labels = label_list(df.columns)
#        values = dict((y,x) for x,y in labels.items())
#        return [{'label': i, 'value': values[i]} for i in sorted(values)]
    
    @app.callback(
    dash.dependencies.Output('select_data', 'options'),
    [dash.dependencies.Input('refresh_button', 'n_clicks')])
    def update_output(n_clicks):
        folders = glob.glob('./Data/*')
        return [{'label': os.path.basename(i), 'value': i} for i in folders]
    
    return app
    
def fix_units(df):
    df['time_hr'] = df['time']/3600
    df['phi_deg'] = np.degrees(df['phi'])
    df['theta_deg'] = np.degrees(df['theta'])
    df['alpha_deg'] = np.degrees(df['alpha'])
    df['gamma_deg'] = np.degrees(df['gamma'])
    df['psi_deg'] = np.degrees(df['psi'])
    try:
        df['x_km'] = df['x_a']/1000
        df['y_km'] = df['y_a']/1000
        df['h_kft'] = df['h_a']*3.2808/1000
        df['h_km'] = df['h_a']/1000
    except:
        df['x_km'] = df['x']/1000
        df['y_km'] = df['y']/1000
        df['h_kft'] = df['h']*3.2808/1000
        df['h_km'] = df['h']/1000
    df['dist_km'] = df['dist']/1000
    df['te_kwh'] = df['te']*0.277778
    df['e_batt_kwh'] = df['e_batt']*0.277778
    df['t_hr'] = df['t']/3600
    df['psi_mod'] = np.mod(df['psi'],2*np.pi)
    df['psi_deg_mod'] = np.mod(df['psi_deg'],360)
    df['pinout'] = df['p_solar'] - df['p_n']
    try:
        df['L/D'] = df['cl']/df['cd']
    except:
        df['L/D'] = df['cl']/df['c_d']
    df['SOC'] = df['e_batt']/(df['e_batt'].iloc[0]/0.20)
    
    # Wind
    try:
        df['gamma_a_deg'] = np.degrees(df['gamma_a'])
        df['chi_deg'] = np.degrees(df['chi'])
    except:
        pass
    
    # Work calculations
    df['distance'] = np.sqrt(df['x'].diff()**2+df['y'].diff()**2+df['h'].diff()**2)
    df['distance'].iloc[0] = 0
    tp = df['tp'].as_matrix()
    df['worktp'] = np.r_[0,(tp[:-1]+tp[1:])/2*df['distance'].iloc[1:]]
    df['worktp'] = df['worktp'].cumsum()
    d = df['d'].as_matrix()
    df['workd'] = np.r_[0,(d[:-1]+d[1:])/2*df['distance'].iloc[1:]]
    df['workd'] = df['workd'].cumsum()
    
       
    return df

def load_units():
    units = {
        'time':'Time Since Dawn (s)',
        'time_hr':'Time Since Dawn (hr)',
        'tp':'Thrust (N)',
        'phi':'Bank Angle (rad)',
        'phi_deg':'Bank Angle (deg)',
        'theta':'Pitch (rad)',
        'theta_deg':'Pitch (deg)',
        'alpha':'Angle of Attack (rad)',
        'alpha_deg':'Angle of Attack (deg)',
        'gamma':'Flight Path Angle (rad)',
        'gamma_deg':'Flight Path Angle (deg)',
        'gamma_a':'Air Relative Flight Path Angle (rad)',
        'gamma_a_deg':'Air Relative Flight Path Angle (deg)',
        'psi':'Heading (rad)',
        'psi_mod':'Heading Wrapped (rad)',
        'psi_deg':'Heading (deg)',
        'psi_deg_mod':'Heading Wrapped (deg)',
        'chi_deg':'Course Angle (deg)',
        'v_g':'Ground Speed (m/s)',
        'v_a':'Airspeed (m/s)',
        'v':'Velocity (m/s)',
        'x':'North Position (m)',
        'x_km':'North Position (km)',
        'y':'East Position (m)',
        'y_km':'East Position (km)',
        'h':'Altitude (m)',
        'h_kft':'Altitude (kft)',
        'dist':'Distance from Center (m)',
        'dist_km':'Distance from Center (km)',
        'te':'Total Energy (MJ)',
        'te_kwh':'Total Energy (kWh)',
        'e_batt':'Battery Energy (MJ)',
        'e_batt_kwh':'Battery Energy (kWh)',
        'p_bat':'Power to Battery (W)',
        'p_solar':'Solar Power Recieved (W)',
        'panel_efficiency':'Solar Panel Efficiency',
        'd':'Drag (N)',
        'cd':'Drag Coefficient',
        'cl':'Lift Coefficient',
        'rho':'Air Density ()',
        'm':'Mass (kg)',
        'nh':'Horizontal Load Factor',
        'nv':'Vertical Load Factor',
        'nu_prop':'Propulsion Efficiency',
        't':'Time Since Midnight (s)',
        't_hr':'Time Since Midnight (hr)',
        'flux':'Solar Flux Available (W/m^2)',
        'g_sol':'Solar Flux Recieved (W/m^2)',
        'mu_solar':'Obliquity Factor',
        'azimuth':'Solar Azimuth (deg)',
        'zenith':'Solar Zenith (deg)',
        'p_n':'Power Needed (W)',
        'pinout':'Power In - Power Out (W)',
        're':'Reynolds Number',
        'iteration_time':'Solve Time (s)',
        'iterations':'Iterations',
        'SOC':'Battery State of Charge'
        }
    
    return units

def label_list(columns):
    units = load_units()
    labels = {}
    for variable in columns:
        if variable in units:
            labels[variable] = units[variable]
        else:
            labels[variable] = variable
    return labels
        

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    
    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
        
    return pl_colorscale

if __name__ == '__main__':
    app = define_app()
#    app.run_server(debug=True)
    app.run_server(host='0.0.0.0',port=8051,debug=True)