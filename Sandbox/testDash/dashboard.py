# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np


def define_app():
    app = dash.Dash()
    
    df = pd.read_excel('data.xlsx')
    df, units = fix_units(df)
    times = np.arange(df.time.min()/3600,df.time.max()/3600,1)
    
    app.layout = html.Div(children=[
        html.H1(children='Optimization Results'),
    
        html.Div([
        html.Div('Select Variable'),
        dcc.Dropdown(
                id='variable',
                options=[{'label': i, 'value': i} for i in df.columns],
                value='h'
            ),
    	dcc.RadioItems(
                    id='dimension',
                    options=[{'label': i, 'value': i} for i in ['2D', '3D']],
                    value='2D'
                )
        ],
        style={'width': '35%','display':'inline-block', 'vertical-align': 'top'}),
        html.Div([
        html.Div('Select Variable'),
        dcc.Dropdown(
                id='variable2',
                options=[{'label': i, 'value': i} for i in df.columns],
                value='te'
            )
        ],
        style={'width': '35%','display':'inline-block', 'vertical-align': 'top'}),
    
        html.Div([
        dcc.Graph(id='2D-plot')],style={'width': '35%', 'display': 'inline-block'}),
        html.Div([
        dcc.Graph(id='TE-plot')],style={'width': '35%', 'display': 'inline-block'}),
        html.Div([dcc.RangeSlider(
    		id='hours',
           #marks={i: 'Hour {}'.format(int(i)) if int(i) == 0 else str(int(i)) for i in np.arange(df.time.min()/3600,df.time.max()/3600,1)},
           marks={i: 'Hour {}'.format(int(i)) if int(i) == 0 else str(int(i)) for i in times},
    		count=1,
    		min=times.min(),
    		max=times.max(),
    		step=0.05,
    		value=[0, 1],
          allowCross=False,
          pushable=0.05
    		)],style={'width': '80%', 'display': 'inline-block'})
    ])
        
    @app.callback(
    dash.dependencies.Output('2D-plot', 'figure'),
    [dash.dependencies.Input('variable', 'value'),
    dash.dependencies.Input('hours', 'value'),
    dash.dependencies.Input('dimension', 'value')])
    def update_graph(variable,hours,dimension):
        
        df_filtered = df.iloc[int(hours[0]*3600/10):int(hours[1]*3600/10),:]
        print(hours)
        
        colors = df_filtered[variable]
        
        # Try to get pretty colorbar title
        if variable in units:
            colortitle = units[variable]
        else:
            colortitle = variable
    	
        if(dimension=='2D'):
            return {
    			'data': [
    					go.Scatter(
    						x=df_filtered['y_km'],
    						y=df_filtered['x_km'],
    						mode='lines+markers',
    						marker=dict(
    							size='10',
    							color = colors, #set color equal to a variable
                            colorbar = go.ColorBar(title=colortitle,titleside = 'right'),
    							colorscale='Jet',
    							showscale=True
    						),
                        text = colors,
    						opacity=0.7,
    					)
    				],
    				'layout': go.Layout(
    					width = 500,
    					height = 500,
    					title = 'Flight Path',
    					xaxis={'title': 'East (km)'},
    					yaxis={'title': 'North (km)'},
    					hovermode='closest'
    				)
    		}
        else:
            return {
            'data': [
                    go.Scatter3d(
                        x=df_filtered['y_km'],
                        y=df_filtered['x_km'],
                        z=df_filtered['h_kft'],
                        mode='lines+markers',
                        marker=dict(
                            size='3',
                            color = colors, #set color equal to a variable
                            colorbar = go.ColorBar(title=colortitle,titleside = 'right'),
                            colorscale='Jet',
                            showscale=True
                        ),
                        opacity=0.7
                    )
                ],
                'layout': go.Layout(
                    width = 500,
                    height = 500,
                    title = 'Flight Path 3D',
                    scene = go.Scene(
                            xaxis=go.XAxis(title='East'),
                            yaxis=go.YAxis(title='North'),
                            zaxis=go.ZAxis(title='Altitude',range=[df_filtered.h_kft.min()-1,df_filtered.h_kft.max()+1])
                            )
                )
        }
        
    @app.callback(
    dash.dependencies.Output('TE-plot', 'figure'),
    [dash.dependencies.Input('variable2', 'value'),
    dash.dependencies.Input('hours', 'value')])
    def update_graph(variable2,hours):
        
        df_filtered = df.iloc[int(hours[0]*3600/10):int(hours[1]*3600/10),:]
        
        values = df_filtered[variable2]
        
        # Try to get pretty axis title
        if variable2 in units:
            axistitle = units[variable2]
        else:
            axistitle = variable2
        
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
                    width = 500,
                    height = 500,
                    title = axistitle,
                    xaxis={'title': 'Time (hr)'},
    					yaxis={'title': axistitle},
                    hovermode='closest'
                )
        }
    
    return app
    
def fix_units(df):
    df['time_hr'] = df['time']/3600
    df['phi_deg'] = np.degrees(df['phi'])
    df['theta_deg'] = np.degrees(df['theta'])
    df['alpha_deg'] = np.degrees(df['alpha'])
    df['gamma_deg'] = np.degrees(df['gamma'])
    df['psi_deg'] = np.degrees(df['psi'])
    df['x_km'] = df['x']/1000
    df['y_km'] = df['y']/1000
    df['h_kft'] = df['h']*3.2808/1000
    df['dist_km'] = df['dist']/1000
    df['te_kwh'] = df['te']*0.277778
    df['e_batt_kwh'] = df['e_batt']*0.277778
    df['t_hr'] = df['t']/3600
    
    # Wind
    try:
        df['gamma_a_deg'] = np.degrees(df['gamma_a'])
        df['chi_deg'] = np.degrees(df['chi'])
    except:
        pass
        
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
        'psi_deg':'Heading (deg)',
        'chi_deg':'Course Angle (deg)',
        'v_g_actual':'Ground Speed (m/s)',
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
        'flux':'Available Solar Flux (W/m^2)',
        'g_sol':'Solar Flux Recieved (W/m^2)',
        'mu_solar':'Obliquity Factor',
        'azimuth':'Solar Azimuth (deg)',
        'zenith':'Solar Zenith (deg)'
        }
    
    return df, units

if __name__ == '__main__':
    app = define_app()
#    app.run_server(debug=True)
    app.run_server(host='0.0.0.0',port=8050,debug=True)