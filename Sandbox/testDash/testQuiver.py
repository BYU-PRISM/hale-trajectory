import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import glob
import os
import plotly.figure_factory as ff


df = pd.read_excel('data.xlsx')
df = df.iloc[::90,:]

z = np.zeros(len(df))

plt.quiver(z,z,df.sn1,df.sn2)

x = df.sn1*3000
y = df.sn2*3000

plt.quiver(x,y,-df.sn1,-df.sn2,pivot='tail')

fig = ff.create_quiver(x,y,-df.sn1*100,-df.sn2*100, scale=1, name='Sun')

points = go.Scatter(x=[0], y=[0],
                    mode='markers',
                    marker=dict(size=12),
                    name='Center')

layout = go.Layout(title = 'Quiver Figure',
                    xaxis={'title': 'X','range':[-3500,3500]},
        				yaxis={'title': 'Y','range':[-3500,3500]},
                    width = 500,
                    height = 500
                )

fig['data'].append(points)
fig['layout'] = (layout)

app = dash.Dash()

app.layout = html.Div([
    html.H3('Quiver Plot'),
    dcc.Graph(id='quiver', figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)