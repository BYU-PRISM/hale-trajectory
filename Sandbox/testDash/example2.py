# -*- coding: utf-8 -*-

import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

N = 70
trace1 = go.Mesh3d(x=(70*np.random.randn(N)),
                   y=(55*np.random.randn(N)),
                   z=(40*np.random.randn(N)),
                   opacity=0.5,
                   color='rgba(244,22,100,0.6)'
                  )

layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        nticks=4, range = [-100,100],),
                    yaxis = dict(
                        nticks=4, range = [-50,100],),
                    zaxis = dict(
                        nticks=4, range = [-100,100],),),
                    width=700,
                    margin=dict(
                    r=20, l=10,
                    b=10, t=10)
                  )
fig = go.Figure(data=[trace1], layout=layout)
py.iplot(fig)