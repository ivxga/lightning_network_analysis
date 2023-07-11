import os
import string

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import pandas as pd
pd.options.plotting.backend = 'plotly'

def bumpchart(df, aliases=None, ax=None, holes=False, top=10, line_args={}, scatter_args={}, holes_args={}):
    
    left_yaxis = ax if ax else plt.gca()
    right_yaxis = left_yaxis.twinx()
    
    axes = [left_yaxis, right_yaxis]
    
    for col in df.columns:
        x, y = df.index.values, df[col]
        
        right_yaxis.plot(x, y, alpha=0)
        left_yaxis.plot(x, y, **line_args, solid_capstyle='round')
        
        left_yaxis.scatter(x, y, **scatter_args)
        
        if holes:
            bg_color = left_yaxis.get_facecolor()
            left_yaxis.scatter(x, y, color= bg_color, **holes_args)

    y_ticks = [*range(1, len(df.columns) + 1)]
    
    for axis in axes:
        axis.invert_yaxis()
        axis.set_yticks(y_ticks)
        axis.set_ylim((top + 0.5, 0.5))
    
    right_labels = df.iloc[-1].sort_values().index
    if aliases:
        right_labels = [''.join(list(filter(lambda x: x in string.printable, aliases[k]))).strip() if k in aliases else k for k in right_labels]
    right_yaxis.set_yticklabels(right_labels)
    
    return axes

def double_scale_line_chart(data, feature1, feature2):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data[feature1],
        name=feature1
    ))

    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data[feature2],
        name=feature2,
        yaxis="y2"
    ))


    fig.update_layout(
        yaxis=dict(
            title=' '.join(list(map(str.capitalize, feature1.split('_')))),
            titlefont=dict(color="#0000ff"),
            tickfont=dict(color="#0000ff")
        ),
        yaxis2=dict(
            title=' '.join(list(map(str.capitalize, feature2.split('_')))),
            titlefont=dict(color="#FF0000"),
            tickfont=dict(color="#FF0000"),
            anchor="free",
            overlaying="y",
            side="right",
            position=0
        ),
    )

    # fig.update_layout(
    #     title_text=f"{' '.join(list(map(str.capitalize, feature1.split('_'))))} and {' '.join(list(map(str.capitalize, feature2.split('_'))))} during time",
    #     xaxis_title='Dump Dates',
    #     legend_title='Feature'
    # )

    fig.update_layout(
        autosize=False,
        width=1350,
        height=400
    )
    
    return fig

def line_chart_min_max_all(data, colors, alpha_fill=0.2, xaxis_title='', yaxis_title=''):
    fig = go.Figure()
    
    for measure in data:
        df = data[measure]
        color = colors[measure]
        fig.add_trace(go.Scatter(x=df.index, y=df.min(axis=1), mode='lines', line=dict(width=0, color=f'rgba{color[3:-1]}, {alpha_fill})'), fillcolor=f'rgba{color[3:-1]}, {alpha_fill})'))
        fig.add_trace(go.Scatter(x=df.index, y=df.max(axis=1), fill='tonexty', mode='lines', line=dict(width=0, color=f'rgba{color[3:-1]}, {alpha_fill})'), fillcolor=f'rgba{color[3:-1]}, {alpha_fill})'))
        fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -1], line=dict(width=2, color=color)))
        
    fig.update_layout(
        width=1350,
        height=700,
        showlegend=False,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig

def fixed_frac_attack_chart(data, fraction, measures=['betweenness', 'closeness', 'degree', 'pagerank', 'random'], xaxis_title='', yaxis_title='', showlegend=False):
    fixed = []
    for measure in data:
        curr = data[measure].loc[fraction].copy()
        curr.name = measure
        fixed.append(curr)
    df = pd.DataFrame(fixed).T

    fig = go.Figure()
    for measure in measures:
        if measure == 'degree':
            fig.add_trace(go.Scatter(x=df.index, y=df[measure], mode='lines', name=measure.capitalize(), line=dict(width=3)))
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df[measure], mode='lines', name=measure.capitalize(), line=dict(dash='longdash')))

    fig.update_layout(
        width=1350,
        height=500,
        showlegend=showlegend,
        yaxis_range=[0, df.max().max() + 5],
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig

def double_scale_bar_chart(data, features, hlines=[], barcolors=['blue', 'red'], title='', xaxis_title='Features'):

    fig = go.Figure()

    for ix, feat in enumerate(features):
        fig.add_trace(go.Bar(
            x=data.index,
            y=data[feat],
            name=' '.join([x.capitalize() for x in feat.split('_')]),
            offsetgroup=ix+1,
            yaxis=f'y{ix+1}',
            text=data[feat].apply(lambda x: f'{round(x, 3)}'),
            textfont=dict(size=10, color=barcolors[ix]),
            textposition='outside',
        ))

    for ix, (val, color) in enumerate(hlines):
        norm_val = (val / max(data[features[ix]])) * max(data[features[0]])
        fig.add_hline(y=norm_val, line_width=3, line_dash="dash", line_color=color)

    fig.update_layout(
        yaxis=dict(
            title=' '.join([x.capitalize() for x in features[0].split('_')[:3]]),
            titlefont=dict(color=barcolors[0]),
            tickfont=dict(color=barcolors[0]),
            range=[0, data[features[0]].max()*1.075]
        ),
        yaxis2=dict(
            title=' '.join([x.capitalize() for x in features[1].split('_')[:3]]),
            titlefont=dict(color=barcolors[1]),
            tickfont=dict(color=barcolors[1]),
            range=[0, data[features[1]].max()*1.075],
            anchor="free",
            overlaying="y",
            side="right",
            position=1,
        ),
    )

    fig.update_layout(
        title_text=title,
        xaxis_title=xaxis_title,
        showlegend=False
    )

    fig.update_layout(
        autosize=False,
        width=1350,
        height=500
    )

    return fig
