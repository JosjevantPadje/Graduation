import getDatasets
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from collections import Counter

def pieChartCount(df, value, title):
    app = dash.Dash(__name__)

    horse_df = df.copy()

    horse_df[value] = [r.lower() for r in horse_df[value]]
    c = dict(Counter(horse_df[value]))

    v = list(c.keys())
    count = list(c.values())

    breed_dict = {value: v, 'count': count}
    breed_df = pd.DataFrame(breed_dict)

    fig = px.pie(breed_df, values='count', names=value, title=title)
    return fig

pieChartCount(getDatasets.get_horse_df(), 'Ras', 'Count of breeds').show()