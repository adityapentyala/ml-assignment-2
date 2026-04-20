import plotly.express as px
import plotly.graph_objects as go

def plot_histogram(df, column):
    fig = px.histogram(df, x=column, nbins=70, title=f'Histogram of {column}')
    fig.update_layout(xaxis_title=column, yaxis_title='Count')
    return fig

def plot_boxplot(df, column):
    fig = px.box(df, y=column, title=f'Boxplot of {column}')
    fig.update_layout(yaxis_title=column)
    return fig

def plot_scatter(df, x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col, title=f'Scatter Plot of {y_col} vs {x_col}')
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    return fig

