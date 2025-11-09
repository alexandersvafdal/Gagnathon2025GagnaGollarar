import dash
from dash import Dash, dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

debug = True

# --- == Professional "DataView Light" Color Palette == ---
COLOR_BACKGROUND = "#F8F9FA"
COLOR_CARD = "#FFFFFF"
COLOR_BORDER = "#DEE2E6"
COLOR_TEXT = "#212529"
COLOR_TEXT_MUTED = "#6C757D"
COLOR_PRIMARY_BLUE = "#007BFF"
COLOR_VALIDATION_TEAL = "#20C997"
COLOR_HISTORICAL_GRAY = "#6C757D"
COLOR_RESIDUAL_ORANGE = "#FD7E14"
COLOR_ACCENT_PURPLE = "#6F42C1"
COLOR_ACCENT_RED = "#DC3545"

COMPARISON_COLORS = [COLOR_PRIMARY_BLUE, COLOR_VALIDATION_TEAL, COLOR_RESIDUAL_ORANGE, COLOR_ACCENT_PURPLE, COLOR_ACCENT_RED]
PLOTLY_TEMPLATE = "plotly_white"

# --- LOAD HISTORICAL DATA AND PREDICTIONS ---
try:
    HISTORICAL_DATA = pd.read_parquet('historical_data.parquet')
    ALL_PREDICTIONS = pd.read_parquet('all_predictions.parquet')
    
    COMBINED_DATA = pd.concat([HISTORICAL_DATA, ALL_PREDICTIONS], ignore_index=True)
    
    COMBINED_DATA['timestamp'] = pd.to_datetime(COMBINED_DATA['timestamp'])
    COMBINED_DATA['year'] = COMBINED_DATA['timestamp'].dt.year
    COMBINED_DATA['value'] = COMBINED_DATA['y_true'].fillna(COMBINED_DATA['y_pred'])

    ALL_COUNTRIES = sorted(COMBINED_DATA['Country'].dropna().unique().tolist())
    ALL_BRANCHES = sorted(COMBINED_DATA['Branches'].dropna().unique().tolist())
    ALL_YEARS = sorted(COMBINED_DATA['year'].dropna().unique().astype(int).tolist())
    
    print(f"Loaded {len(HISTORICAL_DATA)} records and {len(ALL_PREDICTIONS)} predictions.")
except Exception as e:
    print(f"ERROR loading data: {e}")
    HISTORICAL_DATA, ALL_PREDICTIONS, COMBINED_DATA = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ALL_COUNTRIES, ALL_BRANCHES, ALL_YEARS = [], [], []

# --- Initialize the Dash App ---
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ]
)
server = app.server

# --- == Define the App Layout == ---
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.I(className="fa-solid fa-chart-line me-2", style={'color': COLOR_PRIMARY_BLUE})),
                        dbc.Col(dbc.NavbarBrand("Icelandic Export Data & Predictions", className="app-header")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="#",
                style={"textDecoration": "none"},
            ),
        ],
        fluid=True
    ),
    color="white",
    className="mb-4 app-navbar",
)


# --- == Helper Functions == ---

def create_graph_card(graph_id, height='65vh'):
    return dbc.Card(
        dbc.CardBody(
            dcc.Graph(
                id=graph_id, 
                figure={'layout': {'template': PLOTLY_TEMPLATE}}, 
                style={'height': height},
            )
        ),
        className="graph-card"
    )

def create_empty_fig(message="Please make a selection"):
    fig = go.Figure()
    fig.update_layout(
        template=PLOTLY_TEMPLATE, title=message,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message, xref="paper", yref="paper",
                showarrow=False, font=dict(size=16, color=COLOR_TEXT_MUTED)
            )
        ],
        transition_duration=500 
    )
    return fig

def format_yoy_change(change_pct):
    if change_pct > 0:
        icon = "fa-solid fa-arrow-up"
        color = "text-success"
    elif change_pct < 0:
        icon = "fa-solid fa-arrow-down"
        color = "text-danger"
    else:
        icon = "fa-solid fa-equals"
        color = "text-muted"
        
    return html.P([
        html.I(className=f"{icon} me-1"),
        f"{change_pct:+.1%}",
        html.Span(" vs. Previous Year", className="text-muted ms-1")
    ], className=f"mb-0 {color}")

# --- == Tab Layouts == ---

# --- Overview Tab ---
overview_tab = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.Label("Select Year for Overview", className="form-label"),
                    dcc.Dropdown(
                        id='overview-year-dropdown',
                        options=[{'label': str(year), 'value': year} for year in ALL_YEARS],
                        value=ALL_YEARS[-1] if ALL_YEARS else None,
                        clearable=False,
                        className='custom-dropdown'
                    )
                ]),
                className="graph-card"
            )
        ], md=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id='kpi-total-value', className="kpi-card-body")), md=3),
        dbc.Col(dbc.Card(dbc.CardBody(id='kpi-yoy-change', className="kpi-card-body")), md=3),
        dbc.Col(dbc.Card(dbc.CardBody(id='kpi-top-country', className="kpi-card-body")), md=3),
        dbc.Col(dbc.Card(dbc.CardBody(id='kpi-top-sector', className="kpi-card-body")), md=3),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(create_graph_card('overview-map-chart', height='60vh'), md=8),
        dbc.Col([
            dbc.Row(dbc.Col(create_graph_card('overview-top-countries-bar', height='30vh'))),
            dbc.Row(dbc.Col(create_graph_card('overview-top-sectors-bar', height='30vh')), className="mt-2"),
        ], md=4),
    ]),
])

# --- Analysis Dashboard Tab ---
dashboard_sidebar = dbc.Col(
    [
        dbc.Card(
            dbc.CardBody([
                html.H4([html.I(className="fa-solid fa-flag me-2"), " Select Export Data"], className="card-title"),
                html.Hr(),
                html.Label("Country", className="form-label"),
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': country, 'value': country} for country in ALL_COUNTRIES],
                    value=ALL_COUNTRIES[0] if ALL_COUNTRIES else None,
                    clearable=False,
                    className='custom-dropdown'
                ),
                
                html.Label("Branch / Product Category", className="form-label mt-3"),
                dcc.Dropdown(
                    id='branch-dropdown',
                    options=[],
                    value=None,
                    clearable=False,
                    className='custom-dropdown'
                ),
            ]),
            className="sidebar-card",
        ),
    ],
    md=3,
    id="sidebar"
)

# --- MODIFIED DASHBOARD CHARTS LAYOUT ---
dashboard_charts = dbc.Col(
    [
        dbc.Row([
            dbc.Col(create_graph_card('main-forecast-chart', height='50vh'), md=12),
        ]),
        dbc.Row([
            dbc.Col(create_graph_card('accuracy-chart', height='45vh'), md=4),
            dbc.Col(create_graph_card('residuals-hist', height='45vh'), md=4), # <-- NEW CHART
            dbc.Col(create_graph_card('residuals-chart', height='45vh'), md=4),
        ], className="mt-4"),
    ],
    md=9
)
# --- END MODIFICATION ---

dashboard_tab = html.Div([
    dbc.Row([
        dashboard_sidebar,
        dashboard_charts
    ])
])


# --- Country Comparison Tab ---
comparison_tab = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Sector to Compare", className="form-label"),
                    dcc.Dropdown(
                        id='comparison-branch-dropdown',
                        options=[{'label': branch, 'value': branch} for branch in ALL_BRANCHES],
                        value=ALL_BRANCHES[0] if ALL_BRANCHES else None,
                        clearable=False,
                        className='custom-dropdown'
                    )
                ], md=6),
                dbc.Col([
                    html.Label("Select up to 5 countries:", className="form-label"),
                    dcc.Dropdown(
                        id='comparison-countries',
                        options=[{'label': country, 'value': country} for country in ALL_COUNTRIES],
                        value=[ALL_COUNTRIES[0]] if ALL_COUNTRIES else [],
                        multi=True,
                        className='custom-dropdown'
                    )
                ], md=6),
            ]),
            dcc.Graph(
                id='comparison-chart', 
                figure={'layout': {'template': PLOTLY_TEMPLATE}}, 
                style={'height': '70vh', 'marginTop': '20px'},
            )
        ]),
        className="graph-card"
    )
])

# --- Composition Chart Tab ---
pie_chart_tab = html.Div([
    dbc.Card(
        dbc.CardBody([
            html.H5("Build Your Own Composition Chart", className="card-title"),
            html.P("Analyze the composition of exports for any given year.", className="card-text text-muted"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Year", className="form-label"),
                    dcc.Dropdown(
                        id='pie-year-dropdown',
                        options=[{'label': str(year), 'value': year} for year in ALL_YEARS],
                        value=ALL_YEARS[-1] if ALL_YEARS else None,
                        clearable=False,
                        className='custom-dropdown'
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Filter Countries (Optional)", className="form-label"),
                    dcc.Dropdown(
                        id='pie-country-dropdown',
                        options=[{'label': country, 'value': country} for country in ALL_COUNTRIES],
                        multi=True,
                        placeholder="All Countries (Total)",
                        className='custom-dropdown'
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Filter Sectors (Optional)", className="form-label"),
                    dcc.Dropdown(
                        id='pie-branch-dropdown',
                        options=[{'label': branch, 'value': branch} for branch in ALL_BRANCHES],
                        multi=True,
                        placeholder="All Sectors",
                        className='custom-dropdown'
                    )
                ], md=4),
            ]),
            dcc.Graph(
                id='custom-pie-chart',
                figure={'layout': {'template': PLOTLY_TEMPLATE}}, 
                style={'height': '60vh', 'marginTop': '20px'},
            )
        ]),
        className="graph-card"
    )
])

# --- == Main Layout == ---
app.layout = html.Div([
    navbar,
    dbc.Container(
        dbc.Tabs(id='main-tabs', children=[
            dbc.Tab(label="🏠 Overview", tab_id="tab-overview", children=[overview_tab]),
            dbc.Tab(label="📈 Single Series Analysis", tab_id="tab-dashboard", children=[dashboard_tab]),
            dbc.Tab(label="🌍 Country Comparison", tab_id="tab-comparison", children=[comparison_tab]),
            dbc.Tab(label="📊 Composition Chart", tab_id="tab-pie", children=[pie_chart_tab]),
        ], className="main-tabs"),
        fluid=True,
        className="dbc mt-4"
    ),
], style={'backgroundColor': COLOR_BACKGROUND})


# --- == Callbacks == ---

# --- Overview Tab Callback ---
@app.callback(
    Output('kpi-total-value', 'children'),
    Output('kpi-yoy-change', 'children'),
    Output('kpi-top-country', 'children'),
    Output('kpi-top-sector', 'children'),
    Output('overview-map-chart', 'figure'),
    Output('overview-top-countries-bar', 'figure'),
    Output('overview-top-sectors-bar', 'figure'),
    Input('overview-year-dropdown', 'value')
)
def update_overview_tab(selected_year):
    if not selected_year:
        empty_kpi = [html.H5("No Data"), html.H3("-")]
        empty_fig = create_empty_fig("Please select a year")
        return empty_kpi, empty_kpi, empty_kpi, empty_kpi, empty_fig, empty_fig, empty_fig

    dff_current_year = COMBINED_DATA[COMBINED_DATA['year'] == selected_year]
    dff_prev_year = COMBINED_DATA[COMBINED_DATA['year'] == selected_year - 1]

    # KPIs
    total_value = dff_current_year['value'].sum()
    kpi_total_val_children = [
        html.H5("Total Export Value", className="kpi-title"),
        html.H3(f"${total_value:,.0f}", className="kpi-value")
    ]
    
    prev_total_value = dff_prev_year['value'].sum()
    yoy_change = 0.0 if prev_total_value == 0 else (total_value - prev_total_value) / prev_total_value
    kpi_yoy_children = [
        html.H5("Year-over-Year Change", className="kpi-title"),
        html.H3(format_yoy_change(yoy_change), className="kpi-value")
    ]
    
    country_sum = dff_current_year.groupby('Country')['value'].sum()
    sector_sum = dff_current_year.groupby('Branches')['value'].sum()

    kpi_top_country_children = [html.H5("Top Country", className="kpi-title"), html.H3("-", className="kpi-value")]
    if not country_sum.empty:
        top_country_name = country_sum.idxmax()
        top_country_value = country_sum.max()
        kpi_top_country_children = [
            html.H5("Top Country", className="kpi-title"),
            html.H3(top_country_name, className="kpi-value"),
            html.P(f"${top_country_value:,.0f}", className="text-muted mb-0")
        ]

    kpi_top_sector_children = [html.H5("Top Sector", className="kpi-title"), html.H3("-", className="kpi-value")]
    if not sector_sum.empty:
        top_sector_name = sector_sum.idxmax()
        top_sector_value = sector_sum.max()
        kpi_top_sector_children = [
            html.H5("Top Sector", className="kpi-title"),
            html.H3(top_sector_name, className="kpi-value"),
            html.P(f"${top_sector_value:,.0f}", className="text-muted mb-0")
        ]

    # Map Figure
    map_data = dff_current_year.groupby('Country')['value'].sum().reset_index()
    map_fig = go.Figure(go.Choropleth(
        locations=map_data['Country'], locationmode='country names', z=map_data['value'],
        colorscale='Blues', colorbar_title="Export Value",
        marker_line_color='darkgray', marker_line_width=0.5
    ))
    map_fig.update_layout(
        title_text=f'Global Export Distribution {selected_year}',
        template=PLOTLY_TEMPLATE,
        geo=dict(showframe=False, showcoastlines=False, projection_type='mercator'),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Bar Charts
    top_5_countries = country_sum.nlargest(5).sort_values(ascending=True)
    top_5_sectors = sector_sum.nlargest(5).sort_values(ascending=True)

    country_bar_fig = go.Figure(go.Bar(
        y=top_5_countries.index, x=top_5_countries.values, orientation='h',
        text=top_5_countries.values, texttemplate='$%{x:,.0f}', textposition='outside',
        marker_color=COLOR_PRIMARY_BLUE
    ))
    country_bar_fig.update_layout(
        title='Top 5 Countries', template=PLOTLY_TEMPLATE,
        xaxis=dict(visible=False), yaxis=dict(title=''),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    sector_bar_fig = go.Figure(go.Bar(
        y=top_5_sectors.index, x=top_5_sectors.values, orientation='h',
        text=top_5_sectors.values, texttemplate='$%{x:,.0f}', textposition='outside',
        marker_color=COLOR_VALIDATION_TEAL
    ))
    sector_bar_fig.update_layout(
        title='Top 5 Sectors', template=PLOTLY_TEMPLATE,
        xaxis=dict(visible=False), yaxis=dict(title=''),
        margin=dict(l=10, r=10, t=30, b=10)
    )

    return (
        kpi_total_val_children, kpi_yoy_children, kpi_top_country_children, kpi_top_sector_children,
        map_fig, country_bar_fig, sector_bar_fig
    )

# --- Callbacks for "Single Series Analysis" Tab ---
@app.callback(
    Output('branch-dropdown', 'options'),
    Output('branch-dropdown', 'value'),
    Input('country-dropdown', 'value')
)
def update_branch_dropdown(selected_country):
    if not selected_country or COMBINED_DATA is None:
        return [], None
    filtered_branches = COMBINED_DATA[COMBINED_DATA['Country'] == selected_country]['Branches'].dropna().unique()
    branch_options = [{'label': branch, 'value': branch} for branch in sorted(filtered_branches)]
    default_value = sorted(filtered_branches)[0] if len(filtered_branches) > 0 else None
    return branch_options, default_value

@app.callback(
    Output('main-forecast-chart', 'figure'),
    Input('country-dropdown', 'value'),
    Input('branch-dropdown', 'value')
)
def update_main_forecast_chart(selected_country, selected_branch):
    if not selected_country or not selected_branch:
        return create_empty_fig("Please select a Country and Branch")
    
    fig = go.Figure()
    
    dff_hist = HISTORICAL_DATA[
        (HISTORICAL_DATA['Country'] == selected_country) & (HISTORICAL_DATA['Branches'] == selected_branch)
    ].copy()
    dff_preds = ALL_PREDICTIONS[
        (ALL_PREDICTIONS['Country'] == selected_country) & (ALL_PREDICTIONS['Branches'] == selected_branch)
    ].copy()
    
    if dff_preds.empty and dff_hist.empty:
        return create_empty_fig("No data available for this selection")

    if not dff_preds.empty:
        dff_preds['timestamp'] = pd.to_datetime(dff_preds['timestamp'])
        dff_preds = dff_preds.sort_values('timestamp')
        validation = dff_preds[dff_preds['prediction_type'] == 'Validation']
        future = dff_preds[dff_preds['prediction_type'] == 'Future']
        
        actual_data = []
        if not dff_hist.empty:
            dff_hist['timestamp'] = pd.to_datetime(dff_hist['timestamp'])
            actual_data.append(dff_hist[['timestamp', 'y_true']])
        if not validation.empty:
            actual_data.append(validation[['timestamp', 'y_true']])
        
        if actual_data:
            combined_actual = pd.concat(actual_data, ignore_index=True).sort_values('timestamp')
            fig.add_trace(go.Scatter(
                x=combined_actual['timestamp'], y=combined_actual['y_true'], 
                mode='lines', name='Historical Data', 
                line=dict(color=COLOR_HISTORICAL_GRAY, width=2, dash='dot'), opacity=0.7
            ))
        if not validation.empty:
            fig.add_trace(go.Scatter(
                x=validation['timestamp'], y=validation['y_pred'],
                mode='lines', name='Validation Prediction',
                line=dict(color=COLOR_VALIDATION_TEAL, width=2.5)
            ))
        if not future.empty:
            fig.add_trace(go.Scatter(
                x=future['timestamp'], y=future['y_pred'],
                mode='lines', name='Future Forecast',
                line=dict(color=COLOR_PRIMARY_BLUE, width=3)
            ))
    
    title_text = f"Forecast: {selected_country} - {selected_branch}"
    fig.update_layout(
        title=title_text, xaxis_title=None, yaxis_title="Value", template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(255,255,255,0.7)'),
        hovermode='x unified', margin=dict(t=50, b=10), transition_duration=500
    )
    return fig

@app.callback(
    Output('accuracy-chart', 'figure'),
    Input('country-dropdown', 'value'),
    Input('branch-dropdown', 'value')
)
def update_accuracy_chart(selected_country, selected_branch):
    if not selected_country or not selected_branch:
        return create_empty_fig("Select series to see accuracy")
    
    predictions_filtered = ALL_PREDICTIONS[
        (ALL_PREDICTIONS['Country'] == selected_country) & 
        (ALL_PREDICTIONS['Branches'] == selected_branch) &
        (ALL_PREDICTIONS['y_true'].notna())
    ].copy()
    
    if predictions_filtered.empty:
        return create_empty_fig("No validation data to display")
    
    predictions_filtered['timestamp'] = pd.to_datetime(predictions_filtered['timestamp'])
    fig = go.Figure()
    min_val = min(predictions_filtered['y_true'].min(), predictions_filtered['y_pred'].min())
    max_val = max(predictions_filtered['y_true'].max(), predictions_filtered['y_pred'].max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='Perfect Prediction',
        line=dict(color=COLOR_HISTORICAL_GRAY, width=2, dash='dash'),
    ))
    fig.add_trace(go.Scatter(
        x=predictions_filtered['y_true'], y=predictions_filtered['y_pred'],
        mode='markers', name='Validation',
        marker=dict(size=8, color=COLOR_VALIDATION_TEAL, opacity=0.6),
        text=predictions_filtered['timestamp'].dt.strftime('%Y-%m'),
        hovertemplate='<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Model Accuracy (Actual vs. Predicted)",
        xaxis_title="Actual Value", yaxis_title="Predicted Value",
        template=PLOTLY_TEMPLATE, hovermode='closest', showlegend=False,
        margin=dict(t=50, b=40), transition_duration=500
    )
    return fig

@app.callback(
    Output('residuals-chart', 'figure'),
    Input('country-dropdown', 'value'),
    Input('branch-dropdown', 'value')
)
def update_residuals_chart(selected_country, selected_branch):
    if not selected_country or not selected_branch:
        return create_empty_fig("Select series to see residuals")
    
    predictions_filtered = ALL_PREDICTIONS[
        (ALL_PREDICTIONS['Country'] == selected_country) & 
        (ALL_PREDICTIONS['Branches'] == selected_branch) &
        (ALL_PREDICTIONS['y_true'].notna())
    ].copy()
    
    if predictions_filtered.empty:
        return create_empty_fig("No validation data for residuals")
    
    predictions_filtered['timestamp'] = pd.to_datetime(predictions_filtered['timestamp'])
    predictions_filtered = predictions_filtered.sort_values('timestamp')
    predictions_filtered['residual'] = predictions_filtered['y_true'] - predictions_filtered['y_pred']
    
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_HISTORICAL_GRAY)
    fig.add_trace(go.Scatter(
        x=predictions_filtered['timestamp'], y=predictions_filtered['residual'],
        mode='markers+lines', name='Residuals',
        marker=dict(size=6, color=COLOR_RESIDUAL_ORANGE),
        line=dict(color=COLOR_RESIDUAL_ORANGE, width=1, dash='dot')
    ))
    
    fig.update_layout(
        title="Residuals Over Time",
        xaxis_title="Date", yaxis_title="Residual Value",
        template=PLOTLY_TEMPLATE, hovermode='x unified', showlegend=False,
         transition_duration=500, margin=dict(t=50, b=40)
    )
    return fig

# --- NEW CALLBACK FOR RESIDUALS HISTOGRAM ---
@app.callback(
    Output('residuals-hist', 'figure'),
    Input('country-dropdown', 'value'),
    Input('branch-dropdown', 'value')
)
def update_residuals_hist(selected_country, selected_branch):
    if not selected_country or not selected_branch:
        return create_empty_fig("Select series for histogram")
    
    predictions_filtered = ALL_PREDICTIONS[
        (ALL_PREDICTIONS['Country'] == selected_country) & 
        (ALL_PREDICTIONS['Branches'] == selected_branch) &
        (ALL_PREDICTIONS['y_true'].notna())
    ].copy()
    
    if predictions_filtered.empty:
        return create_empty_fig("No validation data for histogram")
    
    predictions_filtered['residual'] = predictions_filtered['y_true'] - predictions_filtered['y_pred']
    
    fig = go.Figure()
    
    # Add histogram trace
    fig.add_trace(go.Histogram(
        x=predictions_filtered['residual'],
        name='Residuals',
        marker_color=COLOR_RESIDUAL_ORANGE,
        opacity=0.7
    ))
    
    # Add a vertical line at 0
    fig.add_vline(
        x=0, 
        line_dash="dash", 
        line_color=COLOR_HISTORICAL_GRAY,
        annotation_text="Zero Error"
    )
    
    fig.update_layout(
        title="Residual Distribution",
        xaxis_title="Residual Value",
        yaxis_title="Count",
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        margin=dict(t=50, b=40),
        transition_duration=500,
        barmode='overlay' # Overlay histograms if we ever add more
    )
    return fig

# --- Callback for "Country Comparison" Tab ---
@app.callback(
    Output('comparison-chart', 'figure'),
    Input('comparison-branch-dropdown', 'value'),
    Input('comparison-countries', 'value')
)
def update_comparison_chart(selected_branch, selected_countries):
    if not selected_countries or not selected_branch:
        return create_empty_fig("Please select a sector and countries to compare")
    
    if len(selected_countries) > 5:
        selected_countries = selected_countries[:5]
    
    fig = go.Figure()
    
    for idx, country in enumerate(selected_countries):
        color = COMPARISON_COLORS[idx % len(COMPARISON_COLORS)]
        
        hist_data = HISTORICAL_DATA[
            (HISTORICAL_DATA['Country'] == country) & 
            (HISTORICAL_DATA['Branches'] == selected_branch)
        ].copy()
        if not hist_data.empty:
            hist_data['timestamp'] = pd.to_datetime(hist_data['timestamp'])
            hist_data = hist_data.sort_values('timestamp')
            fig.add_trace(go.Scatter(
                x=hist_data['timestamp'], y=hist_data['y_true'],
                mode='lines', name=country,
                line=dict(color=color, width=2), opacity=0.9
            ))
        
        forecast_data = ALL_PREDICTIONS[
            (ALL_PREDICTIONS['Country'] == country) & 
            (ALL_PREDICTIONS['Branches'] == selected_branch) &
            (ALL_PREDICTIONS['prediction_type'] == 'Future')
        ].copy()
        if not forecast_data.empty:
            forecast_data['timestamp'] = pd.to_datetime(forecast_data['timestamp'])
            forecast_data = forecast_data.sort_values('timestamp')
            fig.add_trace(go.Scatter(
                x=forecast_data['timestamp'], y=forecast_data['y_pred'],
                mode='lines', name=f'{country} (Forecast)',
                line=dict(color=color, width=2.5, dash='dash'), opacity=0.7
            ))
    
    fig.update_layout(
        title=f"Country Comparison: {selected_branch}",
        xaxis_title="Date", yaxis_title="Value",
        template=PLOTLY_TEMPLATE, hovermode='x unified',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor=COLOR_BORDER),
        transition_duration=500
    )
    return fig

# --- Callback for "Composition Chart" Tab ---
@app.callback(
    Output('custom-pie-chart', 'figure'),
    Input('pie-year-dropdown', 'value'),
    Input('pie-country-dropdown', 'value'),
    Input('pie-branch-dropdown', 'value')
)
def update_pie_chart(selected_year, selected_countries, selected_branches):
    if not selected_year:
        return create_empty_fig("Please select a year")
    
    dff = COMBINED_DATA[COMBINED_DATA['year'] == selected_year].copy()
    
    if selected_countries:
        dff = dff[dff['Country'].isin(selected_countries)]
        
    if selected_branches:
        dff = dff[dff['Branches'].isin(selected_branches)]
        
    pie_data = dff.groupby('Branches')['value'].sum().reset_index()
    pie_data = pie_data[pie_data['value'] > 0]
    
    if pie_data.empty:
        return create_empty_fig("No positive data found for this selection")
        
    title_text = f"Export Composition for {selected_year}"
    if selected_countries:
        title_text += f" | {', '.join(selected_countries)}"
    
    fig = go.Figure(data=[go.Treemap(
        labels=pie_data['Branches'],
        parents=[""] * len(pie_data['Branches']),
        values=pie_data['value'],
        textinfo="label+value+percent root",
        hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percent: %{percentRoot:.1%}<extra></extra>',
        marker_colorscale='Blues',
    )])
    
    fig.update_layout(
        title=title_text,
        template=PLOTLY_TEMPLATE,
        transition_duration=500,
        margin=dict(t=50, l=0, r=0, b=0)
    )
    
    return fig


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8050", debug=debug)