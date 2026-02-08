from dash import dcc, html

def fund_card_html(fund_name: str):
    return html.Div(
        style={
            "border": "1px solid #E8EAEE",
            "border-radius": "12px",
            "padding": "16px",
            "background": "#F0F0F0",
            "box-shadow": "0 2px 6px rgba(0,0,0,0.1)",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "space-between",
            "alignItems": "center",
            "minHeight": "120px",
            "transition": "transform 0.2s, box-shadow 0.2s",
            "cursor": "pointer"
        },
        children=[
            html.Div(
                fund_name,
                style={
                    "font-weight": "700",
                    "margin-bottom": "10px",
                    "text-align": "center",
                    "font-size": "14px"
                }
            ),
            html.Button(
                "Add to Basket",
                id={"type": "add-fund-button", "index": fund_name},
                style={
                    "margin-top": "auto",
                    "padding": "6px 12px",
                    "border-radius": "8px",
                    "border": "none",
                    "background-color": "#FF7F50",
                    "color": "#fff",
                    "font-weight": "600",
                    "cursor": "pointer"
                }
            )
        ]
    )

def create_page1_layout():
    return html.Div(
        style={"padding": "20px", "background-color": "#F0F0F0"},
        children=[
            # --- Top title ---
            html.Div(
                "ARENA-OPTIMA",
                style={
                    "font-size": "32px",
                    "font-weight": "800",
                    "text-align": "center",
                    "color": "#333",
                    "margin-bottom": "30px",
                    "font-family": "Arial, sans-serif",
                    "letter-spacing": "1px"
                }
            ),

            # --- Main content panels ---
            html.Div(
                style={"display": "flex", "gap": "30px"},
                children=[
                    # --- Left panel: basket + build button ---
                    html.Div(
                        style={
                            "width": "25%",
                            "background": "#F0F0F0",
                            "padding": "20px",
                            "border-radius": "12px",
                            "box-shadow": "0 2px 6px rgba(0,0,0,0.1)"
                        },
                        children=[
                            html.H3("Portfolio Basket", style={"margin-bottom": "15px"}),
                            html.Div(
                                id="basket-list",
                                style={"margin-bottom": "20px", "minHeight": "200px"}
                            ),
                            html.Button(
                                "Build Portfolio",
                                id={"type": "build-portfolio-button", "index": 0},
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "padding": "12px",
                                    "border-radius": "8px",
                                    "border": "none",
                                    "background-color": "#FF7F50",
                                    "color": "#fff",
                                    "font-weight": "600",
                                    "cursor": "pointer",
                                    "font-size": "16px"
                                }
                            )
                        ]
                    ),

                    # --- Right panel: fund marketplace ---
                    html.Div(
                        style={
                            "width": "75%",
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))",
                            "gap": "20px"
                        },
                        children=[
                            html.Div(
                                id="fund-cards",
                                style={"display": "contents"}  # children will be fund cards
                            )
                        ]
                    )
                ]
            )
        ]
    )

def create_page2_layout():
    return html.Div(
        style={"display": "flex", "background-color": "#F0F0F0"},
        children=[
            html.Div(
                style={"width": "25%", "padding": "20px"},
                children=[
                    html.H3("Portfolio Settings"),
                    html.Button("Back to Fund Selection", id={"type": "back-to-selection-button", "index": 1}, n_clicks=0),
                    html.Br(), html.Br(),
                    html.Label("Objective"),
                    dcc.RadioItems(
                        id="objective-function",
                        options=[{"label": "Min Volatility", "value": "volatility"}, {"label": "Max Sharpe Ratio", "value": "sharpe_ratio"}],
                        value="volatility"
                    ),
                    html.Br(),
                    html.Label("Covariance Matrix Type"),
                    dcc.RadioItems(
                        id="covariance-type",
                        options=[{"label": "Sample Covariance", "value": "sample_covariance"}, {"label": "Shrunk Covariance", "value": "shrunk_covariance"}],
                        value="shrunk_covariance"),
                    html.Br(),
                    html.Label("Min weight per fund: "),
                    dcc.Input(id="min-weight", type="number", min=0, max=1, step=0.01, value=0.0, style={"margin-top": "20px", "width": "80px"}),
                    html.Br(),
                    html.Label("Max weight per fund: "),
                    dcc.Input(id="max-weight", type="number", min=0, max=1, step=0.01, value=1.0, style={"margin-top": "10px", "width": "80px"}),
                    html.Br(),
                    html.Label("Risk-Free Rate: "),
                    dcc.Input(id="risk-free-rate", type="number", value=0.0, step=0.01, style={"margin-top": "20px"}),

                    html.H4("Expected Returns"),
                    html.Div(
                        style={"marginBottom": "15px"},
                        children=[
                            dcc.RadioItems(
                                id="expected-returns-type",
                                options=[{"label": "Historical Mean Returns", "value": "historical"}, {"label": "Black-Litterman Adjusted Returns", "value": "black_litterman"}],
                                value="historical",
                                labelStyle={"display": "block", "marginBottom": "4px"})
                        ]
                    ),
                    html.Div(id="bl-views-container"),
                    html.Button("Run Optimisation", id="run-optimisation", n_clicks=0, style={"margin-top": "20px"})

                ]
            ),

            html.Div(
                style={"width": "75%", "padding": "20px", "background-color": "#F0F0F0"},
                children=[
                    html.H4("Efficient Frontier"),
                    dcc.Graph(id="efficient-frontier-graph"),
                    html.H4("Optimised Portfolio Weights"),
                    html.Div(id="weights-table"),
                    html.H4("NAV Time-Series — Funds vs Optimised Portfolio"),
                    dcc.Graph(id="portfolio-nav-graph"),
                    html.H4("Correlation Heatmap"),
                    dcc.Graph(id="correlation-heatmap"),
                ]
            )
        ]
    )
