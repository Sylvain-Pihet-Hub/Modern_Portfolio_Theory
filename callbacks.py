import dash
from dash import Input, Output, State, ALL, callback_context, html, dcc
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from ModernPortfolioTheory import PortfolioOptimisation
from layout import fund_card_html

def register_callbacks(app, tickers: list[str] | None=None, df_prices: pd.DataFrame | None=None, frequency: int=252):

    # Populate fund cards dynamically
    @app.callback(
        Output("fund-cards", "children"),
        Input("fund-cards", "id")  # Trigger once on page load
    )
    def create_fund_widget(_):
        if df_prices is not None:
            return [fund_card_html(fund) for fund in df_prices.columns]
        else:
            return [fund_card_html(fund) for fund in tickers]

    @app.callback(
        Output("basket-store", "data"),
        Input({"type": "add-fund-button", "index": ALL}, "n_clicks"),
        #Input({"type": "back-to-selection-button", "index": ALL}, "n_clicks"),
        State("basket-store", "data"),
        prevent_initial_call=True)

    def add_fund_to_basket(n_clicks_list, current_basket):
        ctx = callback_context
        if not ctx.triggered:
            return current_basket
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        fund_name = json.loads(triggered_id)["index"]
        triggered_clicks = ctx.triggered[0]['value']
        if triggered_clicks is None or triggered_clicks == 0:
            return current_basket
        if fund_name not in current_basket:
            current_basket.append(fund_name)
        return current_basket

    @app.callback(
        Output("basket-list", "children"),
        Input("basket-store", "data"))

    def fetch_current_basket(basket_data):
        if not basket_data:
            return "Basket is empty"
        return html.Ul([html.Li(fund) for fund in basket_data])

    @app.callback(
        Output('url', 'pathname'),
        Input({"type": 'build-portfolio-button', "index": ALL}, 'n_clicks'),
        Input({'type': 'back-to-selection-button', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True)

    def navigate_pages(build_clicks, back_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        triggered_id = json.loads(triggered_id)
        if triggered_id["type"] == "build-portfolio-button":
            return '/analytics'
        elif triggered_id["type"] == "back-to-selection-button":
            return '/'
        return dash.no_update

    @app.callback(
        Output("basket-store", "data", allow_duplicate=True),
        Input({"type": "back-to-selection-button", "index": ALL}, "n_clicks"),
        prevent_initial_call=True)

    def reset_basket(back_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        triggered_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
        if triggered_id["type"] == "back-to-selection-button" and any(back_clicks):
            return []
        return dash.no_update

    # Provide return views as input parameters
    @app.callback(
        Output("bl-views-container", "children"),
        Input("expected-returns-type", "value"),
        State("basket-store", "data"))

    def display_black_litterman_views(selected_option, selected_funds):
        if selected_option != "black_litterman" or not selected_funds:
            return ""

        children = []
        for fund in selected_funds:
            children.append(
                html.Div(
                    style={"display": "flex", "alignItems": "center", "marginBottom": "6px"},
                    children=[
                        html.Div(fund, style={"width": "120px"}),
                        dcc.Input(id={"type": "bl-views-input", "index": fund}, type="number", placeholder="Expected return (%)", step=0.01, style={"width": "100px"}, value=0.0)
                    ]
                )
            )
        return children

    @app.callback(
        Output("efficient-frontier-graph", "figure"),
        Input("run-optimisation", "n_clicks"),
        State("basket-store", "data"),
        State("objective-function", "value"),
        State("covariance-type", "value"),
        State("min-weight", "value"),
        State("max-weight", "value"),
        State("expected-returns-type", "value"),
        State({"type": "bl-views-input", "index": ALL}, "value"),
        prevent_initial_call=True)

    def run_efficient_frontier(click, selected_funds, objective, covariance_type, min_weight, max_weight, selected_returns_option, bl_views):
        if not selected_funds or len(selected_funds) < 2:
            return go.Figure()

        view_returns_dict = None
        if selected_returns_option == "black_litterman":
            if not selected_funds or not bl_views:
                return go.Figure()
            view_returns_dict = {fund: expected_return for fund, expected_return in zip(selected_funds, bl_views)}

        if df_prices is not None:
            prices = df_prices[selected_funds]
            portfolio = PortfolioOptimisation(df_prices=prices, objective=objective, frequency=frequency, covariance_type=covariance_type, min_weight=min_weight, max_weight=max_weight, view_returns_dict=view_returns_dict)
        else:
            tickers = selected_funds
            portfolio = PortfolioOptimisation(tickers=tickers, objective=objective, frequency=frequency, covariance_type=covariance_type, min_weight=min_weight, max_weight=max_weight, view_returns_dict=view_returns_dict)

        ef_returns, ef_vols, ef_sharpe_ratios = portfolio.efficient_frontier
        optimised_portfolio_return, optimised_portfolio_vol, optimised_weights = portfolio.calculate_optimised_portfolio(returns_vector=portfolio.returns_vector, covariance_matrix=portfolio.covariance_matrix, objective=objective)
        mc_returns, mc_vols, _ = portfolio.simulate_random_portfolio()
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=mc_vols,
            y=mc_returns,
            mode="markers",
            name="Simulated Portfolios",
            marker=dict(size=4, opacity=0.3)
        ))

        fig.add_trace(go.Scatter(
            x=ef_vols, y=ef_returns, mode="lines+markers", name="Efficient Frontier"
        ))
        fig.add_trace(go.Scatter(
            x=[optimised_portfolio_vol], y=[optimised_portfolio_return],
            mode="markers", name="Max Sharpe",
            marker=dict(size=10, symbol="star")
        ))
        fig.update_layout(xaxis_title="Volatility", yaxis_title="Expected Return", template="plotly_white", paper_bgcolor="#F0F0F0", plot_bgcolor="#F0F0F0")
        return fig

    @app.callback(
        Output("correlation-heatmap", "figure"),
        Input("run-optimisation", "n_clicks"),
        State("basket-store", "data"),
        State("covariance-type", "value"),
        State("min-weight", "value"),
        State("max-weight", "value"),
        State("expected-returns-type", "value"),
        State({"type": "bl-views-input", "index": ALL}, "value"),
        prevent_initial_call=True)

    def run_correlation_matrix(click, selected_funds, covariance_type, min_weight, max_weight, selected_returns_option, bl_views):
        if not selected_funds or len(selected_funds) < 2:
            return go.Figure()

        view_returns_dict = None
        if selected_returns_option == "black_litterman":
            if not selected_funds or not bl_views:
                return go.Figure()
            view_returns_dict = {fund: expected_return for fund, expected_return in zip(selected_funds, bl_views)}

        if df_prices is not None:
            prices = df_prices[selected_funds]
            portfolio = PortfolioOptimisation(df_prices=prices, frequency=frequency, covariance_type=covariance_type, min_weight=min_weight, max_weight=max_weight, view_returns_dict=view_returns_dict)
        else:
            tickers = selected_funds
            portfolio = PortfolioOptimisation(tickers=tickers, frequency=frequency, covariance_type=covariance_type, min_weight=min_weight, max_weight=max_weight, view_returns_dict=view_returns_dict)
        covariance_matrix = portfolio.covariance_matrix
        corr_matrix = covariance_matrix / np.outer(np.sqrt(np.diag(covariance_matrix)), np.sqrt(np.diag(covariance_matrix)))

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdYlGn",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            text=corr_matrix.round(2).values,
            texttemplate="%{text}"
        ))
        cell_size = 100
        fig.update_layout(width=max(800, cell_size * len(selected_funds)), height=max(800, cell_size * len(selected_funds)), xaxis_tickangle=-90,
                          yaxis_autorange='reversed',paper_bgcolor="#F0F0F0", plot_bgcolor="#F0F0F0")
        return fig

    # Create optimised weights table
    @app.callback(
        Output("weights-table", "children"),
        Input("run-optimisation", "n_clicks"),
        State("basket-store", "data"),
        State("objective-function", "value"),
        State("covariance-type", "value"),
        State("min-weight", "value"),
        State("max-weight", "value"),
        State("expected-returns-type", "value"),
        State({"type": "bl-views-input", "index": ALL}, "value"),
        prevent_initial_call=True)

    def create_optimised_weights_table(n_clicks, selected_funds, objective, covariance_type, min_weight, max_weight, selected_returns_option, bl_views):
        if not selected_funds or len(selected_funds) < 2:
            return "No portfolio computed"

        view_returns_dict = None
        if selected_returns_option == "black_litterman":
            if not selected_funds or not bl_views:
                return go.Figure()
            view_returns_dict = {fund: expected_return for fund, expected_return in zip(selected_funds, bl_views)}

        if df_prices is not None:
            prices = df_prices[selected_funds]
            portfolio = PortfolioOptimisation(df_prices=prices, objective=objective, frequency=frequency, covariance_type=covariance_type, min_weight=min_weight, max_weight=max_weight, view_returns_dict=view_returns_dict)
        else:
            tickers = selected_funds
            portfolio = PortfolioOptimisation(tickers=tickers, objective=objective, frequency=frequency, covariance_type=covariance_type, min_weight=min_weight, max_weight=max_weight, view_returns_dict=view_returns_dict)
        optimised_portfolio_return, optimised_portfolio_vol, optimised_weights = portfolio.calculate_optimised_portfolio(returns_vector=portfolio.returns_vector, covariance_matrix=portfolio.covariance_matrix, objective=objective)

        # Convert to dataframe (handle both array or Series)
        weights_df = pd.DataFrame({"Fund": selected_funds, "Weight": optimised_weights})
        weights_df["Weight"] = (weights_df["Weight"] * 100).round(2)

        sharpe_ratio = round(optimised_portfolio_return / optimised_portfolio_vol, 2)
        metrics_df = pd.DataFrame({"Metric": ["Expected Return (%)", "Volatility (%)", "Sharpe Ratio"], "Value": [round(optimised_portfolio_return * 100, 2), round(optimised_portfolio_vol * 100, 2), sharpe_ratio]})

        return html.Table(
            [
                html.Thead(
                    html.Tr([html.Th("Fund"), html.Th("Weight (%)")])
                ),
                html.Tbody([
                    html.Tr([html.Td(row["Fund"]), html.Td(f"{row['Weight']}%")])
                    for _, row in weights_df.iterrows()
                ]),
                html.Tbody([
                    html.Tr([html.Td(colSpan=2, children=html.Hr(style={"margin": "8px 0"}))])
                ]),
                html.Tbody([
                    html.Tr([html.Th("Metric"), html.Th("Value")])
                ]),
                html.Tbody([
                    html.Tr([html.Td(row["Metric"]), html.Td(row["Value"])])
                    for _, row in metrics_df.iterrows()
                ])
            ],
            style={"width": "100%", "borderCollapse": "collapse", "background": "#F0F0F0", "borderRadius": "8px", "overflow": "hidden", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)", "padding": "8px"}
        )

    @app.callback(
        Output("portfolio-nav-graph", "figure"),
        Input("run-optimisation", "n_clicks"),
        State("basket-store", "data"),
        State("objective-function", "value"),
        State("covariance-type", "value"),
        State("min-weight", "value"),
        State("max-weight", "value"),
        State("expected-returns-type", "value"),
        State({"type": "bl-views-input", "index": ALL}, "value"),
        prevent_initial_call=True)

    def plot_nav_comparison(n_clicks, selected_funds, objective, covariance_type, min_weight, max_weight, selected_returns_option, bl_views):
        if not selected_funds or len(selected_funds) < 2:
            return go.Figure()

        view_returns_dict = None
        if selected_returns_option == "black_litterman":
            if not selected_funds or not bl_views:
                return go.Figure()
            view_returns_dict = {fund: expected_return for fund, expected_return in zip(selected_funds, bl_views)}

        if df_prices is not None:
            prices = df_prices[selected_funds].copy().dropna()
            portfolio = PortfolioOptimisation(df_prices=prices, objective=objective, frequency=frequency, covariance_type=covariance_type, min_weight=min_weight, max_weight=max_weight, view_returns_dict=view_returns_dict)
        else:
            tickers = selected_funds
            portfolio = PortfolioOptimisation(tickers=tickers, objective=objective, frequency=frequency, covariance_type=covariance_type, min_weight=min_weight, max_weight=max_weight, view_returns_dict=view_returns_dict)
        optimised_portfolio_return, optimised_portfolio_vol, optimised_weights = portfolio.calculate_optimised_portfolio(returns_vector=portfolio.returns_vector, covariance_matrix=portfolio.covariance_matrix, objective=objective)

        optimised_weights = np.array(optimised_weights)

        nav = portfolio.prices / portfolio.prices.iloc[0] * 100
        portfolio_nav = nav @ optimised_weights
        fig = go.Figure()
        for fund in nav.columns:
            fig.add_trace(go.Scatter(x=nav.index, y=nav[fund], mode="lines", name=fund, opacity=0.3, line=dict(width=2)))

        fig.add_trace(go.Scatter(x=portfolio_nav.index, y=portfolio_nav, mode="lines", name="Optimised Portfolio", line=dict(width=4, color="black")))
        fig.update_layout(xaxis_title="Date", yaxis_title="Indexed NAV (Base = 100)", paper_bgcolor="#F0F0F0", plot_bgcolor="#F0F0F0", legend=dict(orientation="h"), height=500)

        return fig
