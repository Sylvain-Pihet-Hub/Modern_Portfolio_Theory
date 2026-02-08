import pandas as pd
import polars as pl
import numpy as np
import yfinance as yf
from pypfopt import risk_models, expected_returns, CLA
from pypfopt.black_litterman import BlackLittermanModel
from matplotlib import pyplot as plt
import seaborn as sns


class HistoricalPerformance:

    def __init__(self, tickers: list[str] | None=None, df_prices: pd.DataFrame | None=None):
        self.tickers = tickers
        if tickers is not None:
            self.prices = yf.download(self.tickers, period="3y", threads=False, progress=False)["Close"]
        else:
            self.prices = df_prices
        self.prices.index = pd.to_datetime(self.prices.index, format="%d/%m/%Y")

class PortfolioOptimisation(HistoricalPerformance):
    def __init__(self, tickers: list[str] | None=None, df_prices: pd.DataFrame | None=None, objective: str="volatility", frequency: int=12, covariance_type: str="sample_covariance",
                 min_weight: float=0.0, max_weight: float=1.0, view_returns_dict: dict[str, float]=None):
        super().__init__(tickers, df_prices)
        self.objective = objective
        self.frequency = frequency
        self.covariance_type = covariance_type
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.view_returns_dict = view_returns_dict
        self.covariance_matrix = self.calculate_covariance_matrix()
        self.returns_matrix = self.calculate_mean_historical_returns() if self.view_returns_dict is None else self.generate_expected_returns_with_black_litterman()
        self.efficient_frontier = self.calculate_efficient_frontier(self.returns_matrix, self.covariance_matrix, self.objective)

    def calculate_mean_historical_returns(self):
        mean_returns = expected_returns.mean_historical_return(self.prices, frequency=self.frequency)
        return mean_returns

    def calculate_covariance_matrix(self):
        if self.covariance_type == "shrunk_covariance":
            covariance_matrix = risk_models.CovarianceShrinkage(self.prices, frequency=self.frequency).ledoit_wolf()
        elif self.covariance_type == "sample_covariance":
            covariance_matrix = risk_models.sample_cov(prices=self.prices, frequency=self.frequency)
        else:
            raise ValueError("Invalid covariance type")
        return covariance_matrix

    def calculate_efficient_frontier(self, returns_vector: pd.Series, covariance_matrix: pd.DataFrame, objective: str):
        cla = CLA(returns_vector, covariance_matrix, weight_bounds=(self.min_weight, self.max_weight))
        if objective == "volatility":
            cla.min_volatility()
        elif objective == "sharpe_ratio":
            cla.max_sharpe()
        else:
            raise ValueError("Objective has to be Returns or Volatility")
        efficient_frontier = cla.efficient_frontier()

        optimised_portfolio_return = cla.portfolio_performance()[0]
        optimised_portfolio_vol = cla.portfolio_performance()[1]
        optimised_weights = cla.weights
        return efficient_frontier, optimised_portfolio_return, optimised_portfolio_vol, optimised_weights

    def simulate_random_portfolio(self, n_simulations: int = 10_000):
        np.random.seed(122)
        n_assets = self.prices.shape[1]
        mc_returns = np.zeros(n_simulations)
        mc_volatility = np.zeros(n_simulations)
        mc_weights = []
        mu = self.returns_matrix
        vol = self.covariance_matrix

        for i in range(n_simulations):
            weights = np.random.dirichlet(np.ones(n_assets))
            mc_weights.append(weights)
            mc_returns[i] = weights.T @ mu
            mc_volatility[i] = np.sqrt(weights.T @ vol @ weights)
        return mc_returns, mc_volatility, mc_weights

    def generate_expected_returns_with_black_litterman(self):
        prior_estimates = self.calculate_mean_historical_returns()
        bl = BlackLittermanModel(cov_matrix=self.covariance_matrix, pi=prior_estimates, absolute_views=self.view_returns_dict)
        review_returns = bl.bl_returns()
        return review_returns

    def calculate_max_drawdown(self, weights: np.ndarray, base: int=100):
        returns_series = self.prices.pct_change().dropna(axis=0) # maybe better to keep all funds with NA and adjust the weights
        base_row = pd.DataFrame(0, index=[returns_series.index[0] - pd.offsets.MonthEnd(1)], columns=returns_series.columns)
        returns_series = pd.concat([base_row, returns_series])
        normalised_nav = base * (1 + returns_series).cumprod() @ weights
        max_drawdown = (normalised_nav - normalised_nav.cummax()) / normalised_nav.cummax()
        return max_drawdown

    def calculate_sortino_ratio(self):
        pass

class Visualisation(PortfolioOptimisation):

    def __init__(self, tickers: list[str] | None=None, df_prices: pd.DataFrame | None=None, objective: str="volatility", frequency: int=12, covariance_type: str="sample_covariance"):
        super().__init__(tickers=tickers, df_prices=df_prices, objective=objective, frequency=frequency, covariance_type=covariance_type)

    def plot_efficient_frontier(self):
        plt.figure(figsize=(10, 6))
        efficient_returns, efficient_vols, efficient_weights = self.efficient_frontier[0]
        optimised_return, optimised_vol = self.efficient_frontier[1], self.efficient_frontier[2]
        mc_returns, mc_vols, mc_weights = self.simulate_random_portfolio()
        sns.scatterplot(x=efficient_vols, y=efficient_returns, label="Efficient_Frontier", color="green", s=40)
        sns.scatterplot(x=mc_vols, y=mc_returns, label="Simulated_Portfolios", color="grey", alpha=0.3, s=40)
        sns.scatterplot(x=[optimised_vol], y=[optimised_return], label="Max_Sharpe_Ratio", color="red", s=50)
        plt.xlabel("Volatility")
        plt.ylabel("Expected Returns")
        if self.objective == "volatility":
            plt.title("Efficient frontier for minium volatility")
        else:
            plt.title("Efficient frontier for maximum sharpe ratio")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(14, 10))
        corr_matrix = self.covariance_matrix / np.outer(np.sqrt(np.diag(self.covariance_matrix)), np.sqrt(np.diag(self.covariance_matrix)))
        #corr_matrix = risk_models.sample_cov(prices=self.prices, frequency=self.frequency).corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
        plt.title("Correlation Matrix Heatmap")
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.tight_layout()
        plt.show()

    def max_drawdown_graph(self):
        plt.figure(figsize=(10, 6))
        optimised_weights = self.efficient_frontier[3]
        max_drawdown = self.calculate_max_drawdown(weights=optimised_weights)
        pass

##### - Execution: Yahoo Finance Data - #####
# tickers = ["MSFT", "AAPL", "GOOGL", "AMZN"]
# data = HistoricalPerformance(tickers)
# prices = data.prices
# sample_covariance = prices.pct_change().cov() * 252
# prices.pct_change().mean() * 252
#
# portfolio = PortfolioOptimisation(tickers)
# mean_returns = portfolio.calculate_mean_historical_returns()
# covariance_shrinkage = portfolio.calculate_covariance_matrix()
# efficient_frontier = portfolio.calculate_efficient_frontier(mean_returns, covariance_shrinkage, "volatility")
#
# visuals = Visualisation(tickers)
# visuals.plot_efficient_frontier()

##### - Execution: External File Data - #####
# data = pd.read_excel(r"C:\Users\SylvainPihet\OneDrive - S64 Ventures Ltd\Documents\Bank_of_Singapore\Efficient_Frontier\Private_Markets_Performance_Data.xlsx")
# dates = data.iloc[:, ::2]
# monthly_dates = pd.date_range(start=dates.min().min(), end=dates.max().max(), freq="ME")
# df = pd.DataFrame({"Date": monthly_dates})
#
# for i in range(0, data.shape[1], 2):
#     new_dates = data.iloc[:, i]
#     new_returns = data.iloc[:, i + 1]
#     new_prices = pd.Series(index=new_returns.index)
#     first_valid_index = new_returns.first_valid_index()
#     if first_valid_index is not None:
#         new_prices.loc[first_valid_index] = 100
#         new_prices.loc[first_valid_index + 1: ] = new_prices.loc[first_valid_index] * (1 + new_returns).cumprod()
#     new_column = pd.DataFrame({"Date": new_dates, data.columns[i]: new_prices})
#     df = df.merge(new_column, on="Date", how="left")
#
# df = df.set_index("Date")
# pimco_interpolated = df[df.index > "31/05/2022"]["PIMCO PDLF"].interpolate(method="linear", axis=0)
# hg_interpolated = df[df.index > "30/11/2023"]["Hg Fusion Private Capital"].interpolate(method="linear", axis=0)
# df = df.drop(columns=["PIMCO PDLF", "Hg Fusion Private Capital"])
# df = df.merge(pimco_interpolated, on="Date", how="left")
# df = df.merge(hg_interpolated, on="Date", how="left")
#
# df_prices = df[df.index > "30/06/2023"]
# df_prices = df_prices.drop(columns=["MLoan SICAV, S.A.", "Carlyle AlpInvest Private Markets Sub-Fund"])
# df_prices = df_prices.iloc[:-1, :]
#
# portfolio = PortfolioOptimisation(df_prices=df_prices, frequency=12)
# portfolio.covariance_matrix
# portfolio.returns_matrix
# portfolio.efficient_frontier[1] / portfolio.efficient_frontier[2]
# portfolio.efficient_frontier[0]
#
# cla = CLA(portfolio.returns_matrix, portfolio.covariance_matrix, weight_bounds=(0, 1))
# cla.efficient_frontier()
# cla.max_sharpe()
# cla.portfolio_performance()
#
# df_prices.pct_change().cov() * 12
# np.exp(np.log(1 + df_prices.pct_change(fill_method=None)).mean() * 12) - 1
#
# visual_portfolio = Visualisation(df_prices=df_prices, frequency=12)
# visual_portfolio.plot_correlation_heatmap()
#
#
# portfolio.covariance_matrix / np.outer(np.sqrt(np.diag(portfolio.covariance_matrix)), np.sqrt(np.diag(portfolio.covariance_matrix)))

##### - Test Execution: External File Data - #####
# data = pd.read_excel(r"C:\Users\SylvainPihet\OneDrive - S64 Ventures Ltd\Documents\Python\Modern_Portfolio_Theory\Efficient_Frontier\Funds_Valuations_Data.xlsx", sheet_name="Raw_Data")
# data = data.set_index("Date")
# for col in data.columns:
#     ts = data[col]
#     first_index = ts.first_valid_index()
#     if first_index is not None:
#         ts.loc[first_index:] = ts.loc[first_index:].interpolate(method="linear")
#     data[col] = ts
#
# data.isna().sum()
# data = data.drop(columns=["MLoan SICAV S.A.", "Carlyle AelpInvest Private Markets Sub-Fund"])
#
# portfolio = PortfolioOptimisation(df_prices=data, frequency=12, objective="volatility", covariance_type="shrunk_covariance")
# len(portfolio.efficient_frontier[0][2])
# portfolio.efficient_frontier[0][1]
# portfolio.returns_matrix
#
# cla = CLA(portfolio.returns_matrix, portfolio.covariance_matrix, weight_bounds=(0, 1))
# cla.min_volatility()
# cla.portfolio_performance()
# cla.clean_weights().values()
# optimised_weights = cla.weights
#
# drawdown = portfolio.calculate_max_drawdown(weights=optimised_weights)
# drawdown_bis = portfolio.calculate_max_drawdown_bis(weights=optimised_weights)
#
# portfolio_visual = Visualisation(df_prices=data, frequency=12, objective="volatility", covariance_type="shrunk_covariance")
# portfolio_visual.plot_correlation_heatmap()
# portfolio_visual.plot_efficient_frontier()
# #
# # data.columns
# returns_series = portfolio.prices.pct_change().dropna(axis=0) # maybe better to keep all funds with NA and adjust the weights
# base_row = pd.DataFrame(0, index=[returns_series.index[0] - pd.offsets.MonthEnd(1)], columns=returns_series.columns)
# returns_series = pd.concat([base_row, returns_series])
# normalised_nav = 100 * (1 + returns_series).cumprod() @ optimised_weights
#
# combined_series = (100 * (1 + returns_series).cumprod()) @ optimised_weights
#
# prices_1 = portfolio.prices.dropna()
# 100 * (prices_1 / prices_1.iloc[0]) @ optimised_weights
