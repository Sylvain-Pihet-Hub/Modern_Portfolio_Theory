import pandas as pd
import polars as pl
import numpy as np
import yfinance as yf
from pypfopt import risk_models, expected_returns, EfficientFrontier, CLA
from pypfopt.black_litterman import BlackLittermanModel
from matplotlib import pyplot as plt
import seaborn as sns


class HistoricalPerformance:

    def __init__(self, tickers: list[str] | None=None, df_prices: pd.DataFrame | None=None):
        self.tickers = tickers
        if tickers is not None:
            self.prices = yf.download(self.tickers, period="2y", threads=False, progress=False)["Close"]
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
        self.returns_vector = self.calculate_mean_historical_returns() if self.view_returns_dict is None else self.generate_expected_returns_with_black_litterman()
        self.efficient_frontier = self.calculate_efficient_frontier(self.returns_vector, self.covariance_matrix, self.objective)

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

    def calculate_efficient_frontier(self, returns_vector: pd.Series, covariance_matrix: pd.DataFrame, objective: str, n_points: int = 100):
        ef = EfficientFrontier(expected_returns=returns_vector, cov_matrix=covariance_matrix, weight_bounds=(self.min_weight, self.max_weight))
        ef_temporary = ef.deepcopy()
        frontier_returns, frontier_vols, sharpe_ratios = [], [], []
        if objective == "volatility":
            ef_temporary.max_sharpe()
            max_return = ef_temporary.portfolio_performance()[0]
            target_returns = np.linspace(start=min(returns_vector), stop=max_return*0.99, num=n_points)
            for target_return in target_returns:
                ef.efficient_return(target_return=target_return)  # give the optimised weights
                optimised_portfolio_return, optimised_portfolio_vol, sharpe_ratio = ef.portfolio_performance()
                if optimised_portfolio_vol not in frontier_vols:
                    frontier_returns.append(optimised_portfolio_return)
                    frontier_vols.append(optimised_portfolio_vol)
                    sharpe_ratios.append(sharpe_ratio)

        elif objective == "sharpe_ratio":
            ef_temporary.min_volatility()
            min_vol = ef_temporary.portfolio_performance()[1]
            print(f"Min Vol: {min_vol}", f"Upper Min Vol {1.001 * min_vol}")
            target_vols = np.linspace(start=min_vol * 1.01, stop=min_vol * 3, num=n_points)
            for target_vol in target_vols:
                ef.efficient_risk(target_volatility=target_vol)  # give the optimised weights
                optimised_portfolio_return, optimised_portfolio_vol, sharpe_ratio = ef.portfolio_performance()
                frontier_returns.append(optimised_portfolio_return)
                frontier_vols.append(optimised_portfolio_vol)
                sharpe_ratios.append(sharpe_ratio)
        else:
            raise ValueError("Objective has to be Returns or Volatility")

        return frontier_returns, frontier_vols, sharpe_ratios

    def calculate_efficient_frontier_with_cla(self, returns_vector: pd.Series, covariance_matrix: pd.DataFrame, objective: str):
        cla = CLA(expected_returns=returns_vector, cov_matrix=covariance_matrix, weight_bounds=(self.min_weight, self.max_weight))
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

    def calculate_optimised_portfolio(self, returns_vector: pd.Series, covariance_matrix: pd.DataFrame, objective: str):
        ef = EfficientFrontier(expected_returns=returns_vector, cov_matrix=covariance_matrix, weight_bounds=(self.min_weight, self.max_weight))
        if objective == "volatility":
            ef.min_volatility()
        elif objective == "sharpe_ratio":
            ef.max_sharpe()
        else:
            raise ValueError("Objective has to be Returns or Volatility")

        optimised_portfolio_return = ef.portfolio_performance()[0]
        optimised_portfolio_vol = ef.portfolio_performance()[1]
        optimised_weights = ef.weights
        return optimised_portfolio_return, optimised_portfolio_vol, optimised_weights

    def simulate_random_portfolio(self, n_simulations: int = 10_000):
        np.random.seed(122)
        n_assets = self.prices.shape[1]
        mc_returns = np.zeros(n_simulations)
        mc_volatility = np.zeros(n_simulations)
        mc_weights = []
        mu = self.returns_vector
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
