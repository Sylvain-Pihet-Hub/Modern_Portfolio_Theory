import pandas as pd
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
                 min_weight: float=0.0, max_weight: float=1.0, view_returns_dict: dict[str, float]=None, risk_free_rate: float=0.0):
        super().__init__(tickers, df_prices)
        self.objective = objective
        self.frequency = frequency
        self.covariance_type = covariance_type
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.view_returns_dict = view_returns_dict
        self.risk_free_rate = risk_free_rate
        self.covariance_matrix = self.calculate_covariance_matrix()
        self.returns_vector = self.calculate_mean_historical_returns() if self.view_returns_dict is None else self.generate_expected_returns_with_black_litterman()
        self.efficient_frontier = self.calculate_efficient_frontier(returns_vector=self.returns_vector,covariance_matrix=self.covariance_matrix, objective=self.objective)

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

    def calculate_efficient_frontier(self, returns_vector: pd.Series, covariance_matrix: pd.DataFrame, objective: str, n_points: int=100):
        ef = EfficientFrontier(expected_returns=returns_vector, cov_matrix=covariance_matrix, weight_bounds=(self.min_weight, self.max_weight))
        ef_temporary = ef.deepcopy()
        frontier_returns, frontier_vols, sharpe_ratios = [], [], []
        if objective == "volatility":
            target_returns = np.linspace(start=min(returns_vector), stop=ef_temporary._max_return()*0.98, num=n_points)
            for target_return in target_returns:
                ef.efficient_return(target_return=target_return) # give the optimised weights
                optimised_portfolio_return, optimised_portfolio_vol, sharpe_ratio = ef.portfolio_performance()
                if optimised_portfolio_vol not in frontier_vols:
                    frontier_returns.append(optimised_portfolio_return)
                    frontier_vols.append(optimised_portfolio_vol)
                    sharpe_ratios.append(sharpe_ratio)

        elif objective in ["sharpe_ratio", "utility_function"]:
            ef_temporary.min_volatility()
            min_vol = ef_temporary.portfolio_performance()[1]
            target_vols = np.linspace(start=min_vol*1.01, stop=min_vol*3, num=n_points)
            for target_vol in target_vols:
                ef.efficient_risk(target_volatility=target_vol) # give the optimised weights
                optimised_portfolio_return, optimised_portfolio_vol, sharpe_ratio = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
                frontier_returns.append(optimised_portfolio_return)
                frontier_vols.append(optimised_portfolio_vol)
                sharpe_ratios.append(sharpe_ratio)
        else:
            raise ValueError("Objective has to be Max Sharpe Ratio, Min Volatility, or Max Utility Function")

        return frontier_returns, frontier_vols, sharpe_ratios

    def calculate_efficient_frontier_with_cla(self, returns_vector: pd.Series, covariance_matrix: pd.DataFrame, objective: str, risk_free_rate: float=0.0):
        cla = CLA(expected_returns=returns_vector, cov_matrix=covariance_matrix, weight_bounds=(self.min_weight, self.max_weight))
        if objective == "volatility":
            cla.min_volatility()
        elif objective == "sharpe_ratio":
            cla.max_sharpe()
        else:
            raise ValueError("Objective has to be Returns or Volatility")
        efficient_frontier = cla.efficient_frontier()

        optimised_portfolio_return = cla.portfolio_performance(risk_free_rate=risk_free_rate)[0]
        optimised_portfolio_vol = cla.portfolio_performance(risk_free_rate=risk_free_rate)[1]
        optimised_weights = cla.weights
        return efficient_frontier, optimised_portfolio_return, optimised_portfolio_vol, optimised_weights

    def calculate_optimised_portfolio(self, returns_vector: pd.Series, covariance_matrix: pd.DataFrame, objective: str, risk_aversion: float):
        ef = EfficientFrontier(expected_returns=returns_vector, cov_matrix=covariance_matrix, weight_bounds=(self.min_weight, self.max_weight))
        if objective == "volatility":
            ef.min_volatility()
        elif objective == "sharpe_ratio":
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        elif objective == "utility_function":
            scaling_factor = np.mean(returns_vector) / np.mean(np.diag(covariance_matrix))
            effective_lambda = risk_aversion * scaling_factor
            ef.max_quadratic_utility(risk_aversion=effective_lambda)
        else:
            raise ValueError("Objective has to be Max Sharpe Ratio, Min Volatility, or Max Utility Function")

        optimised_portfolio_return = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)[0]
        optimised_portfolio_vol = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)[1]
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

class PCA:

    def __init__(self, prices, log_returns: bool=False):
        self.prices = prices
        self.returns = expected_returns.returns_from_prices(self.prices, log_returns=log_returns)

    def normalise_returns(self):
        returns = np.array(self.returns)
        mean_returns = np.nansum(returns, axis=0) / np.sum(~np.isnan(returns), axis=0)
        covariance_matrix = np.cov(returns, rowvar=False)
        std = np.sqrt(np.diag(covariance_matrix))
        normalised_returns = (returns - mean_returns) / std
        return normalised_returns, mean_returns, std

    def calculate_normalised_covariance(self):
        normalised_returns = self.normalise_returns()[0]
        normalised_covariance = normalised_returns.T @ normalised_returns / (normalised_returns.shape[0] - 1)
        return normalised_covariance

    def calculate_eigenvalues_and_eigenvectors(self):
        normalised_covariance = self.calculate_normalised_covariance()
        eigenvalues, eigenvectors = np.linalg.eig(normalised_covariance)
        sorted_index = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[sorted_index], eigenvectors[:, sorted_index]
        return eigenvalues, eigenvectors

    def calculate_projection_matrix(self, num_components: int):
        eigenvalues, eigenvectors = self.calculate_eigenvalues_and_eigenvectors()
        principal_values = np.real(eigenvalues[: num_components])
        principal_components = np.real(eigenvectors[:, :num_components])
        projection = principal_components @ np.linalg.inv(principal_components.T @ principal_components) @ principal_components.T
        return projection, principal_components, principal_values

    def pca(self, num_components: int):
        projection, principal_components, principal_values = self.calculate_projection_matrix(num_components)
        normalised_returns, mean_returns, std_returns = self.normalise_returns()
        reconstruct = (projection @ normalised_returns.T).T * std_returns + mean_returns
        return reconstruct, principal_components, principal_values

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
