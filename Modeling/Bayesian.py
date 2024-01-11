import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import math
import re
from scipy import stats
from typing import Union, List, Optional
from itertools import cycle
from matplotlib.pyplot import cm

class Gaussian:
    """
    Class for creating the multivariate Gaussian distribution.

    Attributes:
        - mean (np.ndarray): The mean vector of the Gaussian distribution.
        - cov (np.ndarray): The covariance matrix of the Gaussian distribution.

    Methods:
        - sample(N: int) -> np.ndarray: Generates random samples from the Gaussian distribution.

    """
    def __init__(self, mean, cov):
       
        self.mean = mean
        self.cov = cov

    def sample(self, N):
        return stats.multivariate_normal(
            mean=self.mean.reshape(-1), cov=self.cov, allow_singular=True
        ).rvs(N)


class BayesianLinearRegression:
    """
    Class for Bayesian Linear Regression.

    Attributes:
    - dataframe (pd.DataFrame): The input data in the form of a pandas DataFrame.
    - subject_name (str): Identifier for the subject (cow or farm).
    - selected_features (Union[str, List[str]]): The selected features for regression.
    - target (str): The target variable for regression.
    - off_set_bool (bool): Whether to include an offset/intercept term in the model.
    - subject_type (str): The type of the subject, either 'cow' or 'farm'.

    Methods:
    - fit_model(
        prior_mean: Optional[np.ndarray] = None,
        prior_cov: Optional[np.ndarray] = None,
        beta: Optional[float] = None,
    ) -> dict:
        Fits the Bayesian Linear Regression model to the input data.

    - plot_posterior_distributions() -> List[Figure]:
        Plots the posterior distributions of the model parameters.

    - assess_x_vector(ordered_by: str) -> np.ndarray:
        Checks and gets the desired x-vector for plotting.

    - plot_model_samples(
        n_samples: int = 3, ordered_by: str = "Temperature"
    ) -> Figure:
        Samples and plots 'n_samples' models from the posterior.

    - plot_model_uncertainty(
        ordered_by: str = "Temperature", every_th: int = None
    ) -> plt.Figure:
        Plots the mean-model and 1, 2, and 3-standard deviation uncertainties.

    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        subject_name: str,
        selected_features: Union[str, List[str]],
        target: str,
        off_set_bool: bool = True,
        subject_type: str = None,
    ) -> None:
        """
        Initializes the BayesianLinearRegression class.

        Args:
            - dataframe (pd.DataFrame): The input data in the form of a pandas DataFrame.
            - subject_name (str): Identifier for the subject (cow or farm).
            - selected_features (Union[str, List[str]]): The selected features for regression.
            - target (str): The target variable for regression.
            - off_set_bool (bool): Whether to include an offset/intercept term in the model.
            - subject_type (str): The type of the subject, either 'cow' or 'farm'.

        Returns:
            - None
        """
        # Initialize the class with the provided Bayesian data
        if subject_type is None:
            self.subject_type = subject_type
        elif subject_type.lower() not in ["cow", "farm"]:
            raise ValueError(
                'Invalid value for subject_type. Allowed values are "cow" or "farm".'
            )
        else:
            self.subject_type = subject_type  # If we are running the process on a cow or a farm
            
        if self.subject_type is None:
            self.dataframe = dataframe
            self.subject_name = subject_name
        elif self.subject_type.lower() == "cow":
            if subject_name not in dataframe["SE_Number"].unique():
                raise ValueError(f"Invalid cow id: {subject_name}")
            else:
                self.dataframe = dataframe[dataframe["SE_Number"] == subject_name]  # Get the relevant dataset for cow 'subject_name'
            self.subject_name = subject_name
        else:
            if subject_name not in dataframe["FarmName_Pseudo"].unique():
                raise ValueError(f"Invalid farm id : {subject_name}")
            else:
                self.dataframe = dataframe[dataframe["FarmName_Pseudo"] == subject_name]  # Get the relevant dataset for farm 'subject_name'

            self.subject_name = subject_name

        for feat_name in selected_features + target:
            if feat_name not in dataframe.columns:
                raise ValueError(
                    f"Invalid feature or target name: {feat_name}. Check column names in dataframe."
                )

        self.selected_features = selected_features  # A list of features to examine as independent variables, e.g. ['Temperatur', 'HW', 'THI_adj']
        self.target = target # a string indicating the target variable, often 'TotalYield'

        try:
            self.data_subset = self.dataframe[self.selected_features + self.target + ["StartDate", "DateTime"]].dropna() #Create the datasubset and remove nans, additonally adds StartDate and DateTime
            self.StartDate = self.data_subset.pop("StartDate").values # pop the StartDate column and save it in a variable, to be used later in plotting
            self.DateTime = self.data_subset.pop("DateTime").values # pop the DateTime column and save it in a variable, to be used later in plotting
            self.date_column_flag = "both" # A flag indicating wheter or not starttime/datetime exits in dataframe
        except KeyError:
            try:
                self.data_subset = self.dataframe[self.selected_features + self.target + ["DateTime"]].dropna()
                self.DateTime = self.data_subset.pop("DateTime").values
                self.date_column_flag = "DateTime"
            except KeyError:
                try:
                    self.data_subset = self.dataframe[self.selected_features + self.target + ["StartDate"]].dropna()
                    self.StartDate = self.data_subset.pop("StartDate").values
                    self.date_column_flag = "StartDate"
                except KeyError:
                    print("Warning: Neither 'StartDate' nor 'DateTime' found in the dataframe.")
                    self.date_column_flag = None

        self.y = self.data_subset[self.target].values # target saved in vecotr y
        self.X = self.data_subset[self.selected_features].values #The features, saved in the marix X
        self.off_set_bool = off_set_bool # wheter-or-not to include an offset/intercept term in the model
        if self.off_set_bool:
            self.selected_features.insert(0, "Off-set") # if inlcude the offset term, add it to the 'selected_feautres' list
        self.Phi = (
            np.concatenate([np.ones((len(self.X), 1)), self.X], axis=1)
            if off_set_bool
            else self.X) # create Phi, which is the vector with the input features
        
        self.posterior = None #init posterior and result
        self.result = None

        # Set seaborn theme
        sns.set_theme()
        sns.set_context("paper")

    def fit_model(
        self,
        prior_mean: Optional[np.ndarray] = None,
        prior_cov: Optional[np.ndarray] = None,
        beta: Optional[float] = None,
        ) -> dict:
        """
        Fits the model to the features 'self.Phi' to the desired target 'self.y'.

        Args:
        - prior_mean (Optional[np.ndarray]): Prior mean for Bayesian Linear Regression.
        - prior_cov (Optional[np.ndarray]): Prior covariance matrix for Bayesian Linear Regression.
        - beta (Optional[float]): Precision parameter for Bayesian Linear Regression.

        Returns:
        - dict: Dictionary containing the posterior distributions.
        """
        # priors p(\theta) = N(\theta; m0, S0)
        if prior_mean is None:
            self.prior_mean = np.zeros((self.Phi.shape[1], 1))
        else:
            self.prior_mean = prior_mean
        if prior_cov is None:
            self.prior_cov = np.eye(self.Phi.shape[1]) #/ self.Phi.shape[1] #np.identity(self.Phi.shape[1])
        else:
            self.prior_cov = prior_cov

        if beta is None:
            self.beta = 1 / np.var(self.y + 1e-6)

        if self.prior_mean.shape != (len(self.selected_features), 1):
            raise ValueError(
                f"Invalid shape for prior_mean. Expected shape: {(len(self.selected_features) + 1, 1)}, got: {self.prior_mean.shape}."
            )

        if self.prior_cov.shape != np.identity(self.Phi.shape[1]).shape:
            raise ValueError(
                f"Invalid shape for prior_cov. Expected shape: {np.identity(self.Phi.shape[1]).shape}, got: {self.prior_cov.shape}."
            )

        self.prior_cov_inv = np.linalg.inv(self.prior_cov) # inverse of prior covariance matrix
        self.y_reshaped = self.y.reshape(-1, 1) # reshape y to make the linear algebra work

        # posterior p(\theta | y) = N(\theta; mN, SN)
        self.posterior_cov = np.linalg.inv(
            np.linalg.inv(self.prior_cov) + self.beta * self.Phi.T @ self.Phi
            )  # calculate the posterior covariance matrix 
        
        self.posterior_mean = self.posterior_cov @ (
            np.linalg.inv(self.prior_cov) @ self.prior_mean
            + self.beta * self.Phi.T @ self.y_reshaped
            ) # calculate the posterior mean vector

        self.posterior = Gaussian(self.posterior_mean, self.posterior_cov) # save the postieror as a instance of the Gaussian class

        self.result = {} # save the results e.i. the posterior mean and variances, in a dict.
        for i in range(self.Phi.shape[1]):
            mu = self.posterior_mean[i]
            variance = self.posterior_cov[i, i]
            sigma = math.sqrt(variance)

            feature_name = self.selected_features[i]
            self.result[feature_name] = {"mu": mu[0], "sigma": sigma}

        return self.result

    def plot_posterior_distributions(self) -> List[Figure]:
        """
        Plot the posterior distributions of the model parameters.

        Returns:
        - List[Figure]: List of figures.
        """
        figures = []
        for i in range(len(self.selected_features)):
            mu = self.posterior.mean[i]
            variance = self.posterior.cov[i, i]
            sigma = math.sqrt(variance)
            fig, ax = plt.subplots(figsize=(6, 5))
            plt.title(
                f"Posterior Distribution - {self.selected_features[i]}.", 
                fontsize=16
                )
            xx = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            sns.lineplot(
                x=xx.flatten(),
                y=stats.norm.pdf(xx, mu, sigma).flatten(),
                label = f"PDF: $\mu = {mu.item():8.3f}$,\n \t $\sigma = {variance:8.3f}$",
                color="black",
                ax=ax,
                )
            sns.histplot(
                self.posterior.sample(500)[:, i],
                bins=30,
                kde=True,
                stat="density",
                alpha=0.5,
                color="blue",
                label="Samples",
                ax=ax,
                )

            
            plt.xlabel(f"Parameter Value", fontsize=14)
            plt.ylabel("Density", fontsize=14)
            plt.legend(fontsize=12, loc='upper right')
            plt.tight_layout()
            figures.append(fig)
        return figures

    def assess_x_vector(self, ordered_by: str) -> np.ndarray:
        """
        Function which checks and gets the desired x-vector ('ordered_by') to do plotting.

        Args:
        - ordered_by (str): Feature or target variable to use for ordering.

        Returns:
        - np.ndarray: X-vector for plotting.
        """
        if ordered_by not in self.selected_features + self.target + ["StartDate", "DateTime",]:
            raise ValueError(
                f"{ordered_by} is not a valid feature, target, 'StartDate', or 'DateTime'."
                )
        
        if ordered_by == "StartDate" and self.date_column_flag in ["StartDate", "both"]:
            x_vector = self.StartDate
        elif ordered_by == "DateTime" and self.date_column_flag in ["DateTime", "both"]:
            x_vector = self.DateTime
        else:
            x_vector = self.data_subset[ordered_by].values
        return x_vector

    def plot_model_samples(
        self, n_samples: int = 3, ordered_by: str = "Temperatur"
    ) -> Figure:
        """
        Sample and plot 'n_samples' models from the posterior.

        Args:
        - n_samples (int): Number of samples to generate and plot.
        - ordered_by (str): Feature or target variable to use for ordering.

        Returns:
        - Figure: Matplotlib Figure object representing the plot.
        """
        #For samples of w \theta, f(x) = phi(x)^T \theta
        samples = np.array([self.Phi @ t for t in self.posterior.sample(n_samples)]) # draw n_samples from the posterior
        x_vector = self.assess_x_vector(ordered_by=ordered_by)

        ordered_indices = np.argsort(x_vector) # sort the indices by the desired 'ordered_by' string
        ordered_x = x_vector[ordered_indices]
        ordered_samples = samples[:, ordered_indices]
        ordered_y = self.y[ordered_indices]
        # plot   
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)
        plt.scatter(ordered_x, ordered_y, zorder=1)
        color = cycle(cm.rainbow(np.linspace(0, 1, n_samples)))
        i = 1
        for sample, c in zip(ordered_samples, color):
            line, = ax.plot(ordered_x, sample, color=c, alpha=0.7, label = f"Sample {i}")
            i += 1

        plt.xlabel('Daily '+ re.sub(r'([a-z])([A-Z])', r'\1 \2', ordered_by), fontsize=14)
        plt.ylabel("Normalized Daily Yield", fontsize=14)
        plt.title(f"Posterior - samples for {self.subject_name}", fontsize=16)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return fig #return the figure

    def plot_model_uncertainty(
        self, ordered_by: str = "Temperatur", every_th: int = None
        ) -> plt.Figure:
        """
        Plots the mean-model and 1, 2, and 3 - standard deviation uncertainties.

        Args:
        - ordered_by (str): Feature or target variable to use for ordering.
        - every_th (int): Plot every-th data point for computational efficiency.

        Returns:
        - plt.Figure: Matplotlib Figure object representing the plot.
        """
        
        # calculating the uncertanty can be very computationally heavy, using Numba to speed up 
        @nb.njit(parallel=True)
        def compute_uncertainty_matrix(Phi, posterior_cov, beta, y_shape):
            # init uncertainty matrix
            uncs = np.zeros((y_shape, y_shape), dtype=np.float32)
            # first term: Phi @ posterior_cov @ Phi.T
            for i in nb.prange(Phi.shape[0]):
                for j in nb.prange(Phi.shape[1]):
                    for k in nb.prange(Phi.shape[1]):
                        uncs[i, j] += Phi[i, k] * posterior_cov[k, j]

            # add second term: beta**(-1) * np.eye(y_shape)
            for i in range(y_shape):
                uncs[i, i] += beta ** (-1)

            return np.sqrt(np.diag(uncs))

        x_vector = self.assess_x_vector(ordered_by=ordered_by)
        ordered_indices = np.argsort(x_vector) # sort the indices by the desired 'ordered_by' string
        ordered_x = x_vector[ordered_indices]
    
        ordered_y = self.y[ordered_indices]
        mean = self.Phi @ self.posterior_mean
        mean = mean.flatten()
        mean = mean[ordered_indices]

        uncs = compute_uncertainty_matrix(
            self.Phi, self.posterior_cov, self.beta, self.y.shape[0]
        )
        uncs = uncs[ordered_indices]
        # create a temporary dataframe for makeing plotting this easier
        plot_data = pd.DataFrame(
            {
                ordered_by: ordered_x,
                "Yield": ordered_y.flatten(),
                "Mean": mean,
                "Uncertainty": uncs,
            }
            )
        
        plot_data = plot_data.sort_values(by=ordered_by)
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)
        if every_th is None: #Sometimes data is huge, then you set every_th to e.g. 100 to only plot every 100th sample
            every_th = 1
        
        sns.scatterplot(
            x=ordered_by,
            y="Yield",
            data=plot_data.iloc[::every_th],
            zorder=10,
            alpha=0.4,
            color = '#183B87',
            ax = ax)
        
        sns.lineplot(
            x=ordered_by,
            y="Mean",
            data=plot_data.iloc[::every_th],
            color="black",
            label="Mean",
            ax=ax
            )
        
        ax.fill_between(
            plot_data[ordered_by].iloc[::every_th],
            (
                plot_data["Mean"].iloc[::every_th] - 3 * plot_data["Uncertainty"].iloc[::every_th]
            ),
            (
                plot_data["Mean"].iloc[::every_th] + 3 * plot_data["Uncertainty"].iloc[::every_th]
            ),
            color="lightgray",
            label=f'$Mean \pm 3\\sigma$',
            alpha=0.7,
        )

        ax.fill_between(
            plot_data[ordered_by].iloc[::every_th],
            (
                plot_data["Mean"].iloc[::every_th] - 2 * plot_data["Uncertainty"].iloc[::every_th]
            ),
            (
                plot_data["Mean"].iloc[::every_th] + 2 * plot_data["Uncertainty"].iloc[::every_th]
            ),
            color="darkgray",
            label=f'$Mean \pm 2\\sigma$',
            alpha=0.7,
        )
        ax.fill_between(
            plot_data[ordered_by].iloc[::every_th],
            (
                plot_data["Mean"].iloc[::every_th] - 1 * plot_data["Uncertainty"].iloc[::every_th]
            ),
            (
                plot_data["Mean"].iloc[::every_th] + 1 * plot_data["Uncertainty"].iloc[::every_th]
            ),
            color="gray",
            label=f'$Mean \pm 1\\sigma$',
            alpha=0.7,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.xlabel('Daily '+ re.sub(r'([a-z])([A-Z])', r'\1 \2', ordered_by), fontsize=14)
        plt.ylabel("Normalized Daily Yield", fontsize=14)
        plt.title(f"Bayesian Regression - {self.subject_name}", fontsize=16)

        return fig