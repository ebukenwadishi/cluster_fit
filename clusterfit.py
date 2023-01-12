#importing neccessarily libraries
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt

def read_world_bank_data(file_path: str) -> tuple:
    """
    Reads in a file in the World Bank format and returns a tuple containing the original dataframe and 
    a transposed version of the dataframe with meaningful column names.
    
    Args:
    file_path (str): The file path of the World Bank data file.
    
    Returns:
    tuple: A tuple containing the original dataframe and transposed dataframe with meaningful column names.
    """
    df = pd.read_csv(file_path)
    transposed_df = df.T
    transposed_df.columns = transposed_df.iloc[0]
    transposed_df = transposed_df[1:]
    return df.head()

new_df = read_world_bank_data('world data.csv')

x = new_df['Year'].values
y = new_df['GDP'].values

def fit_and_display(x, y, func, init_params, sigma=None, confidence=0.95):
    """
    Fit the data and display the result.
    
    Parameters:
    - x (array-like): Independent variable
    - y (array-like): Dependent variable
    - func (callable): Function to fit the data to. Should take x as the first argument and the parameters as the rest.
    - init_params (array-like): Initial guess for the parameters
    - sigma (array-like, optional): Uncertainties in y. Default is None.
    - confidence (float, optional): Confidence level for the confidence intervals. Default is 0.95.
    
    Returns:
    - popt (array-like): Optimized parameter values
    - pcov (2d array-like): Covariance matrix of the parameters
    """
    
    popt, pcov = curve_fit(func, x, y, p0=init_params, sigma=sigma)
    perr = np.sqrt(np.diag(pcov))
    
    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.plot(x, func(x, *popt), 'r-', label="Best fit")
    
    # Compute confidence intervals
    n = len(x)
    p = len(popt)
    dof = max(0, n - p)
    student_t = scipy.stats.t.ppf((1 + confidence) / 2, dof)
    lower = func(x, *(popt - student_t * perr))
    upper = func(x, *(popt + student_t * perr))
    plt.fill_between(x, lower, upper, color='gray', alpha=0.5, label="Confidence interval")
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
    return popt, pcov

#created the linear model to take the x,y,and b
def linear_model(x, a, b):
    return a * x + b
'''
created the linear model  
'''
params, cov = curve_fit(linear_model, x, y)

x_pred = np.arange(2022, 2031)
y_pred = linear_model(x_pred, params[0], params[1])

#plot the result of the observed and predicted data 
plt.plot(x, y, 'o', label='Observed Data')
plt.plot(x_pred, y_pred, '-', label='Predicted Data')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('GDP Fitted with a Linear Model')
plt.legend()
plt.show()

def func(x, a, b):
    return a*x + b

init_params = [1, 1]

fit_and_display(x, y, func, init_params)

#imported the DescrStatsW for the confidence interval
from statsmodels.stats.api import DescrStatsW

# Compute confidence interval with a 99% confidence level
descr = DescrStatsW(y)
conf_int = descr.tconfint_mean(alpha=0.01)

# Plot data
plt.plot(x,y, label="GDP")

# Add confidence interval to the plot
plt.fill_between(x, conf_int[0], conf_int[1], alpha=0.2, label="Confidence Interval")

# Add labels and show the plot
plt.xlabel("Year")
plt.ylabel("GDP")
plt.title("The confidence Interval of the Year over GDP")
plt.legend()
plt.show()

"""To perform fitting with curve_fit on GDP data, you will need a dataset that contains GDP values for different countries or regions over time. You can then use curve_fit to fit a mathematical model to the data and predict future GDP values.

The confidence interval plotted on the line chart provides a range of values that are likely to contain the true mean of the sample (in this case, the average GDP over the given years). The confidence level indicates the probability that the true mean falls within this range. For example, if the confidence level is 95%, there is a 95% probability that the true mean is within the confidence interval.


The confidence interval is fairly narrow, which suggests that the sample data is relatively consistent and the true mean (the average GDP over the given years) is likely to be close to the sample mean. However, you should always keep in mind that the confidence interval is just a statistical estimate, and there is always some uncertainty about the true mean of the sample.
"""

#loading the dataset for the clustering 
df = pd.read_csv('world data.csv')
X = df[['GDP','ratingimpact']]

#remove rows with any values that are not finite
df_new = X[np.isfinite(X).all(1)]

#importing and building a cluster model Kmeans 
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_new)

#plotted the result of the cluster from the Kmeans
import matplotlib.pyplot as plt
plt.scatter(df_new['GDP'], df_new['ratingimpact'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('GDP')
plt.ylabel('ratingimpact')
plt.title('Countries Clustered by GDP and rating impact')
plt.show()

"""The resulting plot shows two clusters, one with lower GDP and population values and another with higher GDP and population values. This suggests that there is a relationship between GDP and population, where countries with higher GDP tend to have larger populations.

To show an example with python, we can use the KMeans clustering algorithm to group countries into clusters based on certain features. For instance, we can use data on the gross domestic product (GDP), population, and life expectancy of different countries to group them into clusters.

We can also compare countries from different clusters and see how their GDP, population, and life expectancy differ. For example, we can pick a few countries from one cluster and compare them with countries from another cluster or different regions.
"""
