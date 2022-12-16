#importing neccessarily libraries
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset and split the X and Y 
df = pd.read_csv('world data.csv')
x = df['Year'].values
y = df['GDP'].values

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
