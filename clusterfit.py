
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

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
    return df

df = read_world_bank_data('world_data.csv')

df.head()

df['Year'].nunique()

df.shape

# plot a correlation matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), cmap='RdBu', center=0,ax=ax)
plt.show()

# bar plot
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")
plt.title('GDP per countries')
ax = sns.barplot(
    data= df,
    x= 'Country',
    y= 'GDP in USD')

# set figure size
plt.figure(figsize=(7, 5))
sns.set(style="whitegrid")
# plot using seaborn library
ax=sns.lineplot(x='Year', y='GDP in USD', hue='Country', style="Country",palette="Set2", markers=True, dashes=False, data=df, linewidth=2.5)

def group_df(feature):
    # create a new dataframe
    df_grouped=pd.DataFrame()

    # find average for each country
    df_grouped['Avg. ' + feature]=df.groupby('Country')[feature].mean()

    # set the index as a column - countries
    df_grouped['Country']=df_grouped.index

    # drop the index
    df_grouped.reset_index(drop=True, inplace=True)

    # sort the rows based of Avg Birth rate
    df_grouped.sort_values('Avg. '+feature, inplace=True, ascending=False)

    print("Avg. " + feature)
    display(df_grouped)
    
    return df_grouped

def plot_bar(df, x_feature, y_feature):
    # bar plot
    plt.figure(figsize=(8, 5))
    sns.set(style="whitegrid")
    ax = sns.barplot(
        data= df,
        x= x_feature,
        y= "Avg. " + y_feature)

df_birth=group_df('Birth Rate')
plot_bar(df_birth, 'Country', 'Birth Rate')

print("========================================================")
df_death=group_df('Death Rate')
plot_bar(df_death, 'Country', 'Death Rate')

# plot using seaborn library
plt.title('GDP over years')
ax=sns.lineplot(x='Year', y='GDP in USD', hue='Country', style="Country",palette="Set2", markers=True, dashes=False, data=df, linewidth=2.5)

# function to extract specific columns from the DFs for India and China
def form_in_cn_df():
    # for India
    indf=df[['Total Population', 'Electric Power Consumption(kWH per capita)', 'Country']]
    # for China
    cndf=df[['Total Population', 'Electric Power Consumption(kWH per capita)', 'Country']]
    # combine the two dataframes
    in_cn_df=pd.concat([indf, cndf])
    return in_cn_df

# get the desired data
in_cn_df=form_in_cn_df()
print("Few records from the selected features: ")
display(in_cn_df.head())

# scatter plot
plt.figure(figsize=(7, 5))
sns.set(style="whitegrid")
ax=sns.scatterplot(x='Total Population', y='Electric Power Consumption(kWH per capita)', hue='Country', palette="bright", data=in_cn_df)

# read the columns from the df for Canada
new_df=df.loc[3:, ['Electric Power Consumption(kWH per capita)','Total Population', 'Year']]

print("First few records of the data: ")
display(df.head())

# line plot
plt.figure(figsize=(6, 5))
sns.set(style="whitegrid")
plt.title('Total Population over Electric Power Consumption')
sns.lineplot(x='Total Population', y='Electric Power Consumption(kWH per capita)', palette="colorblind",data=new_df, linewidth=2.5)

# bar plot
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")
plt.title('Countries by Employment in Agriculture(%)')
ax = sns.barplot(
    data= df,
    x= 'Country',
    y= 'Employment in Agriculture(%)')

# bar plot
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")
plt.title('Countries by GDP in USD')
ax = sns.barplot(
    data= df,
    x= 'Country',
    y= 'GDP in USD')

df.columns

x = df['GDP in USD']
y = df['Year']

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
    plt.title('GDP Fitted with a Linear Model')
    plt.legend()
    plt.show()
    
    return popt, pcov

def func(x, a, b):
    return a*x + b

init_params = [1, 1]

from scipy.optimize import curve_fit
import numpy as np
import scipy
from scipy import stats

fit_and_display(x, y, func, init_params)

def linear_model(x, a, b):
    return a * x + b

params, cov = curve_fit(linear_model, x, y)

x_pred = np.arange(2022, 2031)
y_pred = linear_model(x_pred, params[0], params[1])

plt.plot(x, y, 'o', label='Observed Data')
plt.plot(x_pred, y_pred, '-', label='Predicted Data')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('GDP Fitted with a Linear Model')
plt.legend()
plt.show()

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

df.columns

X = df[['Total Population', 'Female Population', 'Male Population',
       'Birth Rate', 'Death Rate', 'Compulsory Education Dur.',
       'Employment in Industry(%)', 'Employment in Agriculture(%)',
       'Female Employment in Agriculture(%)',
       'Female Employment in Industry(%)', 'Unemployment(%)', 'GDP in USD',
       'National Income per Capita', 'Net income from Abroad',
       'Agriculture value added(in USD)',
       'Electric Power Consumption(kWH per capita)',
       'Renewable Energy Consumption (%)', 'Fossil Fuel Consumption (%)']]

#remove rows with any values that are not finite
df_new = X[np.isfinite(X).all(1)]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_new)

import matplotlib.pyplot as plt

plt.scatter(df_new['GDP in USD'], df_new['Electric Power Consumption(kWH per capita)'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('GDP')
plt.ylabel('Electric Power Consumption(kWH per capita)')
plt.title('Countries Clustered by GDP and Electric Power Consumption(kWH per capita)')
plt.show()

