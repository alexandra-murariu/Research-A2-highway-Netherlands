from collections import deque
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display
from datetime import datetime
from dateutil.parser import parse
from statsmodels.distributions.empirical_distribution import ECDF
import warnings

warnings.filterwarnings("ignore", 'This pattern has match groups')

pd.options.mode.chained_assignment = None  # default='warn'

# loading the data
data = pd.read_csv("2016-09_rws_filedata.csv", sep=';', decimal=',')

####################################################################################################################
# EXERCISE 1
# storing the required columns in a new dataframe
df = data[['DatumFileBegin', 'TijdFileBegin', 'FileDuur', 'RouteOms', 'OorzaakGronddetail']]

# plotting the number of jams on each highway
df['RouteOms'].value_counts().plot(kind='bar', figsize=(8, 6), title='Number of jams on each highway', xlabel='highway',
                                   ylabel='number of jams')
# plt.show()

# exporting data in a CSV
dff = df['RouteOms'].value_counts().to_frame()
dff.to_csv("number_of_jams_by_highway.csv")

#######################################################################################################################
# EXERCISE 2

# making the weekday field
df['nr_accidents'] = df['OorzaakGronddetail'].map(df['OorzaakGronddetail'].value_counts())
df['weekday'] = df['DatumFileBegin'].apply(lambda x: parse(str(x)).strftime("%A"))

# plotting the number of jams by weekday on A2
df.loc[df['RouteOms'] == 'A2']['weekday'].value_counts().plot(kind='bar', stacked=True, figsize=(8, 6),
                                                              title='Number of jams on each weekday on A2',
                                                              xlabel='Day of week', ylabel='number of jams')
# plt.show()

# plot for Monday (the top 5 causes of jams)
df.loc[df['RouteOms'] == 'A2'].loc[df['weekday'] == 'Monday']['OorzaakGronddetail'].value_counts().head(5).plot(
    kind='barh', title='Monday',
    ylabel='number of jams',
    xlabel='cause of jams',
    figsize=(25, 6))
# plt.show()

# plot for Tuesday (the top 5 causes of jams)
df.loc[df['RouteOms'] == 'A2'].loc[df['weekday'] == 'Tuesday']['OorzaakGronddetail'].value_counts().head(5).plot(
    kind='barh', title='Tuesday',
    ylabel='number of jams',
    xlabel='cause of jams',
    figsize=(25, 6))
# plt.show()

# plot for Wednesday (the top 5 causes of jams)
df.loc[df['RouteOms'] == 'A2'].loc[df['weekday'] == 'Tuesday']['OorzaakGronddetail'].value_counts().head(5).plot(
    kind='barh', title='Wednesday',
    xlabel='number of jams',
    ylabel='cause of jams',
    figsize=(25, 6))
# plt.show()

# plot for Thursday (the top 5 causes of jams)
df.loc[df['RouteOms'] == 'A2'].loc[df['weekday'] == 'Thursday']['OorzaakGronddetail'].value_counts().head(5).plot(
    kind='barh', title='Thursday',
    xlabel='number of jams',
    ylabel='cause of jams',
    figsize=(25, 6))
# plt.show()

# plot for Friday (the top 5 causes of jams)
df.loc[df['RouteOms'] == 'A2'].loc[df['weekday'] == 'Friday']['OorzaakGronddetail'].value_counts().head(5).plot(
    kind='barh', title='Friday',
    xlabel='number of jams',
    ylabel='cause of jams',
    figsize=(25, 6))
# plt.show()

# plot for Saturday (the top 5 causes of jams)
df.loc[df['RouteOms'] == 'A2'].loc[df['weekday'] == 'Saturday']['OorzaakGronddetail'].value_counts().head(5).plot(
    kind='barh', title='Saturday',
    xlabel='number of jams',
    ylabel='cause of jams',
    figsize=(25, 6))
# plt.show()

# plot for Sunday (the top 5 causes of jams)
df.loc[df['RouteOms'] == 'A2'].loc[df['weekday'] == 'Sunday']['OorzaakGronddetail'].value_counts().head(5).plot(
    kind='barh', title='Sunday',
    xlabel='number of jams',
    ylabel='cause of jams',
    figsize=(25, 6))
# plt.show()

# exporting data regarding number of jams per weekday on A2 in a CSV
dff = df.loc[df['RouteOms'] == 'A2'].groupby(['weekday', 'OorzaakGronddetail']).size().to_frame()
dff.to_csv("causes_by_weekday.csv")

#################################################################################################################
# EXERCISE 3

df['DatumFileBegin'] = pd.to_datetime(df['DatumFileBegin'])
df['TijdFileBegin'] = pd.to_timedelta(df['TijdFileBegin'])
df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).to_csv("sorted_A2.csv")

# find the time distances between jams
time_distances = []
date = ""
time = ""
for index, row in df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).iterrows():
    if index != 0:
        if row['DatumFileBegin'] == date:
            time_distances.append((row['TijdFileBegin'] - time).total_seconds())
        else:
            time_distances.append(86400 - (time - row['TijdFileBegin']).total_seconds())
    date = row['DatumFileBegin']
    time = row['TijdFileBegin']

time_distances[0] = 0

# Time intervals for all jams plot

x = range(len(df.loc[df['RouteOms'] == 'A2']))
y = np.array(time_distances)

plt.plot(x, y)
plt.title('Time intervals between jams')
plt.xlabel('Index of jam')
plt.ylabel('Time interval in seconds')
# plt.show()

# Time intervals for first 100 jams plot
x = range(100)
y = np.array(time_distances)[:100]
plt.plot(x, y)
plt.title('Time intervals between first 100 jams')
plt.xlabel('Index of jam')
plt.ylabel('Time interval in seconds')
# plt.show()

# Cumulative time intervals plot for all jams

y = range(len(df.loc[df['RouteOms'] == 'A2']))
x = np.cumsum(np.array(time_distances))
print(x)
plt.plot(x, y)
plt.title('Time intervals between jams')
plt.ylabel('Index of jam')
plt.xlabel('Time interval in seconds')
# plt.show()

# Make a histogram
plt.figure()  # create a new plot window
plt.hist(time_distances, bins=39, rwidth=0.8, density=True)

# Fit a normal distribution

# First and second moment
M1 = np.mean(time_distances)  # first moment
M2 = np.mean(np.power(time_distances, 2))  # second moment

# Estimates for mu and sigma^2
mu = M1
sigma2 = M2 - M1 ** 2
fitNormDist = stats.norm(mu, sqrt(sigma2))

xs = np.arange(min(time_distances), max(time_distances), 10)

''' <- normal distribution against histogram
# Add theoretical density
ys = fitNormDist.pdf(xs)
plt.plot(xs, ys, color='red')
plt.title('Histogram against normal distribution fit')
plt.show()

'''

# Fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.pdf(xs), color='g')
plt.title('Histogram against exponential distribution fit')
# plt.show()

''' <- gamma distribution against histogram
# Fit a gamma distribution
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
plt.plot(xs, fitGammaDist.pdf(xs), color='r')
plt.title('Histogram against gamma distribution fit')
plt.show()
'''

# plot the ECDF
ecdf = ECDF(time_distances)
plt.figure()
plt.step(ecdf.x, ecdf.y, color='black', where='post', label='ECDF')
plt.plot(xs, fitNormDist.cdf(xs), color='b', label='normal distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(time_distances, fitNormDist.cdf)
print('KS Test Normal distribution: ' + str(tst1))
# KstestResult(statistic=0.4289808885632506, pvalue=4.714875122150363e-261)


# Shapiro-Wilk test for normality
tst2 = stats.shapiro(time_distances)
print('Shapiro-Wilk Test: ' + str(tst2))
# ShapiroResult(statistic=0.21883171796798706, pvalue=0.0)


# fit a gamma distribution
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)
fitGammaDist = stats.gamma(alpha, scale=1 / beta)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', label='gamma distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(time_distances, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst1))
# KstestResult(statistic=0.5515270054306297, pvalue=0.0)


# fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.cdf(xs), color='g', label='exponential distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(time_distances, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
# KstestResult(statistic=0.5002585736420017, pvalue=0.0)

# fit a uniform distribution
# (Careful: we are NOT using the method of moments here!!!!!)
a = min(time_distances)
b = max(time_distances)
fitUnifDist = stats.uniform(loc=a, scale=b - a)  # Careful! look at definition!
plt.plot(xs, fitUnifDist.cdf(xs), color='orange', label='uniform distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(time_distances, fitUnifDist.cdf)
print('KS Test Uniform distribution: ' + str(tst1))
# KstestResult(statistic=0.907567858836091, pvalue=0.0)

plt.title('Distribution fittings against ECDF')
plt.legend()
plt.xlabel('time intervals')
# plt.show()

# separate data into weekend data and working days data

# weekend data
df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] == 'Saturday') | (df['weekday'] == 'Sunday')].to_csv("sorted_A2_weekend.csv")

weekend_time_distances = []
for index, row in df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] == 'Saturday') | (df['weekday'] == 'Sunday')].iterrows():
    if index != 0:
        if row['DatumFileBegin'] == date:
            weekend_time_distances.append((row['TijdFileBegin'] - time).total_seconds())
        else:
            weekend_time_distances.append(86400 - time.total_seconds() + row['TijdFileBegin'].total_seconds())
    date = row['DatumFileBegin']
    time = row['TijdFileBegin']

weekend_time_distances[0] = 0

# working days data
df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')].to_csv("sorted_A2_workdays.csv")

workingdays_time_distances = []
for index, row in df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')].iterrows():
    if index != 0:
        if row['DatumFileBegin'] == date:
            workingdays_time_distances.append((row['TijdFileBegin'] - time).total_seconds())
        else:
            workingdays_time_distances.append(86400 - time.total_seconds() + row['TijdFileBegin'].total_seconds())
    date = row['DatumFileBegin']
    time = row['TijdFileBegin']

workingdays_time_distances[0] = 0

# weekend distribution fitting

# Cumulative time intervals plot for all jams

y = range(len(df.loc[df['RouteOms'] == 'A2'].loc[(df['weekday'] == 'Saturday') | (df['weekday'] == 'Sunday')]))
x = np.cumsum(np.array(weekend_time_distances))
print(x)
plt.plot(x, y)
plt.title('Time intervals between weekend jams')
plt.ylabel('Index of jam')
plt.xlabel('Time interval in seconds')
# plt.show()

# Make a histogram
plt.figure()  # create a new plot window
plt.hist(weekend_time_distances, bins=int(sqrt(len(weekend_time_distances))), rwidth=0.8, density=True)

# Fit a normal distribution

# First and second moment
M1 = np.mean(weekend_time_distances)  # first moment
M2 = np.mean(np.power(weekend_time_distances, 2))  # second moment

# Estimates for mu and sigma^2
mu = M1
sigma2 = M2 - M1 ** 2
fitNormDist = stats.norm(mu, sqrt(sigma2))

xs = np.arange(min(weekend_time_distances), max(weekend_time_distances), 10)

''' <- normal distribution against histogram
# Add theoretical density
ys = fitNormDist.pdf(xs)
plt.plot(xs, ys, color='red')
plt.title('Histogram against normal distribution fit (weekend)')
plt.xlabel('length of time distance in seconds')
plt.ylabel('number of time distances')
plt.show()

'''

# Fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.pdf(xs), color='g')
plt.title('Histogram against exponential distribution fit (weekend)')
# plt.show()

''' <- gamma distribution against histogram
# Fit a gamma distribution
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
plt.plot(xs, fitGammaDist.pdf(xs), color='r')
plt.title('Histogram against gamma distribution fit (weekend)')
plt.show()
'''

# plot the ECDF
ecdf = ECDF(weekend_time_distances)
plt.figure()
plt.step(ecdf.x, ecdf.y, color='black', where='post', label='ECDF (weekend)')
plt.plot(xs, fitNormDist.cdf(xs), color='b', label='normal distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(weekend_time_distances, fitNormDist.cdf)
print('KS Test Normal distribution: ' + str(tst1))
# KstestResult(statistic=0.2572919466246687, pvalue=0.026843257767918627)


# Shapiro-Wilk test for normality
tst2 = stats.shapiro(weekend_time_distances)
print('Shapiro-Wilk Test: ' + str(tst2))
# ShapiroResult(statistic=0.723546028137207, pvalue=2.636721546878107e-06)


# fit a gamma distribution
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)
fitGammaDist = stats.gamma(alpha, scale=1 / beta)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', label='gamma distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(weekend_time_distances, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst1))
# KstestResult(statistic=0.15155087712605453, pvalue=0.4322140629371565)


# fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.cdf(xs), color='g', label='exponential distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(weekend_time_distances, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
# KstestResult(statistic=0.31924186148097344, pvalue=0.002611234941497509)

# fit a uniform distribution
# (Careful: we are NOT using the method of moments here!!!!!)
a = min(weekend_time_distances)
b = max(weekend_time_distances)
fitUnifDist = stats.uniform(loc=a, scale=b - a)  # Careful! look at definition!
plt.plot(xs, fitUnifDist.cdf(xs), color='orange', label='uniform distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(weekend_time_distances, fitUnifDist.cdf)
print('KS Test Uniform distribution: ' + str(tst1))
# KstestResult(statistic=0.555966139401595, pvalue=1.517786245907751e-09)

plt.title('Distribution fittings against ECDF (weekend)')
plt.legend()
plt.xlabel('time intervals')
# plt.show()

# workdays distribution fitting

# Cumulative time intervals plot for all jams

y = range(len(df.loc[df['RouteOms'] == 'A2'].loc[(df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')]))
x = np.cumsum(np.array(workingdays_time_distances))
print(x)
plt.plot(x, y)
plt.title('Time intervals between working days jams')
plt.ylabel('Index of jam')
plt.xlabel('Time interval in seconds')
# plt.show()

# Make a histogram
plt.figure()  # create a new plot window
plt.hist(workingdays_time_distances, bins=int(sqrt(len(workingdays_time_distances))), rwidth=0.8, density=True)

# Fit a normal distribution

# First and second moment
M1 = np.mean(workingdays_time_distances)  # first moment
M2 = np.mean(np.power(workingdays_time_distances, 2))  # second moment

# Estimates for mu and sigma^2
mu = M1
sigma2 = M2 - M1 ** 2
fitNormDist = stats.norm(mu, sqrt(sigma2))

xs = np.arange(min(workingdays_time_distances), max(workingdays_time_distances), 10)

''' <- normal distribution against histogram
# Add theoretical density
ys = fitNormDist.pdf(xs)
plt.plot(xs, ys, color='red')
plt.title('Histogram against normal distribution fit (working days)')
plt.xlabel('length of time distance in seconds')
plt.ylabel('number of time distances')
plt.show()

'''

# Fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.pdf(xs), color='g')
plt.title('Histogram against exponential distribution fit (working days)')
# plt.show()

''' <- gamma distribution against histogram
# Fit a gamma distribution
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
plt.plot(xs, fitGammaDist.pdf(xs), color='r')
plt.title('Histogram against gamma distribution fit (working days)')
plt.show()
'''

# plot the ECDF
ecdf = ECDF(workingdays_time_distances)
plt.figure()
plt.step(ecdf.x, ecdf.y, color='black', where='post', label='ECDF (working days)')
plt.plot(xs, fitNormDist.cdf(xs), color='b', label='normal distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(workingdays_time_distances, fitNormDist.cdf)
print('KS Test Normal distribution: ' + str(tst1))
# KstestResult(statistic=0.4206697775197583, pvalue=1.2887219342158755e-245)

# Shapiro-Wilk test for normality
tst2 = stats.shapiro(workingdays_time_distances)
print('Shapiro-Wilk Test: ' + str(tst2))
# ShapiroResult(statistic=0.20386534929275513, pvalue=0.0)


# fit a gamma distribution
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)
fitGammaDist = stats.gamma(alpha, scale=1 / beta)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', label='gamma distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(workingdays_time_distances, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst1))
# KstestResult(statistic=0.5660371492990007, pvalue=0.0)


# fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.cdf(xs), color='g', label='exponential distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(workingdays_time_distances, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
# KstestResult(statistic=0.44946657074694424, pvalue=3.39569863544941e-282)

# fit a uniform distribution
# (Careful: we are NOT using the method of moments here!!!!!)
a = min(workingdays_time_distances)
b = max(workingdays_time_distances)
fitUnifDist = stats.uniform(loc=a, scale=b - a)  # Careful! look at definition!
plt.plot(xs, fitUnifDist.cdf(xs), color='orange', label='uniform distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(workingdays_time_distances, fitUnifDist.cdf)
print('KS Test Uniform distribution: ' + str(tst1))
# KstestResult(statistic=0.9131967098203935, pvalue=0.0)

plt.title('Distribution fittings against ECDF (working days)')
plt.legend()
plt.xlabel('time intervals')
# plt.show()

################################################################################################
# EXERCISE 4

# find the length of all traffic jams

df.loc[df['RouteOms'] == 'A2'].to_csv("A2_data.csv")
jam_durations = df.loc[df['RouteOms'] == 'A2']['FileDuur'].to_list()

# Time durations for all jams plot

df.loc[df['RouteOms'] == 'A2'].to_csv("A2_data.csv")
jam_durations = df.loc[df['RouteOms'] == 'A2']['FileDuur'].to_list()

# Time durations for all jams plot

x = range(len(jam_durations))
y = np.array(jam_durations)

plt.figure()
plt.plot(x, y)
plt.title('Duration of traffic jams')
plt.xlabel('Index of jam')
plt.ylabel('Time interval in minutes')
# plt.show()


# Time durations for first 100 jams plot
plt.figure()
x = range(100)
y = np.array(jam_durations)[:100]
plt.plot(x, y)
plt.title('Time durations for the first 100 jams')
plt.xlabel('Index of jam')
plt.ylabel('Duration of traffic jams in minutes')
# plt.show()


# Cumulative time durations plot for all jams
plt.figure()
y = range(len(df.loc[df['RouteOms'] == 'A2']))
x = np.cumsum(np.array(jam_durations))
print(x)
plt.plot(x, y)
plt.title('Cumulative traffic jams durations')
plt.ylabel('Index of jam')
plt.xlabel('Time durations in minutes')
plt.show()

# Make a histogram
plt.figure()  # create a new plot window
plt.hist(jam_durations, bins=39, rwidth=0.8, density=True)

# Fit a normal distribution

# First and second moment
M1 = np.mean(jam_durations)  # first moment
M2 = np.mean(np.power(jam_durations, 2))  # second moment

# Estimates for mu and sigma^2
mu = M1
sigma2 = M2 - M1 ** 2
fitNormDist = stats.norm(mu, sqrt(sigma2))

xs = np.arange(min(jam_durations), max(jam_durations), 10)

''' <- normal distribution against histogram
# Add theoretical density
ys = fitNormDist.pdf(xs)
plt.plot(xs, ys, color='red')
plt.title('Histogram against normal distribution fit')
plt.show()

'''

# Fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.pdf(xs), color='g')
plt.title('Histogram against exponential distribution fit')
# plt.show()

'''
 <- gamma distribution against histogram
# Fit a gamma distribution
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
plt.plot(xs, fitGammaDist.pdf(xs), color='r')
plt.title('Histogram against gamma distribution fit')
plt.show()
'''

# plot the ECDF
ecdf = ECDF(jam_durations)
plt.figure()
plt.step(ecdf.x, ecdf.y, color='black', where='post', label='ECDF')
plt.plot(xs, fitNormDist.cdf(xs), color='b', label='normal distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations, fitNormDist.cdf)
print('KS Test Normal distribution: ' + str(tst1))
# KstestResult(statistic=0.4289808885632506, pvalue=4.714875122150363e-261)


# Shapiro-Wilk test for normality
tst2 = stats.shapiro(jam_durations)
print('Shapiro-Wilk Test: ' + str(tst2))
# ShapiroResult(statistic=0.21883171796798706, pvalue=0.0)


# fit a gamma distribution
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)
fitGammaDist = stats.gamma(alpha, scale=1 / beta)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', label='gamma distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst1))
# KstestResult(statistic=0.5515270054306297, pvalue=0.0)


# fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.cdf(xs), color='g', label='exponential distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
# KstestResult(statistic=0.5002585736420017, pvalue=0.0)

# fit a uniform distribution
# (Careful: we are NOT using the method of moments here!!!!!)
a = min(jam_durations)
b = max(jam_durations)
fitUnifDist = stats.uniform(loc=a, scale=b - a)  # Careful! look at definition!
plt.plot(xs, fitUnifDist.cdf(xs), color='orange', label='uniform distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations, fitUnifDist.cdf)
print('KS Test Uniform distribution: ' + str(tst1))
# KstestResult(statistic=0.907567858836091, pvalue=0.0)

plt.title('Distribution fittings against ECDF')
plt.legend()
plt.xlabel('time durations')
plt.show()

# plot top 10 traffic jam causes
df.loc[df['RouteOms'] == 'A2']['OorzaakGronddetail'].value_counts().head(10).plot(
    kind='barh', title='Traffic jams',
    ylabel='number of jams',
    xlabel='cause of jams',
    figsize=(25, 6))
plt.show()

# now we sort the traffic jams by cause

# Spitsfile = rush hour jams - total number: 1377

df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('Spitsfile')) & (
    df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].to_csv('Spitsfile_jams.csv')

jam_durations_spitsfile = df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('Spitsfile')) & (
    df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))]['FileDuur'].to_list()

# Time durations for all jams plot

x = range(len(jam_durations_spitsfile))
y = np.array(jam_durations_spitsfile)

plt.figure()
plt.plot(x, y)
plt.title('Duration of Spitsfile')
plt.xlabel('Index of jam')
plt.ylabel('Time duration in minutes')
# plt.show()
print(jam_durations)

# Cumulative time intervals plot for all jams

y = range(len(df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('Spitsfile')) & (
    df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))]))
x = np.cumsum(np.array(jam_durations_spitsfile))
plt.figure()
plt.plot(x, y)
plt.title('Cumultative Spitsfile durations')
plt.ylabel('Index of jam')
plt.xlabel('Time duration in minutes')
plt.show()

# Make a histogram
plt.figure()  # create a new plot window
plt.hist(jam_durations_spitsfile, bins=int(sqrt(len(jam_durations_spitsfile))), rwidth=0.8, density=True)

# Fit a normal distribution

# First and second moment
M1 = np.mean(jam_durations_spitsfile)  # first moment
M2 = np.mean(np.power(jam_durations_spitsfile, 2))  # second moment

# Estimates for mu and sigma^2
mu = M1
sigma2 = M2 - M1 ** 2
fitNormDist = stats.norm(mu, sqrt(sigma2))

xs = np.arange(min(jam_durations_spitsfile), max(jam_durations_spitsfile), 10)

''' <- normal distribution against histogram
# Add theoretical density
ys = fitNormDist.pdf(xs)
plt.plot(xs, ys, color='red')
plt.title('Histogram against normal distribution fit (Spitsfile)')
plt.xlabel('length of jam duration in minutes')
plt.ylabel('number of jams')
plt.show()
'''

# Fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.pdf(xs), color='g')
plt.title('Histogram against exponential distribution fit (Spitsfile)')
# plt.show()


''' <- gamma distribution against histogram
# Fit a gamma distribution
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
plt.plot(xs, fitGammaDist.pdf(xs), color='r')
plt.title('Histogram against gamma distribution fit (Spitsfile)')
plt.show()

'''

# plot the ECDF
ecdf = ECDF(jam_durations_spitsfile)
plt.figure()
plt.step(ecdf.x, ecdf.y, color='black', where='post', label='ECDF (Spitsfile)')
plt.plot(xs, fitNormDist.cdf(xs), color='b', label='normal distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations_spitsfile, fitNormDist.cdf)
print('KS Test Normal distribution: ' + str(tst1))
# KstestResult(statistic=0.4206697775197583, pvalue=1.2887219342158755e-245)

# Shapiro-Wilk test for normality
tst2 = stats.shapiro(jam_durations_spitsfile)
print('Shapiro-Wilk Test: ' + str(tst2))
# ShapiroResult(statistic=0.20386534929275513, pvalue=0.0)


# fit a gamma distribution
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)
fitGammaDist = stats.gamma(alpha, scale=1 / beta)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', label='gamma distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations_spitsfile, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst1))
# KstestResult(statistic=0.5660371492990007, pvalue=0.0)


# fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.cdf(xs), color='g', label='exponential distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations_spitsfile, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
# KstestResult(statistic=0.44946657074694424, pvalue=3.39569863544941e-282)

# fit a uniform distribution
# (Careful: we are NOT using the method of moments here!!!!!)
a = min(jam_durations_spitsfile)
b = max(jam_durations_spitsfile)
fitUnifDist = stats.uniform(loc=a, scale=b - a)  # Careful! look at definition!
plt.plot(xs, fitUnifDist.cdf(xs), color='orange', label='uniform distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations_spitsfile, fitUnifDist.cdf)
print('KS Test Uniform distribution: ' + str(tst1))
# KstestResult(statistic=0.9131967098203935, pvalue=0.0)

plt.title('Distribution fittings against ECDF (Spitsfile)')
plt.legend()
plt.xlabel('durations')
plt.show()

# File buiten spits (geen oorzaak gemeld)  -- total number: 65

df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('File buiten spits')) & (
    df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].to_csv('File_buiten_spits_jams.csv')

jam_durations_filebuitenspits = \
    df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('File buiten spits')) & (
        df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))]['FileDuur'].to_list()

# Time durations for all jams plot

x = range(len(jam_durations_filebuitenspits))
y = np.array(jam_durations_filebuitenspits)

plt.figure()
plt.plot(x, y)
plt.title('Duration of File buiten spits')
plt.xlabel('Index of jam')
plt.ylabel('Time duration in minutes')
# plt.show()


# Cumulative time intervals plot for all jams

y = range(len(df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('File buiten spits')) & (
    df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))]))
x = np.cumsum(np.array(jam_durations_filebuitenspits))
plt.figure()
plt.plot(x, y)
plt.title('Cumultative File buiten spits durations')
plt.ylabel('Index of jam')
plt.xlabel('Time duration in minutes')
# plt.show()


# Make a histogram
plt.figure()  # create a new plot window
plt.hist(jam_durations_filebuitenspits, bins=int(sqrt(len(jam_durations_filebuitenspits))), rwidth=0.8, density=True)

# Fit a normal distribution

# First and second moment
M1 = np.mean(jam_durations_filebuitenspits)  # first moment
M2 = np.mean(np.power(jam_durations_filebuitenspits, 2))  # second moment

# Estimates for mu and sigma^2
mu = M1
sigma2 = M2 - M1 ** 2
fitNormDist = stats.norm(mu, sqrt(sigma2))

xs = np.arange(min(jam_durations_filebuitenspits), max(jam_durations_filebuitenspits), 10)

''' <- normal distribution against histogram
# Add theoretical density
ys = fitNormDist.pdf(xs)
plt.plot(xs, ys, color='red')
plt.title('Histogram against normal distribution fit (File buiten spits)')
plt.xlabel('length of jam duration in minutes')
plt.ylabel('number of jams')
plt.show()

'''

# Fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.pdf(xs), color='g')
plt.title('Histogram against exponential distribution fit (File buiten spits)')
# plt.show()


''' <- gamma distribution against histogram
# Fit a gamma distribution
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
plt.plot(xs, fitGammaDist.pdf(xs), color='r')
plt.title('Histogram against gamma distribution fit (File buiten spits)')
plt.show()

'''

# plot the ECDF
ecdf = ECDF(jam_durations_filebuitenspits)
plt.figure()
plt.step(ecdf.x, ecdf.y, color='black', where='post', label='ECDF (File buiten spits)')
plt.plot(xs, fitNormDist.cdf(xs), color='b', label='normal distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations_filebuitenspits, fitNormDist.cdf)
print('KS Test Normal distribution: ' + str(tst1))
# KstestResult(statistic=0.4206697775197583, pvalue=1.2887219342158755e-245)

# Shapiro-Wilk test for normality
tst2 = stats.shapiro(jam_durations_filebuitenspits)
print('Shapiro-Wilk Test: ' + str(tst2))
# ShapiroResult(statistic=0.20386534929275513, pvalue=0.0)


# fit a gamma distribution
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)
fitGammaDist = stats.gamma(alpha, scale=1 / beta)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', label='gamma distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations_filebuitenspits, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst1))
# KstestResult(statistic=0.5660371492990007, pvalue=0.0)


# fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.cdf(xs), color='g', label='exponential distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations_filebuitenspits, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
# KstestResult(statistic=0.44946657074694424, pvalue=3.39569863544941e-282)

# fit a uniform distribution
# (Careful: we are NOT using the method of moments here!!!!!)
a = min(jam_durations_filebuitenspits)
b = max(jam_durations_filebuitenspits)
fitUnifDist = stats.uniform(loc=a, scale=b - a)  # Careful! look at definition!
plt.plot(xs, fitUnifDist.cdf(xs), color='orange', label='uniform distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations_filebuitenspits, fitUnifDist.cdf)
print('KS Test Uniform distribution: ' + str(tst1))
# KstestResult(statistic=0.9131967098203935, pvalue=0.0)

plt.title('Distribution fittings against ECDF (File buiten spits)')
plt.legend()
plt.xlabel('durations')
plt.show()

# Ongeval(len) -- total number: 47

df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('Ongeval')) & (
    df['OorzaakGronddetail'].str.contains('len'))].to_csv('Ongeval_jams.csv')

jam_durations_ongeval = df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('Ongeval')) & (
    df['OorzaakGronddetail'].str.contains('len'))]['FileDuur'].to_list()

# Time durations for all jams plot

x = range(len(jam_durations_ongeval))
y = np.array(jam_durations_ongeval)

plt.figure()
plt.plot(x, y)
plt.title('Duration of Ongeval jams')
plt.xlabel('Index of jam')
plt.ylabel('Time duration in minutes')
# plt.show()


# Cumulative time intervals plot for all jams

y = range(len(df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('Ongeval')) & (
    df['OorzaakGronddetail'].str.contains('len'))]))
x = np.cumsum(np.array(jam_durations_ongeval))
plt.figure()
plt.plot(x, y)
plt.title('Cumultative Ongeval traffic jams durations')
plt.ylabel('Index of jam')
plt.xlabel('Time duration in minutes')
# plt.show()


# Make a histogram
plt.figure()  # create a new plot window
plt.hist(jam_durations_ongeval, bins=int(sqrt(len(jam_durations_ongeval))), rwidth=0.8, density=True)

# Fit a normal distribution

# First and second moment
M1 = np.mean(jam_durations_ongeval)  # first moment
M2 = np.mean(np.power(jam_durations_ongeval, 2))  # second moment

# Estimates for mu and sigma^2
mu = M1
sigma2 = M2 - M1 ** 2
fitNormDist = stats.norm(mu, sqrt(sigma2))

xs = np.arange(min(jam_durations_ongeval), max(jam_durations_ongeval), 10)

''' <- normal distribution against histogram
# Add theoretical density
ys = fitNormDist.pdf(xs)
plt.plot(xs, ys, color='red')
plt.title('Histogram against normal distribution fit (Ongeval jams)')
plt.xlabel('length of jam duration in minutes')
plt.ylabel('number of jams')
plt.show()


'''
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
'''
# Fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.pdf(xs), color='g')
plt.title('Histogram against exponential distribution fit (Ongeval jams)')
plt.show()
'''

''' <- gamma distribution against histogram
# Fit a gamma distribution
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
plt.plot(xs, fitGammaDist.pdf(xs), color='r')
plt.title('Histogram against gamma distribution fit (Ongeval jams)')
plt.show()

'''

# plot the ECDF
ecdf = ECDF(jam_durations_ongeval)
plt.figure()
plt.step(ecdf.x, ecdf.y, color='black', where='post', label='ECDF (Ongeval jams)')
plt.plot(xs, fitNormDist.cdf(xs), color='b', label='normal distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations_ongeval, fitNormDist.cdf)
print('KS Test Normal distribution: ' + str(tst1))
# KstestResult(statistic=0.4206697775197583, pvalue=1.2887219342158755e-245)

# Shapiro-Wilk test for normality
tst2 = stats.shapiro(jam_durations_ongeval)
print('Shapiro-Wilk Test: ' + str(tst2))
# ShapiroResult(statistic=0.20386534929275513, pvalue=0.0)


# fit a gamma distribution
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)
fitGammaDist = stats.gamma(alpha, scale=1 / beta)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', label='gamma distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations_ongeval, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst1))
# KstestResult(statistic=0.5660371492990007, pvalue=0.0)


# fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.cdf(xs), color='g', label='exponential distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations_ongeval, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
# KstestResult(statistic=0.44946657074694424, pvalue=3.39569863544941e-282)

# fit a uniform distribution
# (Careful: we are NOT using the method of moments here!!!!!)
a = min(jam_durations_ongeval)
b = max(jam_durations_ongeval)
fitUnifDist = stats.uniform(loc=a, scale=b - a)  # Careful! look at definition!
plt.plot(xs, fitUnifDist.cdf(xs), color='orange', label='uniform distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(jam_durations_ongeval, fitUnifDist.cdf)
print('KS Test Uniform distribution: ' + str(tst1))
# KstestResult(statistic=0.9131967098203935, pvalue=0.0)

plt.title('Distribution fittings against ECDF (Ongeval jams)')
plt.legend()
plt.xlabel('durations')
plt.show()

# There are less than 10 jams for the other causes each, so we cannot derive any valid distributions
# from that data.

######################################################################################################################
################################################## PART 2 ###########################################################
#####################################################################################################################

# from part 1 we know that there had been 1559 traffic jams on A2. This means that we take the rate of
# the compound Poisson process to be 1559.
# We take the durations of traffic jams as exponentially distributed, since the exponential distribution has the
# highest p-value of them all (and it looks well fitted on the histogram). lambda = 0.059221504266474434


data1 = pd.read_csv("2016-09_rws_filedata.csv", header=0, delimiter=';', decimal=',',
                    parse_dates=['DatumFileBegin', 'TijdFileBegin'])
data_jams = data1.loc[data1['RouteOms']== 'A2']
data2 = data_jams.sort_values(['DatumFileBegin', 'TijdFileBegin'], ascending=[True, True])
n = len(data2)

# list of datetime starting times of jams (unique)
trafficJamStart = list(dict.fromkeys([datetime.combine(datetime.date(data2['DatumFileBegin'].iloc[i]),
                                    datetime.time(data2['TijdFileBegin' ].iloc[i]))  for i in  range(n)]))

# making a Poisson process of starting times of the jams
times = list(filter(lambda a: a != 0, time_distances))
lam = 1 / (np.mean(times) / 60)

def simPoissonProcess(lam, t):    # lambda is a reserved word
    expDist = stats.expon(scale=1/lam)
    time = expDist.rvs()
    nT = 0
    while time < t:
        nT += 1
        time += expDist.rvs()
    return nT

# simulate N(t) for t=10 and lambda=lam
n = 10000  # number of runs
results = [simPoissonProcess(lam, 10) for _ in range(n)]
print(results)
print(np.mean(results))
print(np.var(results))
plt.figure()
plt.hist(results, bins=np.arange(-0.5, max(results)+0.5, 1), rwidth=0.8, density=True)
plt.title('Poisson process for all jams')
# theory: this is Poisson(2*10) distributed
x = np.arange(0, max(results))
plt.plot(x, stats.poisson(lam*10).pmf(x), 'go')
plt.show()
rate = lam

# bare minimum CPP

M1 = np.mean(list(filter(lambda a: a != 0, jam_durations)))
M2 = np.mean(np.power(jam_durations, 2))  # second moment
lam = 1 / M1
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)

def simulateCompoundPoissonProcess(lam, jumpDist, T):
    arrivalTimes = deque()  # the most efficient data structure
    levels = deque()  # the levels of the CPP
    currentLevel = 0
    expDist = stats.expon(scale=1 / lam)
    t = expDist.rvs()
    print(t)
    while t < T:
        print(t)
        arrivalTimes.append(t)
        currentLevel += jumpDist.rvs()
        levels.append(currentLevel)
        t += expDist.rvs()
    return arrivalTimes, levels


jumpdist = stats.expon(1 / lam)
T = 34  # number of minutes in September is 34560
data_simulation = simulateCompoundPoissonProcess(rate, jumpdist, 43200)
plt.figure()
plt.plot(data_simulation[0], data_simulation[1], marker='o', linestyle=None)
plt.title('Compound Poisson process of traffic jam durations for T = 1 month')
plt.ylabel('sum of time durations (minutes)')
plt.xlabel('T (minutes)')
plt.show()
print(data_simulation[1][-1])
print(sum(jam_durations))

# making a Poisson process of starting times of the Spitsfile

time_distances_spitsfile = []
date = ""
time = ""
for index, row in df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('Spitsfile')) & (
    df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).iterrows():
    if index != 0:
        if row['DatumFileBegin'] == date:
            time_distances_spitsfile.append((row['TijdFileBegin'] - time).total_seconds())
        else:
            time_distances_spitsfile.append(86400 - (time - row['TijdFileBegin']).total_seconds())
    date = row['DatumFileBegin']
    time = row['TijdFileBegin']

time_distances_spitsfile[0] = 0

times = list(filter(lambda a: a != 0, time_distances_spitsfile))
lam = 1 / (np.mean(times) / 60)

# simulate N(t) for t=10 and lambda=lam
n = 10000  # number of runs
results = [simPoissonProcess(lam, 10) for _ in range(n)]
print(results)
print(np.mean(results))
print(np.var(results))
plt.figure()
plt.hist(results, bins=np.arange(-0.5, max(results)+0.5, 1), rwidth=0.8, density=True)
plt.title('Poisson process for all Spitsfile')
# theory: this is Poisson(2*10) distributed
x = np.arange(0, max(results))
plt.plot(x, stats.poisson(lam*10).pmf(x), 'go')
plt.show()
rate = lam

# Spitsfile CPP

M1 = np.mean(list(filter(lambda a: a != 0, jam_durations_spitsfile)))
M2 = np.mean(np.power(jam_durations_spitsfile, 2))  # second moment
lam = 1 / M1
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)

def simulateCompoundPoissonProcess(lam, jumpDist, T):
    arrivalTimes = deque()  # the most efficient data structure
    levels = deque()  # the levels of the CPP
    currentLevel = 0
    expDist = stats.expon(scale=1 / lam)
    t = expDist.rvs()
    print(t)
    while t < T:
        print(t)
        arrivalTimes.append(t)
        currentLevel += jumpDist.rvs()
        levels.append(currentLevel)
        t += expDist.rvs()

    return arrivalTimes, levels

sum_s = 0
big_simulations = 1000
jumpdist = stats.expon(1 / lam)
for i in range(big_simulations):
    data_simulation = simulateCompoundPoissonProcess(rate, jumpdist, 43200)
    sum_s += data_simulation[1][-1]
#plt.figure()
# plt.plot(data_simulation[0], data_simulation[1], marker='o', linestyle=None)
# plt.title('Compound Poisson process of Spitsfile durations for T = 1 month')
# plt.ylabel('sum of time durations (minutes)')
# plt.xlabel('T (minutes)')
# plt.show()
rate_Spitsfile = rate
print('spitsfile' + str(sum_s / big_simulations))

# making a Poisson process of starting times of the File buiten spits

time_distances_file_buiten_spits = []
date = ""
time = ""
for index, row in df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('File buiten spits')) & (
    df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).iterrows():
    if index != 0:
        if row['DatumFileBegin'] == date:
            time_distances_file_buiten_spits.append((row['TijdFileBegin'] - time).total_seconds())
        else:
            time_distances_file_buiten_spits.append(86400 - (time - row['TijdFileBegin']).total_seconds())
    date = row['DatumFileBegin']
    time = row['TijdFileBegin']

time_distances_file_buiten_spits[0] = 0

times = list(filter(lambda a: a != 0, time_distances_file_buiten_spits))
lam = 1 / (np.mean(times) / 60)

# simulate N(t) for t=10 and lambda=lam
n = 10000  # number of runs
results = [simPoissonProcess(lam, 10) for _ in range(n)]
print(results)
print(np.mean(results))
print(np.var(results))
plt.figure()
plt.hist(results, bins=np.arange(-0.5, max(results)+0.5, 1), rwidth=0.8, density=True)
plt.title('Poisson process for all File buiten spits')
# theory: this is Poisson(2*10) distributed
x = np.arange(0, max(results))
plt.plot(x, stats.poisson(lam*10).pmf(x), 'go')
plt.show()
rate = lam

# File buiten spits CPP

M1 = np.mean(list(filter(lambda a: a != 0, jam_durations_filebuitenspits)))
M2 = np.mean(np.power(jam_durations_filebuitenspits, 2))  # second moment
lam = 1 / M1
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)


big_simulations = 1000
sum_s = 0
jumpdist = stats.expon(1 / lam)
for i in range(big_simulations):
    print('simulation ' + str(i))
    data_simulation = simulateCompoundPoissonProcess(rate, jumpdist, 43200)
    sum_s += data_simulation[1][-1]
# plt.figure()
# plt.plot(data_simulation[0], data_simulation[1], marker='o', linestyle=None)
# plt.title('Compound Poisson process of File buiten spits durations for T = 1 month')
# plt.ylabel('sum of time durations (minutes)')
# plt.xlabel('T (minutes)')
# plt.show()
rate_File_buiten_spits = rate
print(rate)
print('file buiten' + str(sum_s / big_simulations))

# making a Poisson process of starting times of the Ongeval jams

time_distances_ongeval = []
date = ""
time = ""
for index, row in df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('Ongeval')) & (
    df['OorzaakGronddetail'].str.contains('len'))].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).iterrows():
    if index != 0:
        if row['DatumFileBegin'] == date:
            time_distances_ongeval.append((row['TijdFileBegin'] - time).total_seconds())
        else:
            time_distances_ongeval.append(86400 - (time - row['TijdFileBegin']).total_seconds())
    date = row['DatumFileBegin']
    time = row['TijdFileBegin']

time_distances_ongeval[0] = 0

times = list(filter(lambda a: a != 0, time_distances_ongeval))
lam = 1 / (np.mean(times) / 60)

# simulate N(t) for t=10 and lambda=lam
n = 10000  # number of runs
results = [simPoissonProcess(lam, 10) for _ in range(n)]
print(results)
print(np.mean(results))
print(np.var(results))
plt.figure()
plt.hist(results, bins=np.arange(-0.5, max(results)+0.5, 1), rwidth=0.8, density=True)
plt.title('Poisson process for all Ongeval caused jams')
# theory: this is Poisson(2*10) distributed
x = np.arange(0, max(results))
plt.plot(x, stats.poisson(lam*10).pmf(x), 'go')
plt.show()
rate = lam

# Ongeval CPP

M1 = np.mean(list(filter(lambda a: a != 0, jam_durations_ongeval)))
M2 = np.mean(np.power(jam_durations_ongeval, 2))  # second moment
lam = 1 / M1
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)


jumpdist = stats.expon(1 / lam)
sum_s = 0
for i in range(1000):
    data_simulation = simulateCompoundPoissonProcess(rate, jumpdist, 43200)
    sum_s += data_simulation[1][-1]
# plt.figure()
# plt.plot(data_simulation[0], data_simulation[1], marker='o', linestyle=None)
# plt.title('Compound Poisson process of Ongeval jams durations for T = 1 month')
# plt.ylabel('sum of time durations (minutes)')
# plt.xlabel('T (minutes)')
# plt.show()
rate_ongeval = rate
print('ongeval' + str(sum_s/1000))

# time dependent poisson process for Spitsfile


# Spitsfile working days data
df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')].loc[
    (df['OorzaakGronddetail'].str.contains('Spitsfile')) & (
        df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].to_csv("sorted_A2_workdays_spitsfile.csv")

df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] == 'Saturday') | (df['weekday'] == 'Sunday')].loc[
    (df['OorzaakGronddetail'].str.contains('Spitsfile')) & (
        df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].to_csv("sorted_A2_weekend_spitsfile.csv")

# weekend data # THERE ARE NO SPITSFILE IN WEEKENDS


workingdays_time_distances_spitsfile = []
for index, row in df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')].loc[
    (df['OorzaakGronddetail'].str.contains('Spitsfile')) & (
            df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].iterrows():
    if index != 0:
        if row['DatumFileBegin'] == date:
            workingdays_time_distances_spitsfile.append((row['TijdFileBegin'] - time).total_seconds())
        else:
            workingdays_time_distances_spitsfile.append(
                86400 - time.total_seconds() + row['TijdFileBegin'].total_seconds())
    date = row['DatumFileBegin']
    time = row['TijdFileBegin']

workingdays_time_distances_spitsfile[0] = 0

# Cumulative time intervals plot for all jams

y = range(len(df.loc[df['RouteOms'] == 'A2'].loc[(df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')].loc[(df['OorzaakGronddetail'].str.contains('Spitsfile')) & (
    df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))]))
x = np.cumsum(np.array(workingdays_time_distances_spitsfile))
print(x)
plt.plot(x, y)
plt.title('Time intervals between working days Spitsfile')
plt.ylabel('Index of jam')
plt.xlabel('Time interval in seconds')
#plt.show()

# Make a histogram
plt.figure()  # create a new plot window
plt.hist(workingdays_time_distances_spitsfile, bins=int(sqrt(len(workingdays_time_distances_spitsfile))), rwidth=0.8, density=True)

# Fit a normal distribution

# First and second moment
M1 = np.mean(workingdays_time_distances_spitsfile)  # first moment
M2 = np.mean(np.power(workingdays_time_distances_spitsfile, 2))  # second moment

# Estimates for mu and sigma^2
mu = M1
sigma2 = M2 - M1 ** 2
fitNormDist = stats.norm(mu, sqrt(sigma2))

xs = np.arange(min(workingdays_time_distances_spitsfile), max(workingdays_time_distances_spitsfile), 10)

''' <- normal distribution against histogram
# Add theoretical density
ys = fitNormDist.pdf(xs)
plt.plot(xs, ys, color='red')
plt.title('Histogram against normal distribution fit (Spitsfile working days)')
plt.xlabel('length of time distance in seconds')
plt.ylabel('number of time distances')
plt.show()
'''

'''
# Fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.pdf(xs), color='g')
plt.title('Histogram against exponential distribution fit (Spitsfile working days)')
print(lam)
plt.show()
'''
'''<- gamma distribution against histogram
# Fit a gamma distribution
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
plt.plot(xs, fitGammaDist.pdf(xs), color='r')
plt.title('Histogram against gamma distribution fit (Spitsfile working days)')
plt.show()
'''

# plot the ECDF
ecdf = ECDF(workingdays_time_distances_spitsfile)
plt.figure()
plt.step(ecdf.x, ecdf.y, color='black', where='post', label='ECDF (Spitsfile working days)')
plt.plot(xs, fitNormDist.cdf(xs), color='b', label='normal distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(workingdays_time_distances_spitsfile, fitNormDist.cdf)
print('KS Test Normal distribution: ' + str(tst1))
# KstestResult(statistic=0.4206697775197583, pvalue=1.2887219342158755e-245)

# Shapiro-Wilk test for normality
tst2 = stats.shapiro(workingdays_time_distances_spitsfile)
print('Shapiro-Wilk Test: ' + str(tst2))
# ShapiroResult(statistic=0.20386534929275513, pvalue=0.0)


# fit a gamma distribution
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)
fitGammaDist = stats.gamma(alpha, scale=1 / beta)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', label='gamma distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(workingdays_time_distances_spitsfile, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst1))
# KstestResult(statistic=0.5660371492990007, pvalue=0.0)


# fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.cdf(xs), color='g', label='exponential distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(workingdays_time_distances_spitsfile, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
# KstestResult(statistic=0.44946657074694424, pvalue=3.39569863544941e-282)

# fit a uniform distribution
# (Careful: we are NOT using the method of moments here!!!!!)
a = min(workingdays_time_distances_spitsfile)
b = max(workingdays_time_distances_spitsfile)
fitUnifDist = stats.uniform(loc=a, scale=b - a)  # Careful! look at definition!
plt.plot(xs, fitUnifDist.cdf(xs), color='orange', label='uniform distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(workingdays_time_distances_spitsfile, fitUnifDist.cdf)
print('KS Test Uniform distribution: ' + str(tst1))
# KstestResult(statistic=0.9131967098203935, pvalue=0.0)

plt.title('Distribution fittings against ECDF (Spitsfile working days)')
plt.legend()
plt.xlabel('time intervals')
plt.show()

# time dependent poisson process for File buiten spits

# Spitsfile working days data
df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')].loc[
    (df['OorzaakGronddetail'].str.contains('File buiten spits')) & (
        df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].to_csv("sorted_A2_workdays_filebuiten.csv")

df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] == 'Saturday') | (df['weekday'] == 'Sunday')].loc[
    (df['OorzaakGronddetail'].str.contains('File buiten spits')) & (
        df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].to_csv("sorted_A2_weekend_filebuiten.csv")

# weekend data

weekend_time_distances_filebuiten = []
for index, row in df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] == 'Saturday') | (df['weekday'] == 'Sunday')].loc[
    (df['OorzaakGronddetail'].str.contains('File buiten spits')) & (
            df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].iterrows():
    if index != 0:
        if row['DatumFileBegin'] == date:
            weekend_time_distances_filebuiten.append((row['TijdFileBegin'] - time).total_seconds())
        else:
            weekend_time_distances_filebuiten.append(
                86400 - time.total_seconds() + row['TijdFileBegin'].total_seconds())
    date = row['DatumFileBegin']
    time = row['TijdFileBegin']

weekend_time_distances_filebuiten[0] = 0

# working days data
workingdays_time_distances_filebuiten = []
for index, row in df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')].loc[
    (df['OorzaakGronddetail'].str.contains('File buiten spits')) & (
            df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].iterrows():
    if index != 0:
        if row['DatumFileBegin'] == date:
            workingdays_time_distances_filebuiten.append((row['TijdFileBegin'] - time).total_seconds())
        else:
            workingdays_time_distances_filebuiten.append(
                86400 - time.total_seconds() + row['TijdFileBegin'].total_seconds())
    date = row['DatumFileBegin']
    time = row['TijdFileBegin']

workingdays_time_distances_filebuiten[0] = 0


# Cumulative time intervals plot for file buiten spits working days

y = range(len(df.loc[df['RouteOms'] == 'A2'].loc[(df['weekday'] == 'Saturday') | (df['weekday'] == 'Sunday')].loc[(df['OorzaakGronddetail'].str.contains('File buiten spits')) & (
    df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))]))
x = np.cumsum(np.array(weekend_time_distances_filebuiten))
print(x)
plt.plot(x, y)
plt.title('Time intervals between weekend File buiten spits')
plt.ylabel('Index of jam')
plt.xlabel('Time interval in seconds')
plt.show()


# Make a histogram
plt.figure()  # create a new plot window
plt.hist(weekend_time_distances_filebuiten, bins=int(sqrt(len(weekend_time_distances_filebuiten))), rwidth=0.8,
         density=True)

# Fit a normal distribution

# First and second moment
M1 = np.mean(weekend_time_distances_filebuiten)  # first moment
M2 = np.mean(np.power(weekend_time_distances_filebuiten, 2))  # second moment

# Estimates for mu and sigma^2
mu = M1
sigma2 = M2 - M1 ** 2
fitNormDist = stats.norm(mu, sqrt(sigma2))

xs = np.arange(min(weekend_time_distances_filebuiten), max(weekend_time_distances_filebuiten), 10)

'''
# <- normal distribution against histogram
# Add theoretical density
ys = fitNormDist.pdf(xs)
plt.plot(xs, ys, color='red')
plt.title('Histogram against normal distribution fit (File buiten spits weekend)')
plt.xlabel('length of time distance in seconds')
plt.ylabel('number of time distances')
plt.show()
'''

'''
# Fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.pdf(xs), color='g')
plt.title('Histogram against exponential distribution fit (File buiten spits weekend)')
print(lam)
plt.show()
'''
'''
#<- gamma distribution against histogram
# Fit a gamma distribution
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
plt.plot(xs, fitGammaDist.pdf(xs), color='r')
plt.title('Histogram against gamma distribution fit (File buiten spits weekend)')
plt.show()
print(alpha)
print(beta)
'''

# plot the ECDF
ecdf = ECDF(weekend_time_distances_filebuiten)
plt.figure()
plt.step(ecdf.x, ecdf.y, color='black', where='post', label='ECDF (File buiten spits weekend)')
plt.plot(xs, fitNormDist.cdf(xs), color='b', label='normal distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(weekend_time_distances_filebuiten, fitNormDist.cdf)
print('KS Test Normal distribution: ' + str(tst1))
# KstestResult(statistic=0.4206697775197583, pvalue=1.2887219342158755e-245)

# Shapiro-Wilk test for normality
tst2 = stats.shapiro(weekend_time_distances_filebuiten)
print('Shapiro-Wilk Test: ' + str(tst2))
# ShapiroResult(statistic=0.20386534929275513, pvalue=0.0)


# fit a gamma distribution
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)
fitGammaDist = stats.gamma(alpha, scale=1 / beta)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', label='gamma distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(weekend_time_distances_filebuiten, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst1))
# KstestResult(statistic=0.5660371492990007, pvalue=0.0)


# fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.cdf(xs), color='g', label='exponential distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(weekend_time_distances_filebuiten, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
# KstestResult(statistic=0.44946657074694424, pvalue=3.39569863544941e-282)

# fit a uniform distribution
# (Careful: we are NOT using the method of moments here!!!!!)
a = min(weekend_time_distances_filebuiten)
b = max(weekend_time_distances_filebuiten)
fitUnifDist = stats.uniform(loc=a, scale=b - a)  # Careful! look at definition!
plt.plot(xs, fitUnifDist.cdf(xs), color='orange', label='uniform distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(weekend_time_distances_filebuiten, fitUnifDist.cdf)
print('KS Test Uniform distribution: ' + str(tst1))
# KstestResult(statistic=0.9131967098203935, pvalue=0.0)

plt.title('Distribution fittings against ECDF (File buiten weekend)')
plt.legend()
plt.xlabel('time intervals')
plt.show()

# time dependent poisson process for Ongeval

# Spitsfile working days data
df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')].loc[
    (df['OorzaakGronddetail'].str.contains('Ongeval')) & (
        df['OorzaakGronddetail'].str.contains('len'))].to_csv("sorted_A2_workdays_ongeval.csv")

df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] == 'Saturday') | (df['weekday'] == 'Sunday')].loc[
    (df['OorzaakGronddetail'].str.contains('Ongeval')) & (
        df['OorzaakGronddetail'].str.contains('len'))].to_csv("sorted_A2_weekend_ongeval.csv")

# weekend data

weekend_time_distances_ongeval = []
for index, row in df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] == 'Saturday') | (df['weekday'] == 'Sunday')].loc[
    (df['OorzaakGronddetail'].str.contains('Ongeval')) & (
            df['OorzaakGronddetail'].str.contains('len'))].iterrows():
    if index != 0:
        if row['DatumFileBegin'] == date:
            weekend_time_distances_ongeval.append((row['TijdFileBegin'] - time).total_seconds())
        else:
            weekend_time_distances_ongeval.append(86400 - time.total_seconds() + row['TijdFileBegin'].total_seconds())
    date = row['DatumFileBegin']
    time = row['TijdFileBegin']

weekend_time_distances_filebuiten[0] = 0

# working days data
workingdays_time_distances_ongeval = []
for index, row in df.loc[df['RouteOms'] == 'A2'].sort_values(by=['DatumFileBegin', 'TijdFileBegin']).loc[
    (df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')].loc[
    (df['OorzaakGronddetail'].str.contains('Ongeval')) & (
            df['OorzaakGronddetail'].str.contains('len'))].iterrows():
    if index != 0:
        if row['DatumFileBegin'] == date:
            workingdays_time_distances_ongeval.append((row['TijdFileBegin'] - time).total_seconds())
        else:
            workingdays_time_distances_ongeval.append(
                86400 - time.total_seconds() + row['TijdFileBegin'].total_seconds())
    date = row['DatumFileBegin']
    time = row['TijdFileBegin']

workingdays_time_distances_ongeval[0] = 0


# Cumulative time intervals plot for ongeval spits working days

y = range(len(df.loc[df['RouteOms'] == 'A2'].loc[(df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')].loc[(df['OorzaakGronddetail'].str.contains('Ongeval')) & (
    df['OorzaakGronddetail'].str.contains('len'))]))
x = np.cumsum(np.array(workingdays_time_distances_ongeval))
print(x)
plt.plot(x, y)
plt.title('Time intervals between working days Ongeval caused jams')
plt.ylabel('Index of jam')
plt.xlabel('Time interval in seconds')
plt.show()


# Make a histogram
plt.figure()  # create a new plot window
plt.hist(workingdays_time_distances_ongeval, bins=int(sqrt(len(workingdays_time_distances_ongeval))), rwidth=0.8,
         density=True)

# Fit a normal distribution

# First and second moment
M1 = np.mean(workingdays_time_distances_ongeval)  # first moment
M2 = np.mean(np.power(workingdays_time_distances_ongeval, 2))  # second moment

# Estimates for mu and sigma^2
mu = M1
sigma2 = M2 - M1 ** 2
fitNormDist = stats.norm(mu, sqrt(sigma2))

xs = np.arange(min(workingdays_time_distances_ongeval), max(workingdays_time_distances_ongeval), 10)


# <- normal distribution against histogram
# Add theoretical density
ys = fitNormDist.pdf(xs)
plt.plot(xs, ys, color='red')
plt.title('Histogram against normal distribution fit (Ongeval caused jams workdays)')
plt.xlabel('length of time distance in seconds')
plt.ylabel('number of time distances')
plt.show()
'''

# Fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.pdf(xs), color='g')
plt.title('Histogram against exponential distribution fit (Ongeval workdays)')
print(lam)
plt.show()

'''
#<- gamma distribution against histogram
# Fit a gamma distribution
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
plt.plot(xs, fitGammaDist.pdf(xs), color='r')
plt.title('Histogram against gamma distribution fit (Ongeval workdays)')
plt.show()
print(alpha)
print(beta)


# plot the ECDF
ecdf = ECDF(workingdays_time_distances_ongeval)
plt.figure()
plt.step(ecdf.x, ecdf.y, color='black', where='post', label='ECDF (Ongeval workdays)')
plt.plot(xs, fitNormDist.cdf(xs), color='b', label='normal distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(workingdays_time_distances_ongeval, fitNormDist.cdf)
print('KS Test Normal distribution: ' + str(tst1))
# KstestResult(statistic=0.4206697775197583, pvalue=1.2887219342158755e-245)

# Shapiro-Wilk test for normality
tst2 = stats.shapiro(workingdays_time_distances_ongeval)
print('Shapiro-Wilk Test: ' + str(tst2))
# ShapiroResult(statistic=0.20386534929275513, pvalue=0.0)


# fit a gamma distribution
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)
fitGammaDist = stats.gamma(alpha, scale=1 / beta)
plt.plot(xs, fitGammaDist.cdf(xs), color='r', label='gamma distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(workingdays_time_distances_ongeval, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst1))
# KstestResult(statistic=0.5660371492990007, pvalue=0.0)


# fit an exponential distribution
lam = 1 / M1
fitExpDist = stats.expon(scale=1 / lam)
plt.plot(xs, fitExpDist.cdf(xs), color='g', label='exponential distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(workingdays_time_distances_ongeval, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))
# KstestResult(statistic=0.44946657074694424, pvalue=3.39569863544941e-282)

# fit a uniform distribution
# (Careful: we are NOT using the method of moments here!!!!!)
a = min(workingdays_time_distances_ongeval)
b = max(workingdays_time_distances_ongeval)
fitUnifDist = stats.uniform(loc=a, scale=b - a)  # Careful! look at definition!
plt.plot(xs, fitUnifDist.cdf(xs), color='orange', label='uniform distribution')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(workingdays_time_distances_ongeval, fitUnifDist.cdf)
print('KS Test Uniform distribution: ' + str(tst1))
# KstestResult(statistic=0.9131967098203935, pvalue=0.0)

plt.title('Distribution fittings against ECDF (Ongeval workdays)')
plt.legend()
plt.xlabel('time intervals')
plt.show()

# we merge separate Poisson processes with different rates

# workdays merge (and conversion to minutes)
workingdays_times_distances_spitsfile = list(filter(lambda a: a != 0, workingdays_time_distances_spitsfile))
workingdays_times_distances_filebuiten = list(filter(lambda a: a != 0, workingdays_time_distances_filebuiten))
workingdays_times_distances_ongeval = list(filter(lambda a: a != 0, workingdays_time_distances_ongeval))
weekend_times_distances_filebuiten = list(filter(lambda a: a != 0, weekend_time_distances_filebuiten))
weekend_times_distances_ongeval = list(filter(lambda a: a != 0, weekend_time_distances_ongeval))

M1_spitsfile_workdays = np.mean(workingdays_times_distances_spitsfile) / 60
M2_spitsfile_workdays = np.mean(np.power(np.true_divide(workingdays_times_distances_spitsfile, 60), 2))
lam_spitsfile_workdays = 1 / M1
beta_spitsfile_workdays = M1_spitsfile_workdays / (M2_spitsfile_workdays - M1_spitsfile_workdays ** 2)

M1_filebuiten_workdays = np.mean(workingdays_times_distances_filebuiten) / 60
M2_filebuiten_workdays = np.mean(np.power(np.true_divide(workingdays_times_distances_filebuiten, 60), 2))
lam_filebuiten_workdays = 1 / M1
beta_filebuiten_workdays = M1_filebuiten_workdays / (M2_filebuiten_workdays - M1_filebuiten_workdays ** 2)

M1_ongeval_workdays = np.mean(workingdays_times_distances_ongeval) / 60
M2_ongeval_workdays = np.mean(np.power(np.true_divide(workingdays_times_distances_ongeval, 60), 2))
lam_ongeval_workdays = 1 / M1
beta_ongeval_workdays = M1_ongeval_workdays / (M2_ongeval_workdays - M1_ongeval_workdays ** 2)

M1_filebuiten_weekend = np.mean(weekend_times_distances_filebuiten) / 60
M2_filebuiten_weekend = np.mean(np.power(np.true_divide(weekend_times_distances_filebuiten, 60), 2))
lam_filebuiten_weekend = 1 / M1
beta_filebuiten_weekend = M1_filebuiten_weekend / (M2_filebuiten_weekend - M1_filebuiten_weekend ** 2)

M1_ongeval_weekend = np.mean(weekend_times_distances_ongeval) / 60
M2_ongeval_weekend = np.mean(np.power(np.true_divide(weekend_times_distances_ongeval, 60), 2))
lam_ongeval_weekend = 1 / M1
beta_ongeval_weekend = M1_ongeval_weekend / (M2_ongeval_weekend - M1_ongeval_weekend ** 2)

jam_durations_filebuitenspits_weekend = \
    df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('File buiten spits')) & (
        df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].loc[
        (df['weekday'] == 'Saturday') | (df['weekday'] == 'Sunday')]['FileDuur'].to_list()

jam_durations_ongeval_weekend = \
    df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('Ongeval')) & (
        df['OorzaakGronddetail'].str.contains('len'))].loc[
        (df['weekday'] == 'Saturday') | (df['weekday'] == 'Sunday')]['FileDuur'].to_list()

jam_durations_filebuitenspits_workdays = \
    df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('File buiten spits')) & (
        df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))].loc[
        (df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')]['FileDuur'].to_list()

jam_durations_ongeval_workdays = \
    df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('Ongeval')) & (
        df['OorzaakGronddetail'].str.contains('len'))].loc[
        (df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')]['FileDuur'].to_list()

# CPP for working days File buiten spits

time_distances_spitsfile = []
date = ""
time = ""
for index, row in df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('Ongeval')) & (
        df['OorzaakGronddetail'].str.contains('len'))].loc[
    (df['weekday'] != 'Saturday') & (df['weekday'] != 'Sunday')].sort_values(
    by=['DatumFileBegin', 'TijdFileBegin']).iterrows():
    if index != 0:
        if row['DatumFileBegin'] == date:
            time_distances_spitsfile.append((row['TijdFileBegin'] - time).total_seconds())
        else:
            time_distances_spitsfile.append(86400 - (time - row['TijdFileBegin']).total_seconds())
    date = row['DatumFileBegin']
    time = row['TijdFileBegin']

time_distances_spitsfile[0] = 0

times = list(filter(lambda a: a != 0, time_distances_spitsfile))
lam = 1 / (np.mean(times) / 60)
rate = lam


# Spitsfile CPP

def simulateCompoundPoissonProcess(lam, jumpDist, T):
    arrivalTimes = deque()  # the most efficient data structure
    levels = deque()  # the levels of the CPP
    currentLevel = 0
    expDist = stats.expon(scale=1 / lam)
    t = expDist.rvs()
    print(t)
    while t < T:
        print(t)
        arrivalTimes.append(t)
        currentLevel += jumpDist.rvs()
        levels.append(currentLevel)
        t += expDist.rvs()
    return arrivalTimes, levels


M1 = np.mean(list(filter(lambda a: a != 0, jam_durations_ongeval_workdays)))
M2 = np.mean(np.power(jam_durations_ongeval_workdays, 2))  # second moment
lam = 1 / M1
alpha = M1 ** 2 / (M2 - M1 ** 2)
beta = M1 / (M2 - M1 ** 2)
mu = M1
sigma2 = M2 - M1 ** 2


def simulateCompoundPoissonProcess(lam, jumpDist, T):
    arrivalTimes = deque()  # the most efficient data structure
    levels = deque()  # the levels of the CPP
    currentLevel = 0
    expDist = stats.expon(scale=1 / lam)
    t = expDist.rvs()
    print(t)
    while t < T:
        print(t)
        arrivalTimes.append(t)
        currentLevel += jumpDist.rvs()
        levels.append(currentLevel)
        t += expDist.rvs()

    return arrivalTimes, levels


sum_s = 0
big_simulations = 1000
# jumpdist = stats.norm(mu, sqrt(sigma2))
jumpdist = stats.gamma(alpha, 1 / beta)
# jumpdist = stats.expon(1/lam)
data_for_confidence = []
data_sums = []
for i in range(1):
    sum_s = 0
    for i in range(big_simulations):
        data_simulation = simulateCompoundPoissonProcess(rate, jumpdist, 30240)
        sum_s += data_simulation[1][-1]
        data_for_confidence.append(data_simulation[1][-1])
    data_sums.append(sum_s / big_simulations)
# plt.figure()
# plt.plot(data_simulation[0], data_simulation[1], marker='o', linestyle=None)
# plt.title('Compound Poisson process of Spitsfile for T = 1 month')
# plt.ylabel('sum of time durations (minutes)')
# plt.xlabel('T (minutes)')
# plt.show()

s2 = np.var(data_for_confidence)
m = np.mean(data_for_confidence)
print("s2=" + str(s2))
print("m=" + str(m))
z = 1.96
n = (z ** 2) * s2 / (50 * 50)
print("n=", n)

halfWidth = z * sqrt(s2 / (int(n) + 1))

interval = (m - halfWidth, m + halfWidth)
print(interval)

print('File buiten weekend gamma: ' + str(np.mean(data_sums)))

# final pie charts

fig = plt.figure(figsize=[10, 10])
ax = fig.add_axes([0, 0, 1, 1])
causes = 'Rush hour', 'Non rush hour weekend', 'Non rush hour workdays', 'Accident weekend', 'Accident workdays'
sizes = [20903.36, 144.844, 997.49, 292.81, 1457.63]
explode = (0.1, 0, 0, 0, 0)
values = [[21807, 20903.36], [170.97, 144.844], [1295.65, 997.49], [240.82, 292.81], [750.18, 1457.63]]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=causes, labeldistance=1.05, textprops={'fontsize': 7}, autopct='%1.1f%%',
        shadow=False)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Data statistics')
plt.show()
print(df.loc[df['RouteOms'] == 'A2']['weekday'].value_counts())
print(df.loc[df['RouteOms'] == 'A2'].loc[(df['OorzaakGronddetail'].str.contains('Spitsfile')) & (
    df['OorzaakGronddetail'].str.contains('geen oorzaak gemeld'))]['weekday'].value_counts())