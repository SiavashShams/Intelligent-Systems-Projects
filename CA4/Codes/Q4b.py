from scipy.stats import expon, binom
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# make the exponential distribution array
n = 1000
data_expon = expon.rvs(scale=0.5, loc=0, size=n)
ax = sns.displot(data_expon,
                 kde=True,
                 bins=100,
                 color='blue')
ax.set(xlabel='Exponential Distribution', ylabel='Frequency')
plt.show()
# take samples
s = 1000
samples = []
for i in range(s):
    samples.append(np.random.choice(data_expon, size=10, replace=False))
# plot mean of samples
samples = np.array(samples)
dx = samples.mean(axis=1)
plt.hist(dx, bins=30)
plt.show()
# mean and std of samples
mean = samples.mean()
std = np.std(samples)
print("Mean of the samples is equal to: ", mean)
print("Standard deviation of the samples is equal to: ", std)

n = 1000

# do the same for binomial distribution
data_binom = binom.rvs(20, 0.8, size=n)
ax = sns.displot(data_binom,
                 kde=False,
                 bins=100,
                 color='blue')
ax.set(xlabel='Binomial Distribution', ylabel='Frequency')
plt.show()
s = 1000
samples = []
for i in range(s):
    samples.append(np.random.choice(data_binom, size=10, replace=False))

samples = np.array(samples)
dx = samples.mean(axis=1)
plt.hist(dx, bins=30)
plt.show()

mean = samples.mean()
std = np.std(samples)
print("Mean of the samples is equal to: ", mean)
print("Standard deviation of the samples is equal to: ", std)
