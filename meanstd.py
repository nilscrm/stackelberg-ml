from math import sqrt

means = [15.940, 10.380, 15.340, 20.520, 25.945]
stds = [12.014, 12.130, 10.672, 14.578, 9.434]
nn = 10
ns = [nn]*5

N = sum(ns)
mean = sum([m*n for m,n in zip(means, ns)]) / N
std = sqrt( sum([n*(s**2) + m**2 for n,s,m in zip(ns, stds, means)]) / N - mean )


print(mean, std)