from matplotlib import pyplot as plt
from tueplots import bundles
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate some random data
# np.random.seed(42)
# x = np.random.rand(100, 1) * 10  # Random values between 0 and 10
# y = 2.5 * x + np.random.randn(100, 1) * 3  # Linear relation with noise

name_exp = 'goals_18'
folder = 'maze'
y = np.loadtxt(f"/home/nuria/phd/controllable_agent/figs/correl/maze/ef_{name_exp}.csv", delimiter=",").reshape(-1,1)
x = np.loadtxt(f"/home/nuria/phd/controllable_agent/figs/correl/maze/eq_{name_exp}.csv", delimiter=",").reshape(-1,1)

# name_exp = 'tr_mean'
# folder = 'walker'
# y = np.loadtxt(f"/home/nuria/phd/controllable_agent/figs/correl/walker/ef_{name_exp}.csv", delimiter=",").reshape(-1,1)
# x = np.loadtxt(f"/home/nuria/phd/controllable_agent/figs/correl/walker/eq_{name_exp}.csv", delimiter=",").reshape(-1,1)

# x_outl = x[x<400]
# y_outl = y[x<400]

# x = x_outl[y_outl<40000]
# y = y_outl[y_outl<40000]

# STANDARDIZE
x = (x - np.mean(x))/np.std(x)
y = (y - np.mean(y))/np.std(y)

if len(x.shape) < 2:
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

plt.rcParams.update(bundles.iclr2024(
    family="serif", rel_width=0.9, nrows=1.0, ncols=1.0, usetex=False))

ourblue = (0.368, 0.507, 0.71)
ourdarkblue = (0.368, 0.607, 0.9)

# Fit a linear regression model
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Plot the scatter plot and regression line
plt.scatter(x, y, color=ourblue, alpha=0.5, label='Data')
plt.plot(x, y_pred, color=ourdarkblue, linewidth=2, label='Regression Line')
plt.xlabel("Var[Q]")
# plt.ylabel("Det(Covar[F])")
plt.ylabel("tr(Covar[F])")
# plt.show()
dir_figs = f'/home/nuria/phd/controllable_agent/figs/correl/{folder}'
name_fig = f'correl_{name_exp}_standardized'
plt.savefig(f'{dir_figs}/{name_fig}.pdf', bbox_inches='tight')