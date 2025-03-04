from matplotlib import pyplot as plt
from tueplots import bundles
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate some random data
# np.random.seed(42)
# x = np.random.rand(100, 1) * 10  # Random values between 0 and 10
# y = 2.5 * x + np.random.randn(100, 1) * 3  # Linear relation with noise

name_exp = 'goals_18_det'
y = np.loadtxt(f"/home/nuria/phd/controllable_agent/figs/correl/ef_{name_exp}.csv", delimiter=",").reshape(-1,1)
x = np.loadtxt(f"/home/nuria/phd/controllable_agent/figs/correl/eq_{name_exp}.csv", delimiter=",").reshape(-1,1)


plt.rcParams.update(bundles.iclr2024(
    family="serif", rel_width=0.9, nrows=1.0, ncols=1.0, usetex=False))

ourblue = (0.368, 0.507, 0.71)
ourdarkblue = (0.368, 0.607, 0.9)
ourorange = (0.881, 0.611, 0.142)
ourgreen = (0.56, 0.692, 0.195)
ourred = (0.923, 0.386, 0.209)
ourviolet = (0.528, 0.471, 0.701)
ourbrown = (0.772, 0.432, 0.102)
ourlightblue = (0.364, 0.619, 0.782)
ourdarkgreen = (0.572, 0.586, 0.0)
ourdarkred = (0.923, 0.386, 0.)

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
dir_figs = '/home/nuria/phd/controllable_agent/figs/correl'
name_fig = f'correl_{name_exp}'
plt.savefig(f'{dir_figs}/{name_fig}.pdf', bbox_inches='tight')