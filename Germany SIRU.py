import pandas as pd
df = pd.read_csv('covid_19_data.csv')


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = df.loc[df['Country/Region'] == "Germany"]

df.iloc[0]

len(df['Confirmed'].values )


y=df['Confirmed'].values[0:94]

germany=plt.figure()
plt.rcParams.update({'font.size':11})
plt.plot(y, label="Actual Cases")
plt.xlabel("Time in days")
plt.ylabel("Number of people")
plt.title("Actual Cumulated Confirmed Cases")
germany.show()

germany.savefig('germany.png')
#files.download('germany.png')

def fit(x, a, b, c):
  #if a <= 0.0 or b<=0.0 or c<=0.0:return 1.0E10
  return a * np.exp(b * x) + c


#45 is optimal for south korea
y=df['Confirmed'].values[0:50]
x=np.arange(1,len(y)+1)

popt, pcov = curve_fit(fit, x, y)


germany50=plt.figure()
plt.plot(x, fit(x, *popt), label="Fitted Curve")
plt.plot(y, label="Actual Cases")
plt.legend(loc="upper left")

plt.title("Fitting curve to Cumulated Confirmed Cases")
plt.xlabel('Time in days')
plt.ylabel('Number of people')
germany50.show()

germany50.savefig('ngermany50.pdf')


v = 1/7
n = 1/7
f0 = 0.8
X1 = popt[0]
X2 = popt[1]
X3 = -1 * popt[2]


# Without public health intervention

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#S0 =           # Total population is susceptible initially
#I0 =                # Initial number of infected individuals, I0
R0 = 0              # Initial number of reported infectious individuals, R0.
#U0 =               # Initial number of unreported infectious individuals, U0.
CR0 = 0             # Cumulative reported infectious individuals at time t0, CR0.

#v =               # ......  rate
#n =             # Mean ...... rate (1/days)

Actual = df['Confirmed'].values            # Actual from data

t = np.linspace(0, 150, 150)      # A grid of time points (in days)
# linspace ( start, stop, number of entries in interval)

# The SIRU model differential equations.
def deriv(y, t, tao0, v, n, f0 ):
    S, I, R, U, CR = y
    dSdt = -tao0 * S * (I + U)                   # It means dS/dt
    dIdt =  tao0 * S * (I + U) - v * I
    dRdt = v * f0 * I - n * R
    dUdt = v * (1-f0) * I - n * U
    dCRdt = v * f0 * I
    return dSdt, dIdt, dRdt, dUdt, dCRdt

y0 = S0, I0, R0, U0, CR0                      # Initial conditions vector
# Integrate the SIR equations over the time grid, t.
# odeiRnt (model that returns dy/dt, y0, t)
ret = odeint(deriv, y0, t, args=(tao0, v, n, f0))
S, I, R, U, CR = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')

ax = fig.add_subplot(111, axisbelow=True)

#import pandas as pd
#date1 = '2019-11-01'
#date2 = '2020-04-08'
#t1 = pd.date_range(date1, date2).tolist()

# ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
# ax.plot(t, I 'y', alpha=0.5, lw=2, label='Asymptomatic Infected')
ax.plot(t, R, 'r', alpha=0.5, lw=2, label='Reported Symptomatic Infected')
ax.plot(t, U, 'g', alpha=0.5, lw=2, label='Uneported Symptomatic Infected')
ax.plot(t, CR, 'y', alpha=0.5, lw=2, label='Cummulative Reported Cases')
#ax.plot(t, Actual, 'b', alpha=0.5, lw=2, label='Actual Cummulative Reported Cases')

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}".format(xmax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

annot_max(t,R)

ax.set_xlabel('Time in days')
ax.set_ylabel('Number of people')

#ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)

ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)

plt.title("SIRU: Germany")
plt.show()

fig.savefig('SIRUwoMeasures.png')




## With Measures

## time t till no measures

y0 = S0, I0, R0, U0, CR0                      # Initial conditions vector
t= np.linspace(0,84,84)   
# Integrate the SIR equations over the time grid, t.
# odeiRnt (model that returns dy/dt, y0, t)
ret = odeint(deriv, y0, t, args=(tao0, v, n, f0))
S, I, R, U, CR = ret.T


y10 = S[-1], I[-1], R[-1], U[-1], CR[-1]
tao0=0
f0=0.9
t = np.linspace(0, 86, 86) 
ret1 = odeint(deriv, y10, t, args=(tao0, v, n, f0))
S1, I1, R1, U1, CR1 = ret1.T

#append S,I,R,U,CR with S1,I1,R1,U1,CR1
s = np.concatenate((S, S1))
i = np.concatenate((I, I1))
r = np.concatenate((R, R1))
u = np.concatenate((U, U1))
cr = np.concatenate((CR, CR1))

# t: added length
t = np.linspace(0, 170, 170)  

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')

ax = fig.add_subplot(111, axisbelow=True)

#ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
# ax.plot(t, I 'y', alpha=0.5, lw=2, label='Asymptomatic Infected')
ax.plot(t, r, 'r', alpha=0.5, lw=2, label='Reported Symptomatic Infected')
ax.plot(t, u, 'g', alpha=0.5, lw=2, label='Uneported Symptomatic Infected')
ax.plot(t, cr, 'y', alpha=0.5, lw=2, label='Cummulative Reported Cases')
#ax.plot(t, Actual, 'g', alpha=0.5, lw=2, label='Actual Cummulative Reported Cases')

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}".format(xmax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

annot_max(t,r)

ax.set_xlabel('Time in days')
ax.set_ylabel('Number of people')

#ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)

ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)

plt.title("SIRU with public health measures: Germany")
plt.rcParams.update({'font.size':11})
plt.show()

fig.savefig('SIRUwMeasures.png')

