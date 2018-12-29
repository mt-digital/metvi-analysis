import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from correlate_tweets import data_for_model
from datetime import datetime


def fits_plots(tsdata, figsize=(8, 3), logdays=False):
    fig, axes = plt.subplots(1, 3, figsize=figsize)  # , sharey=True)

    for idx, col in enumerate(['days_from_debate', 'clinton', 'trump']):
        data = tsdata.copy()
        if idx == 0:
            data['metvi_all'] = np.log(data['metvi_all'])
        formula = 'metvi_all ~ ' + col
        mod = smf.ols(
            formula=formula,
            data=data
        )
        res = mod.fit()
        intercept = res.params['Intercept']
        coeff = res.params[col]
            
        x = data[col]
        y = data['metvi_all']
        
        x0 = x.min()
        x1 = x.max()
        
        y0 = intercept + (coeff * x0)
        y1 = intercept + (coeff * x1)
        
        ax = axes[idx]
        ax.plot(x, y, '.')
        ax.plot([x0, x1], [y0, y1], color='r')
#         ax.set_ylim(0, 12)
#         ax.set_yticks([0, 4, 8, 12])

        if idx == 0:
            ax.set_ylabel('log(MV frequency)')
        else:
            ax.set_ylabel('MV frequency')

        if col == 'days_from_debate':
            ax.set_xlim(-2.5, 44)
            ax.set_xlabel('Days from a debate', size=12)
        else:
            ax.set_xlim(-5, 105)
            if col == 'clinton':
                ax.set_xlabel('@HillaryClinton tweets', size=12)
            elif col == 'trump':
                ax.set_xlabel('@realDonaldTrump tweets', size=12)

        ax.grid()


def fits_all_plot_predictions(tsdata, dependent='metvi_all', showdata=True, showpred=True):

    mod = smf.ols(
        formula=dependent + ' ~ days_from_debate + clinton + trump',
        data=tsdata
    )

    res = mod.fit()

    # print(res.predict({'days_from_debate': 2, 'clinton': 5, 'trump': 10}))
    # print(res.predict(tsdata[['days_from_debate', 'clinton', 'trump']]))

    pred_X = tsdata[['days_from_debate', 'clinton', 'trump']]
    pred_y = res.predict(pred_X)

    if showdata:
        tsdata[dependent].plot(style='o', mfc='white', mew=1.5, ms=8, label='Actual')
    if showpred:
        pred_y.plot(mec='red', mfc='red', style='o', ms=6, label='Predicted', alpha=0.8)
    plt.legend()
    yheight = 0.1
    plt.axvline(
        datetime(2016, 9, 26), ymax=yheight, color='k' #, zorder=zo
    )
    plt.axvline(
        datetime(2016, 10, 9), ymax=yheight, color='k' #, zorder=zo
    )
    plt.axvline(
        datetime(2016, 10, 19), ymax=yheight, color='k' #, zorder=zo
    )
    textargs = dict(
        size=13,
        ha='right',
        bbox=dict(alpha=0.8, color='white')
    )
    plt.text('2016-9-24', 0.3, "Debate #1", **textargs)
    plt.text('2016-10-7', 0.3, "#2", **textargs)
    plt.text('2016-10-17', 0.3, "#3", **textargs)
    plt.grid(axis='y')

    plt.xlabel('Date')
    plt.ylabel('MV Frequency')

    print(res.summary())
