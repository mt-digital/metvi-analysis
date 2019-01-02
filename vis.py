import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from correlate_tweets import data_for_model
from datetime import datetime

from collections import Counter

def subjobj_barchart(full_df, by='month', save_path=None):

    # First, aggregate over the groupby column.
    agg_dict = {a: sum for a in ['RepSubj', 'DemSubj', 'RepObj', 'DemObj']}
    gb = full_df.groupby(by).agg(agg_dict)

    # We need to normalize by the number of episodes aggregated over
    # for each group (either networks or months).
    by_vec = full_df[by]
    by_counts = Counter(by_vec)

    fig, ax = plt.subplots(figsize=(6.25, 3.75))

    if by == 'month':
        gb.loc['September'] = gb.loc['September'] / by_counts['September']
        gb.loc['October'] = gb.loc['October'] / by_counts['October']
        gb.loc['November'] = gb.loc['November'] / by_counts['November']
        gb = gb.loc[['September', 'October', 'November']]
        color = ['k', 'r', 'w']

        ax.set_ylim(0, .7)

    elif by == 'network':
        gb.loc['MSNBCW'] = gb.loc['MSNBCW'] / by_counts['MSNBCW']
        gb.loc['CNNW'] = gb.loc['CNNW'] / by_counts['CNNW']
        gb.loc['FOXNEWSW'] = gb.loc['FOXNEWSW'] / by_counts['FOXNEWSW']
        gb = gb.loc[['MSNBCW', 'CNNW', 'FOXNEWSW']]
        color = ['w', 'k', 'r']
        # ax.set_ylim(0, .7)


    # import ipdb
    # ipdb.set_trace()
    gb.transpose().plot(
        kind='bar', color=color, ec='k', ax=ax, grid='on', lw=1, zorder=3
    )
    # Turn off vertical grid lines.
    ax.grid(axis='x')

    ax.set_ylabel('Metaphor uses per episode', size=14.5)
    ax.set_xlabel('\nCandidate party and grammatical type', size=15)
    print(list(ax.get_yticklabels()))
    ax.set_xticklabels(
        ['Republican \nas Subject', 'Democrat \nas Subject',
         'Republican \nas Object', 'Democrat \nas Object'],
        size=14,
        rotation=0
    )

    # For some reason can't get_yticklabels as I would expect/normally do.
    ax.set_yticklabels(['0.' + str(i) for i in range(8)], size=12)

    ax.get_legend().set_title(by.title(), prop={'size': 13})

    if save_path is not None:
        if save_path[-4:] == '.png':
            plt.savefig(save_path, dpi=300)
        else:
            plt.savefig(save_path)

    return ax





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
