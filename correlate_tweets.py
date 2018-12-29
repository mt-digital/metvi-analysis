'''
Testing hypothesis that metaphorical violence usage in 2016 is correlated with
Donald Trump's tweeting.
'''
import datetime
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import Counter
from datetime import datetime as DATETIME
from os.path import join as osjoin
from scipy.stats import pearsonr, linregress

os.environ['CONFIG_FILE'] = 'conf/default.cfg'

from projects.common import get_project_data_frame
from projects.common.analysis import daily_frequency


METAPHORS_URL_TEMPLATE = \
    'http://metacorps.io/static/data/viomet-sep-nov-{}.csv'
    # 'http://metacorps.io/static/data/viomet-{}-snapshot-project-df.csv'

def date_range(year):
    return pd.date_range(
        datetime.date(year, 9, 1), datetime.date(year, 11, 30), freq='D'
    )

NETWORKS = ['MSNBC', 'CNN', 'Fox News']


def _local_date(d_str):
    '''
    Checking the raw data against
    http://www.trumptwitterarchive.com/archive/none/ftff/9-1-2016_11-30-2016
    revealed that the raw time is UTC.
    '''
    return pd.to_datetime(
            d_str
        ).tz_localize(
            'UTC'
        ).tz_convert(
            'US/Eastern'
        ).date()


def get_tweets_ts(candidate, year=2016):

    tweets = json.load(
        open('data/{}_tweets_{}.json'.format(candidate, year), 'r')
    )

    dates = [_local_date(el['created_at']) for el in tweets]

    focal_dates = [d for d in dates
                   if datetime.date(year, 9, 1) <= d
                   and d <= datetime.date(year, 11, 30)]

    date_counts = Counter(focal_dates)
    tweets_ts = pd.Series(index=date_range(year), data=0)
    for d, count in date_counts.items():
        tweets_ts[d] = count

    return tweets_ts


def plot_regressions(ts_df, by='all', year=2016, save_path=None):
    '''
    Arguments:
        ts_df (pandas.DataFrame): DataFrame with DateIndex of Twitter account
            stats and all faceted metaphorical violence frequency timeseries.
    '''
    if year == 2016:
        cand_handle = ['@HillaryClinton', '@realDonaldTrump']
        short_candidates = ['clinton', 'trump']
    elif year == 2012:
        cand_handle = ['@BarackObama', '@MittRomney']
        short_candidates = ['obama', 'romney']

    cand_color = ['blue', 'red']

    # Plot All MV Use against Trump/Clinton tweeting.
    if by == 'all':
        def annotate(x, y, axidx):
            axes[axidx].text(
                6.0, 6.25, 'r={:.2f}; p={:.5f}'.format(*pearsonr(x, y)),
                fontsize=14
            )

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))

        xcol = short_candidates[0]
        ycol = 'metvi_all'
        # If there are na vals it's due to missing data; do not use.
        dfcols = ts_df[[xcol, ycol]].dropna()
        x = dfcols[xcol]
        y = dfcols[ycol]

        sns.regplot(x=x, y=y, ax=axes[0], color='blue', ci=None)

        linreg = linregress(x, y)
        axes[0].set_title(
            r'slope={:.2f}; $r^2={:.2f}$, $p={:.3f}$'.format(
               linreg[0], linreg[2]**2, linreg[3]
            ),
            fontsize=14
        )

        xcol = short_candidates[1]
        dfcols = ts_df[[xcol, ycol]].dropna()
        x = dfcols[xcol]
        y = dfcols[ycol]

        sns.regplot(x=x, y=y, ax=axes[1], color='red', ci=None)

        axes[0].set_ylim(axes[1].get_ylim())
        # axes[0].set_xlim(axes[1].get_xlim() + np.array([-1.0, 5.0]))

        # axes[1].set_title(
        #     'r={:.2f}; p={:.3f}'.format(*pearsonr(x, y)),
        #     fontsize=14
        # )
        linreg = linregress(x, y)
        axes[1].set_title(
            r'slope={:.2f}; $r^2={:.2f}$, $p={:.3f}$'.format(
               linreg[0], linreg[2]**2, linreg[3]
            ),
            fontsize=14
        )

        plt.subplots_adjust(wspace=2.5)

        # axes[1].set_xlim(axes[0].get_xlim())
        axes[0].set_xlim(-2.5, 100)
        axes[1].set_xlim(-2.5, 100)

        axes[0].set_xlabel('# {} tweets'.format(cand_handle[0]))
        axes[1].set_xlabel('# {} tweets'.format(cand_handle[1]))
        axes[0].set_ylabel('MV Frequency (All)')
        axes[1].set_ylabel('')

    # Plot MV use by network against Trump/Clinton tweeting.
    if by == 'network':
        networks = ['msnbc', 'cnn', 'foxnews']
        cand_color = ['blue', 'red']
        network_name = ['MSNBC', 'CNN', 'Fox News']

        fig, axes = plt.subplots(3, 2, figsize=(9.5, 10))
        for net_idx, network in enumerate(networks):
            for cand_idx, candidate in enumerate(short_candidates):
                xcol = candidate
                ycol = 'metvi_' + network
                dfcols = ts_df[[xcol, ycol]].dropna()
                x = dfcols[xcol]
                y = dfcols[ycol]

                sns.regplot(
                    x=x, y=y, ax=axes[net_idx, cand_idx],
                    color=cand_color[cand_idx], ci=None
                )
                if net_idx == 2:
                    axes[net_idx, cand_idx].set_xlabel(
                        '# {} tweets'.format(cand_handle[cand_idx])
                    )
                else:
                    axes[net_idx, cand_idx].set_xlabel('')
                if cand_idx == 0:
                    axes[net_idx, cand_idx].set_ylabel(
                        '{} MV Freq.'.format(network_name[net_idx])
                    )
                else:
                    axes[net_idx, cand_idx].set_ylabel('')

                axes[net_idx, cand_idx].set_xlim(-2.5, 100)
                axes[net_idx, cand_idx].set_ylim(-0.5, 8.5)

                # Don't know why but need this astype to avoid scipy err. XXX
                linreg = linregress(x.astype(float), y.astype(float))
                axes[net_idx, cand_idx].set_title(
                    r'slope={:.2f}; $r^2={:.2f}$, $p={:.3f}$'.format(
                       linreg[0], linreg[2]**2, linreg[3]
                    ),
                    fontsize=14
                )
                # axes[net_idx, cand_idx].set_title(
                #     'r={:.2f}; p={:.3f}'.format(*pearsonr(x, y)),
                #     fontsize=14
                # )

            plt.subplots_adjust(wspace=2.5)


    # Plot MV use by subject/object against Trump/Clinton tweeting.
    # Here we want to see how often Clinton's tweeting casts her as
    # subject and Trump as object, and vice-versa. Now we want two columns,
    # one for each tweeting. Two rows, one for self-subject and one for
    # other-object.
    if by == 'subjobj':

        fig, axes = plt.subplots(2, 2, figsize=(9.5, 8))

        # Set up column names and y-axis labels of interest.
        if year == 2016:
            ycol_label_append = [
                'clinton_subj', 'trump_subj', 'trump_obj', 'clinton_obj'
            ]
            ycol_labels = [
                'Subject=Clinton MV freq', 'Subject=Trump MV freq',
                'Object=Trump MV freq', 'Object=Clinton MV freq'
            ]
        elif year == 2012:
            ycol_label_append = [
                'obama_subj', 'romney_subj', 'romney_obj', 'obama_obj'
            ]
            ycol_labels = [
                'Subject=Obama MV freq', 'Subject=Romney MV freq',
                'Object=Romney MV freq', 'Object=Obama MV freq'
            ]


        # Track which timeseries dataframe column we are plotting.
        ycol_idx = 0
        for row_idx in range(2):
            for cand_idx, candidate in enumerate(short_candidates):
                xcol = candidate
                ycol = 'metvi_' + ycol_label_append[ycol_idx]
                dfcols = ts_df[[xcol, ycol]].dropna()
                x = dfcols[xcol]
                y = dfcols[ycol]

                sns.regplot(
                    x=x, y=y, ax=axes[row_idx, cand_idx],
                    color=cand_color[cand_idx], ci=None
                )
                if row_idx == 1:
                    axes[row_idx, cand_idx].set_xlabel(
                        '# {} tweets'.format(cand_handle[cand_idx])
                    )
                else:
                    axes[row_idx, cand_idx].set_xlabel('')

                axes[row_idx, cand_idx].set_ylabel(
                    ycol_labels[ycol_idx]
                )

                # axes[row_idx, cand_idx].set_title(
                #     'r={:.2f}; p={:.5f}'.format(*pearsonr(x, y)),
                #     fontsize=14
                # )
                linreg = linregress(x, y)
                axes[row_idx, cand_idx].set_title(
                    r'slope={:.2f}; $r^2={:.2f}$, $p={:.3f}$'.format(
                       linreg[0], linreg[2]**2, linreg[3]
                    ),
                    fontsize=14
                )
                axes[row_idx, cand_idx].set_ylim(-0.2, 5.25)
                axes[row_idx, cand_idx].set_xlim(-2.5, 100)

                ycol_idx += 1

    if save_path:
        plt.savefig(save_path)


def get_subj_ts(df, subj, year=2016):
    '''
    Get MV use frequency timeseries for a single subject, e.g., Hillary Clinton
    '''
    df = df.copy()
    # Noticed some cases of, e.g., 'Donald Trump '.
    df.subjects = df.subjects.str.strip()
    subj_df = df[df.subjects == subj]

    # If we have na at this step it's due to dividing by zero counts.
    # See daily_frequency for more.
    return daily_frequency(
        subj_df, date_range(year), by=['subjects']  #, predropna=True
    )[subj].fillna(0.0)


def get_obj_ts(df, obj, year=2016):
    '''
    Get MV use frequency timeseries for a single subject, e.g., Hillary Clinton
    '''
    df = df.copy()
    # Noticed some cases of, e.g., 'Donald Trump '.
    df.objects = df.objects.str.strip()
    obj_df = df[df.objects == obj]

    # If we have na at this step it's due to dividing by zero counts.
    return daily_frequency(
        obj_df, date_range(year), by=['objects']  # , predropna=True
    )[obj].fillna(0.0)


def get_network_ts(df, network, year=2016):
    '''
    MV use frequency for each network.
    '''
    df = df.copy()
    # XXX daily_frequency by network works somewhat differently, but not
    # too sure how to describe it so...WATCH OUT!
    return daily_frequency(
        df, date_range(year), by=['network']
    )[network].dropna()


DEBATE_DATES = {
    2012: np.array(['2012-10-03', '2012-10-16', '2012-10-22'],
                     dtype='datetime64[D]'),
    2016: np.array(['2016-09-26', '2016-10-09', '2016-10-19'],
                     dtype='datetime64[D]')
}


def _days_from_debate(year, dates):

    debate_dates = DEBATE_DATES[year]
    sy = str(year)
    dates = np.array(dates, dtype='datetime64[D]')
    # dates = np.arange(sy + '-09-01', sy + '-11-30', dtype='datetime64[D]')
    days_from_debates = np.array([date - debate_dates for date in dates])
    return np.absolute(days_from_debates).min(axis=1).astype(int)

def data_for_model(year=2016, save_dir=None):
    '''
    Create a dataframe with all series needed to make regressions of
    faceted MV frequencies.
    '''
    # Create metaphorical violence frequency series across all networks.
    csv = os.path.join('Data', 'viomet-sep-nov-{}.csv'.format(year))
    # viomet_df = pd.read_csv(url, na_values='',
    #                  parse_dates=['start_localtime'])
    project_df = get_project_data_frame(csv)
    project_df = project_df[project_df.include]
    freq_df = daily_frequency(project_df, date_range(year))
    metvi_ts = pd.Series(index=freq_df.index, data=freq_df['freq'], dtype=float)

    days_from_debate = _days_from_debate(year, freq_df.index)

    # Create timeseries of tweets.
    if year == 2016:
        ts_data = dict(
            # Number of days before or after debate.
            days_from_debate=days_from_debate,

            # Twitter timeseries.
            trump=get_tweets_ts('trump'),
            clinton=get_tweets_ts('clinton'),

            # All metaphorical violence freq timeseries.
            metvi_all=metvi_ts,

            # Trump as subject or object metvi freq timeseries.
            metvi_trump_subj=get_subj_ts(project_df, 'Donald Trump'),
            metvi_trump_obj=get_obj_ts(project_df, 'Donald Trump'),

            # Clinton as subject or object metvi freq timeseries.
            metvi_clinton_subj=get_subj_ts(project_df, 'Hillary Clinton'),
            metvi_clinton_obj=get_obj_ts(project_df, 'Hillary Clinton'),

            # Metvi freq on networks timeseries.
            metvi_msnbc=get_network_ts(project_df, 'MSNBCW'),
            metvi_cnn=get_network_ts(project_df, 'CNNW'),
            metvi_foxnews=get_network_ts(project_df, 'FOXNEWSW')
        )
    elif year == 2012:
        ts_data = dict(
            # Number of days before or after debate.
            days_from_debate=days_from_debate,
            # Twitter timeseries.
            romney=get_tweets_ts('romney', year=2012),
            obama=get_tweets_ts('obama', year=2012),

            # All metaphorical violence freq timeseries.
            metvi_all=metvi_ts,

            # Trump as subject or object metvi freq timeseries.
            metvi_romney_subj=get_subj_ts(project_df, 'Mitt Romney', year=2012),
            metvi_romney_obj=get_obj_ts(project_df, 'Mitt Romney', year=2012),

            # Clinton as subject or object metvi freq timeseries.
            metvi_obama_subj=get_subj_ts(project_df, 'Barack Obama', year=2012),
            metvi_obama_obj=get_obj_ts(project_df, 'Barack Obama', year=2012),

            # Metvi freq on networks timeseries.
            metvi_msnbc=get_network_ts(project_df, 'MSNBCW', year=2012),
            metvi_cnn=get_network_ts(project_df, 'CNNW', year=2012),
            metvi_foxnews=get_network_ts(project_df, 'FOXNEWSW', year=2012)
        )

    return pd.DataFrame(ts_data)
