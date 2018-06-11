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
from os.path import join as osjoin
from scipy.stats import pearsonr, linregress

os.environ['CONFIG_FILE'] = 'conf/default.cfg'

from metacorps.app.models import IatvCorpus


METAPHORS_URL_TEMPLATE = \
    'http://metacorps.io/static/data/viomet-{}-snapshot-project-df.csv'

def date_range(year):
    return pd.date_range(
        datetime.date(year, 9, 1), datetime.date(year, 11, 30), freq='D'
    )

def iatv_corpus_name(year):
    return 'Viomet Sep-Nov {}'.format(year)

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


def get_project_dataframe(url=METAPHORS_URL_TEMPLATE.format(2016)):

    return pd.read_csv(
        url, na_values='', parse_dates=['start_localtime']
    )


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
        subj_df, date_range(year), iatv_corpus_name(year), by=['subjects'], predropna=True
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
        obj_df, date_range(year), iatv_corpus_name(year), by=['objects'], predropna=True
    )[obj].fillna(0.0)


def get_network_ts(df, network, year=2016):
    '''
    MV use frequency for each network.
    '''
    df = df.copy()
    # XXX daily_frequency by network works somewhat differently, but not
    # too sure how to describe it so...WATCH OUT!
    return daily_frequency(
        df, date_range(year), iatv_corpus_name(year), by=['network']
    )[network].dropna()


def correlate_data(year=2016, save_dir=None):
    '''
    Create a dataframe with all series needed to make regressions of
    faceted MV frequencies.
    '''
    # Create metaphorical violence frequency series across all networks.
    url = METAPHORS_URL_TEMPLATE.format(year)
    viomet_df = pd.read_csv(url, na_values='',
                     parse_dates=['start_localtime'])
    freq_df = daily_frequency(viomet_df, date_range(year), iatv_corpus_name(year))
    metvi_ts = pd.Series(index=freq_df.index, data=freq_df['freq'])

    # Create timeseries of tweets.
    if year == 2016:
        ts_data = dict(
            # Twitter timeseries.
            trump=get_tweets_ts('trump'),
            clinton=get_tweets_ts('clinton'),

            # All metaphorical violence freq timeseries.
            metvi_all=metvi_ts,

            # Trump as subject or object metvi freq timeseries.
            metvi_trump_subj=get_subj_ts(viomet_df, 'Donald Trump'),
            metvi_trump_obj=get_obj_ts(viomet_df, 'Donald Trump'),

            # Clinton as subject or object metvi freq timeseries.
            metvi_clinton_subj=get_subj_ts(viomet_df, 'Hillary Clinton'),
            metvi_clinton_obj=get_obj_ts(viomet_df, 'Hillary Clinton'),

            # Metvi freq on networks timeseries.
            metvi_msnbc=get_network_ts(viomet_df, 'MSNBCW'),
            metvi_cnn=get_network_ts(viomet_df, 'CNNW'),
            metvi_foxnews=get_network_ts(viomet_df, 'FOXNEWSW')
        )
    elif year == 2012:
        ts_data = dict(
            # Twitter timeseries.
            romney=get_tweets_ts('romney', year=2012),
            obama=get_tweets_ts('obama', year=2012),

            # All metaphorical violence freq timeseries.
            metvi_all=metvi_ts,

            # Trump as subject or object metvi freq timeseries.
            metvi_romney_subj=get_subj_ts(viomet_df, 'Mitt Romney', year=2012),
            metvi_romney_obj=get_obj_ts(viomet_df, 'Mitt Romney', year=2012),

            # Clinton as subject or object metvi freq timeseries.
            metvi_obama_subj=get_subj_ts(viomet_df, 'Barack Obama', year=2012),
            metvi_obama_obj=get_obj_ts(viomet_df, 'Barack Obama', year=2012),

            # Metvi freq on networks timeseries.
            metvi_msnbc=get_network_ts(viomet_df, 'MSNBCW', year=2012),
            metvi_cnn=get_network_ts(viomet_df, 'CNNW', year=2012),
            metvi_foxnews=get_network_ts(viomet_df, 'FOXNEWSW', year=2012)
        )

    return pd.DataFrame(ts_data)


def daily_frequency(df, date_index, iatv_corpus, by=None, predropna=False):

    if by is not None and 'network' in by:
        spd = shows_per_date(date_index, iatv_corpus, by_network=True)
        if predropna:
            spd = spd.dropna()
            daily = daily_metaphor_counts(df, date_index, by=by)
        else:
            daily = daily_metaphor_counts(df, date_index, by=by)

        ret = daily.div(spd, axis='rows')

    elif by is None:
        spd = shows_per_date(date_index, iatv_corpus)
        if predropna:
            spd = spd.dropna()
            daily = daily_metaphor_counts(df, date_index, by=by)
        else:
            daily = daily_metaphor_counts(df, date_index, by=by)

        ret = daily.div(spd, axis='rows')
        ret.columns = ['freq']

    else:
        spd = shows_per_date(date_index, iatv_corpus)
        if predropna:
            spd = spd.dropna()
            daily = daily_metaphor_counts(df, date_index, by=by)
        else:
            daily = daily_metaphor_counts(df, date_index, by=by)

        ret = daily.div(spd, axis='rows')

    return ret


def _get_prog_date(doc):
    '''
    If the hour of broadcast is in the early morning it is a re-run
    from previous day
    '''
    if type(doc) is pd.Timestamp:
        slt = doc
        d = slt.date()
    else:
        slt = doc.start_localtime
        d = slt.date()

    if slt.hour < 8:
        d -= datetime.timedelta(days=1)

    return d


def shows_per_date(date_index, iatv_corpus, by_network=False):
    '''
    Arguments:
        date_index (pandas.DatetimeIndex): Full index of dates covered by
            data
        iatv_corpus (app.models.IatvCorpus): Obtained, e.g., using
            `iatv_corpus = IatvCorpus.objects.get(name='Viomet Sep-Nov 2016')`
        by_network (bool): whether or not to do a faceted daily count
            by network

    Returns:
        (pandas.Series) if by_network is False, (pandas.DataFrame)
            if by_network is true.
    '''
    if type(iatv_corpus) is str:
        iatv_corpus = IatvCorpus.objects(name=iatv_corpus)[0]

    docs = iatv_corpus.documents

    n_dates = len(date_index)
    if not by_network:

        # get all date/show name tuples & remove show re-runs from same date
        prog_dates = set(
            [
                (d.program_name, _get_prog_date(d))
                for d in docs
            ]
        )

        # count total number of shows on each date
        # note we count the second entry of the tuples, which is just the
        # date, excluding program name
        year = list(prog_dates)[0][1].year
        shows_per_date = Counter(el[1] for el in prog_dates
                                 if el[1] > datetime.date(year, 8, 31))

        spd_series = pd.Series(
            index=date_index,
            data={'counts': np.zeros(n_dates)}
        ).sort_index()


        try:
            for date in shows_per_date:
                spd_series.loc[date] = shows_per_date[date]
        except:
            import ipdb
            ipdb.set_trace()

        return spd_series

    else:
        # get all date/network/show name tuples
        # & remove show re-runs from same date
        prog_dates = set(
            [
                (d.program_name, d.network, _get_prog_date(d))
                for d in docs
            ]
        )

        # count total number of shows on each date for each network
        # note we count the second entry of the tuples, which is just the
        # date, excluding program name
        year = list(prog_dates)[0][2].year
        shows_per_network_per_date = Counter(
            el[1:] for el in prog_dates if el[2] > datetime.date(year, 8, 31)
        )

        n_dates = len(date_index)
        spd_frame = pd.DataFrame(
            index=date_index,
            columns=['MSNBCW', 'CNNW', 'FOXNEWSW']
        )

        for tup in shows_per_network_per_date:
            spd_frame.loc[tup[1]][tup[0]] = shows_per_network_per_date[tup]

        return spd_frame


def daily_metaphor_counts(df, date_index, by=None):
    '''
    Given an Analyzer.df, creates a pivot table with date_index as index. Will
    group by the column names given in by. First deals with hourly data in
    order to build a common index with hourly data, which is the data's
    original format.

    Arguments:
        df (pandas.DataFrame)
        by (list(str))
        date_index (pandas.core.indexes.datetimes.DatetimeIndex): e.g.
            `pd.date_range('2016-09-01', '2016-11-30', freq='D')`
    '''
    # get initial counts by localtime
    if by is None:
        by = []

    counts = _count_by_start_localtime(df, column_list=by)

    # groupby_spec = [counts.start_localtime.dt.date, *counts[by]]
    groupby_spec = [
        pd.Series([_get_prog_date(d) for d in counts.start_localtime], name='start_localtime'),
        *counts[by]
    ]

    counts_gb = counts.groupby(groupby_spec).sum().reset_index()

    ret = pd.pivot_table(counts_gb, index='start_localtime', values='counts',
                         columns=by, aggfunc='sum').fillna(0)

    return ret


def _count_by_start_localtime(df,
                              column_list=['program_name',
                                           'network',
                                           'facet_word']):
    '''
    Count the number of instances grouped by column_list. Adds a 'counts'
    column.

    Arguments:
        df (pandas.DataFrame): Analyzer.df attribute from Analyzer class
        column_list (list): list of columns on which to groupby then count

    Returns:
        (pandas.DataFrame) counts per start_localtime of tuples with types
            given in column_list
    '''
    all_cols = ['start_localtime'] + column_list

    subs = df[all_cols]

    c = subs.groupby(all_cols).size()

    ret_df = c.to_frame()
    ret_df.columns = ['counts']
    ret_df.reset_index(inplace=True)

    return ret_df
