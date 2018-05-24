'''
Testing hypothesis that metaphorical violence usage in 2016 is correlated with
Donald Trump's tweeting.
'''
import datetime as dt
import json
import os
import numpy as np
import pandas as pd

from collections import Counter
from os.path import join as osjoin

os.environ['CONFIG_FILE'] = 'conf/default.cfg'

from metacorps.app.models import IatvCorpus


METAPHORS_URL = \
    'http://metacorps.io/static/data/viomet-2016-snapshot-project-df.csv'
DATE_RANGE = pd.date_range('2016-9-1', '2016-11-30', freq='D')
IATV_CORPUS_NAME = 'Viomet Sep-Nov 2016'


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


def get_tweets_ts(candidate):

    tweets = json.load(open('data/{}_tweets_2016.json'.format(candidate), 'r'))
    dates = [_local_date(el['created_at']) for el in tweets]
    focal_dates = [d for d in dates
                   if dt.date(2016, 9, 1) <= d
                   and d <= dt.date(2016, 11, 30)]
    date_counts = Counter(focal_dates)
    tweets_ts = pd.Series(index=DATE_RANGE, data=0)
    for d, count in date_counts.items():
        tweets_ts[d] = count

    return tweets_ts


def get_subjobj_ts(**kwargs):
    pass


def correlate(save_dir=None):

    # Create metaphorical violence frequency series across all networks.
    df = pd.read_csv(METAPHORS_URL, na_values='',
                     parse_dates=['start_localtime'])
    freq_df = daily_frequency(df, DATE_RANGE, IATV_CORPUS_NAME)
    metvi_ts = pd.Series(index=freq_df.index, data=freq_df['freq'])
    metvi_ts.fillna(0.0, inplace=True)

    # Create timeseries of Trump tweets.
    ts_data = dict(
        trump=get_tweets_ts('trump'),
        clinton=get_tweets_ts('clinton'),
        metvi_all=metvi_ts,
        # metvi_trump_subj=get_subjobj_ts(subj='trump'),
        # metvi_trump_obj=get_subjobj_ts(obj='trump'),
        # metvi_clinton_subj=get_subjobj_ts(subj='clinton'),
        # metvi_clinton_obj=get_subjobj_ts(obj='clinton')
    )

    if save_dir is not None:

        def mkpath(name):
            return osjoin(save_dir, name + '.csv')

        save_paths = dict(
            trump=mkpath('trump-tweets'),
            clinton=mkpath('clinton-tweets'),
            metvi_all=mkpath('metvi-all'),
            # metvi_trump_subj=mkpath('metvi-trump-subj'),
            # metvi_trump_obj=mkpath('metvi-trump-obj'),
            # metvi_clinton_subj=mkpath('metvi-clinton-subj'),
            # metvi_clinton_obj=mkpath('metvi-clinton-obj')
        )

        for key, path in save_paths.items():
            ts = ts_data[key]
            ts.to_csv(path, header=False)

    return ts_data


def daily_frequency(df, date_index, iatv_corpus, by=None):

    if by is not None and 'network' in by:
        spd = shows_per_date(date_index, iatv_corpus, by_network=True)
        daily = daily_metaphor_counts(df, date_index, by=by)
        ret = daily.div(spd, axis='rows')

    elif by is None:
        spd = shows_per_date(date_index, iatv_corpus)
        daily = daily_metaphor_counts(df, date_index, by=by)
        ret = daily.div(spd, axis='rows')
        ret.columns = ['freq']

    else:
        spd = shows_per_date(date_index, iatv_corpus)
        daily = daily_metaphor_counts(df, date_index, by=by)
        ret = daily.div(spd, axis='rows')

    return ret


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
                (d.program_name, d.start_localtime.date())
                for d in docs
            ]
        )

        # count total number of shows on each date
        # note we count the second entry of the tuples, which is just the
        # date, excluding program name
        shows_per_date = Counter(el[1] for el in prog_dates)

        spd_series = pd.Series(
            index=date_index,
            data={'counts': np.zeros(n_dates)}
        ).sort_index()

        for date in shows_per_date:
            spd_series.loc[date] = shows_per_date[date]

        return spd_series

    else:
        # get all date/network/show name tuples
        # & remove show re-runs from same date
        prog_dates = set(
            [
                (d.program_name, d.network, d.start_localtime.date())
                for d in docs
            ]
        )

        # count total number of shows on each date for each network
        # note we count the second entry of the tuples, which is just the
        # date, excluding program name
        shows_per_network_per_date = Counter(el[1:] for el in prog_dates)

        n_dates = len(date_index)
        spd_frame = pd.DataFrame(
            index=date_index,
            data={
                'MSNBCW': np.zeros(n_dates),
                'CNNW': np.zeros(n_dates),
                'FOXNEWSW': np.zeros(n_dates)
            }
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

    groupby_spec = [counts.start_localtime.dt.date, *counts[by]]

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
