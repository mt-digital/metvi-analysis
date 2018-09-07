import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from collections import OrderedDict, Counter
from copy import deepcopy
from datetime import datetime, timedelta
from urllib.parse import urlparse


DEFAULT_FACET_WORDS = [
    'attack',
    'hit',
    'beat',
    'grenade',
    'slap',
    'knock',
    'jugular',
    'smack',
    'strangle',
    'slug',
]


def get_project_data_frame(project_name):
    '''
    Convenience method for creating a newly initialized instance of the
    Analyzer class. Currently the only argument is year since the projects all
    contain a year. In the future we may want to match some other unique
    element of a title, or create some other kind of wrapper to search
    all Project names in the metacorps database.

    Arguments:
        project_name (str): Identifier for building a dataframe. It could be
            a year corresponding to either Viomet study year (even though this
            is in projects/common), a URL or local path to an exported .csv
            file. Must be made first by giving the name of the IatvCorpus
            collection in MongoDB to the ProjectExporter (this directory).

    Returns:
        (pandas.DataFrame): ready for use in analyses
    '''
    def is_url(s): return urlparse(project_name).hostname is not None

    if is_url(project_name) or os.path.exists(project_name):
        ret = pd.read_csv(project_name, na_values='',
                          parse_dates=['start_localtime'])
        return ret


def _select_range_and_pivot_subj_obj(date_range, counts_df, subj_obj):

    rng_sub = counts_df[
        date_range[0] <= counts_df.start_localtime
    ][
        counts_df.start_localtime <= date_range[1]
    ]

    rng_sub_sum = rng_sub.groupby(['network', subj_obj]).agg(sum)

    ret = rng_sub_sum.reset_index().pivot(
        index='network', columns=subj_obj, values='counts'
    )

    return ret


def _count_daily_subj_obj(df, sub_obj):

    subs = df[['start_localtime', 'network', 'subjects', 'objects']]

    subs.subjects = subs.subjects.map(lambda s: s.strip().lower())
    subs.objects = subs.objects.map(lambda s: s.strip().lower())

    try:
        trcl = subs[
            (subs[sub_obj].str.contains('hillary clinton') |
             subs[sub_obj].str.contains('donald trump')) &
            subs[sub_obj].str.contains('/').map(lambda b: not b) &
            subs[sub_obj].str.contains('campaign').map(lambda b: not b)
        ]
    except KeyError:
        raise RuntimeError('sub_obj must be "subjects" or "objects"')

    c = trcl.groupby(['start_localtime', 'network', sub_obj]).size()

    ret_df = c.to_frame()
    ret_df.columns = ['counts']
    ret_df.reset_index(inplace=True)

    # cleanup anything like 'republican nominee'
    ret_df.loc[
        :, sub_obj
    ][

        ret_df[sub_obj].str.contains('donald trump')

    ] = 'donald trump'

    ret_df.loc[
        :, sub_obj
    ][

        ret_df[sub_obj].str.contains('hillary clinton')

    ] = 'hillary clinton'

    return ret_df


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


def shows_per_date(project_df, date_index, by_network=False):
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
    n_dates = len(date_index)

    if not by_network:

        # Get all date/show name tuples & remove show re-runs from same date.
        prog_dates = set(
            zip(project_df.program_name, project_df.start_localtime.dt.date)
        )

        # Count total number of shows on each date
        # note we count the second entry of the tuples, which is just the
        # date, excluding program name.
        shows_per_date = Counter(el[1] for el in prog_dates)

        spd_series = pd.Series(
            index=date_index,
            data={'counts': np.zeros(n_dates)}
        ).sort_index()

        for date in shows_per_date:
            spd_series.loc[date] = shows_per_date[date]

        return spd_series

    else:
        # Get all date/network/show name tuples
        # & remove show re-runs from same date.
        prog_dates = set(
            zip(project_df.program_name,
                project_df.network,
                project_df.start_localtime.dt.date
            )
        )

        # Count total number of shows on each date for each network
        # note we count the second entry of the tuples, which is just the
        # date, excluding program name.
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


def daily_metaphor_counts(project_df, date_index, by=None):
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

    counts = _count_by_start_localtime(project_df, column_list=by)

    counts = counts.rename(columns={'start_localtime': 'date'})
    counts.date = counts.date.dt.date

    groupby_spec = ['date'] + by

    counts_gb = counts.groupby(groupby_spec).sum().reset_index()

    ret = pd.pivot_table(counts_gb, index='date', values='counts',
                         columns=by, aggfunc='sum').fillna(0)

    return ret


def daily_frequency(project_df, date_index, by=None):

    instances = project_df[project_df.include]

    if by is not None and 'network' in by:
        spd = shows_per_date(project_df, date_index, by_network=True)
        daily = daily_metaphor_counts(instances, date_index, by=by)
        ret = daily.div(spd, axis='rows')

    elif by is None:
        spd = shows_per_date(project_df, date_index)
        daily = daily_metaphor_counts(instances, date_index, by=by)
        ret = daily.div(spd, axis='rows')
        ret.columns = ['freq']

    else:
        spd = shows_per_date(project_df, date_index)
        daily = daily_metaphor_counts(instances, date_index, by=by)
        ret = daily.div(spd, axis='rows')

    return ret


def facet_word_count(analyzer_df, facet_word_index, by_network=True):
    '''
    Count the number of times each facet word has been used. If by_network is
    True, compute the usage of each word by network.

    Arguments:
        analyzer_df (pandas.DataFrame): dataframe of the IatvCorpus annotations
        by_network (bool): group each partition's word counts by network?

    Returns:
        (pandas.DataFrame) or (pandas.Series) of counts depending on by_network
    '''
    if by_network:
        return analyzer_df.groupby(
                ['network', 'facet_word']
            ).size().unstack(level=0)[
                ['MSNBCW', 'CNNW', 'FOXNEWSW']
            ].loc[facet_word_index].fillna(0.0)
    else:
        return analyzer_df.groupby(
                ['facet_word']
            ).size().loc[facet_word_index].fillna(0.0)
