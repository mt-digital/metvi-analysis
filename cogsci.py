import numpy as np
import pandas as pd

from collections import Counter
from datetime import date


# For some reason there are still reruns. Remove them here, but later
# need to fix upstream somewhere.
#
# e.g. 'CNNW_20161102_000000_Anderson_Cooper_360',
#      'CNNW_20161102_010000_Anderson_Cooper_360'
def remove_reruns(df):

    def _remove_time(iatv_id):
        spl = iatv_id.split('_')
        network = spl[0]
        date = spl[1]
        time = spl[2]

        program_name = ' '.join(spl[3:])

        return (network, pd.to_datetime(date).date(), program_name)

    def _extract_time(iatv_id):
        return iatv_id.split('_')[2]

    date_shows = df.iatv_id.apply(_remove_time)
    times = df.iatv_id.apply(_extract_time)

    ds_counts = Counter(date_shows)

    # Find dates where there are re-runs of the same show.
    dates_with_reruns = [k for k in ds_counts.keys() if ds_counts[k] > 1]

    for r in dates_with_reruns:
        network = r[0]
        date_with_reruns = r[1]
        program_name = r[2]

        # Get rid of all other instances of AC 360 that aren't shown
        # at this time because they are re-runs.
        # keep_time = '010000'
        relevant_rows = (
            (df.date == date_with_reruns) &
            (df.program_name == program_name)
        )

        # Build list of IATV-formatted times, e.g. 010000, for the
        # current date with reruns. We will drop all but the first
        # one, with the list sorted in ascending order.
        times_to_remove = [
            r[1].iatv_id.split('_')[2]  # iterrows returns (index, row)
            for r in df[relevant_rows].iterrows()
        ]
        times_to_remove = sorted(times_to_remove)[1:]

        str_date = date_with_reruns.strftime('%Y%m%d')

        iatv_ids_to_remove = [
            '{}_{}_{}_'.format(
                network,
                str_date,
                time_to_remove
            ).replace('-', '_') +   # worst python code ever :-X
            '_'.join(program_name.split(' '))
            for time_to_remove in times_to_remove
        ]
        df = df[~df.iatv_id.isin(iatv_ids_to_remove)]

    return df


def normalized_groupby_n(full_df, by='month', count_var='n'):
    '''
    Return metaphor counts grouped by `by` and normalized by the number of
    episodes in the relevant grouping.
    '''
    full_df = remove_reruns(full_df)

    metaphors_gb_summed = full_df.groupby(by).sum()

    by_vec = full_df[by]
    by_counts = Counter(by_vec)

    print('total counts:', metaphors_gb_summed.n)
    print('\nnumber of episodes per category:', by_counts)

    # Series data are counts of metaphor use in groupby category normalized
    # by the total number of episodes in that category.
    data = [
        metaphors_gb_summed.loc[gb_key][count_var] / by_counts[gb_key]
        for gb_key in metaphors_gb_summed.index
    ]

    return pd.Series(
        # Our output Series has the same index as the summed group-by above.
        index=metaphors_gb_summed.index,
        data=data
    )


def make_data_table(base_df, metvi_df, year=2016):
    '''
    base_df has an entry for every date that has at least one show, whether
    or not there was a violence metaphor observed. The metvi_df has metaphor
    annotations. For each row of the base_df, look up the number of times the
    Republican or Democrat candidate was either the subject or object of
    metaphorical violence, appending four new columns for each combination of
    grammatical type and political party. Further aggregations, such as
    resampling by month, are to be done after the four columns are added by
    calling this function.
    '''
    metvi_df['date'] = pd.to_datetime(metvi_df.start_time).dt.date

    if year == 2016:
        republican = 'Donald Trump'
        democrat = 'Hillary Clinton'
    elif year == 2012:
        republican = 'Mitt Romney'
        democrat = 'Barack Obama'
    else:
        raise ValueError('No data for year ' + str(year))

    def _make_counts(iatv_id, metvi_df):

        # For some reason date_ is sometimes being read as string. My solution
        # is to lazily do this instead of fixing actual problem...
        # date_ = pd.to_datetime(date_).date()

        rows = metvi_df[metvi_df.iatv_id == iatv_id]

        repsubj = np.sum(rows.subjects.str.contains(republican))
        repobj = np.sum(rows.objects.str.contains(republican))
        demsubj = np.sum(rows.subjects.str.contains(democrat))
        demobj = np.sum(rows.objects.str.contains(democrat))

        return (repsubj, repobj, demsubj, demobj)

    subjobj_rows = [
        _make_counts(iatv_id, metvi_df)
        for iatv_id in base_df.iatv_id
    ]

    subjobj = pd.DataFrame(
        data=subjobj_rows, columns=['RepSubj', 'RepObj', 'DemSubj', 'DemObj']
    )

    full = pd.concat([base_df, subjobj], axis=1, join_axes=[base_df.index])

    # Read date string.
    full['date'] = pd.to_datetime(full.date)
    # Create special month column for convenience, I suppose.
    full['month'] = full.date.dt.month_name()
    full['date'] = full['date'].map(lambda d: d.date())

    # Calculate number of days before or after temporally-nearest debate.
    if year == 2016:
        db_dates = [
            date(2016, 9, 26), date(2016, 10, 9), date(2016, 10, 19)
        ]
    elif year == 2012:
        db_dates = [
            date(2012, 10, 3), date(2012, 10, 16), date(2012, 10, 22)
        ]
    full['daysFromDebate'] = full.date.map(
        lambda d:
            np.min([abs((db_d - d).days) for db_d in db_dates])
    )

    # Candidate Twitter.
    # if year == 2016:
    #     rep = 'Donald Trump'
    #     rep_short = 'trump'
    #     dem = 'Hillary Clinton'
    #     dem_short = 'clinton'

    # elif year == 2012:
    #     rep = 'Mitt Romney'
    #     rep_short = 'romney'
    #     dem = 'Barack Obama'
    #     dem_short = 'obama'

    # rep_ts = get_tweets_ts(rep_short)
    # dem_ts = get_tweets_ts(dem_short)

    # full['RepTweets'] = rep_ts[full['date']].values
    # full['DemTweets'] = dem_ts[full['date']].values

    return full
