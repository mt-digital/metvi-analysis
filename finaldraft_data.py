##
# Function to create the data for statistical modeling in R.
#
# Ideally the final version of tabular data to be analyzed for Results section
# of the viomet paper being submitted to Political Communication.
#
# Date: 12/28/2018
#
import numpy as np
import pandas as pd

from datetime import datetime, date, timedelta

from metacorps.app.models import IatvDocument
from correlate_tweets import get_tweets_ts
# from projects.common.export_project import ProjectExporter


SHOWS = [
    'The Last Word With Lawrence O\'Donnell',
    'The O\'Reilly Factor',
    'Erin Burnett OutFront',
    'Anderson Cooper 360',
    'The Kelly File',
    'The Rachel Maddow Show'
]


def _remove_reruns(documents, hours_delta=10):
    '''
    I've probably implemented this before elsewhere, but I'm trying to do only
    the things necessary to make the target dataset and do them right. This
    removes all duplicate program_names shown within 12 hours of the original.
    I (PLAN TO) spotcheck this, with google sheet here (ADD LINK) showing which
    episodes this removes, and any relevant notes.

    This seems much slower than it should be, but I'm leaving it as-is because
    it seems to be correct and it should be used sparingly.

    Arguments:
        documents (list or query set of app.models.IatvDocuments)
    Returns:
        (list or query set of app.models.IatvDocuments) with reruns filtered
        out
    '''
    keep_list = []
    remove_list = []

    for doc in documents:
        # Check if there is a re-run within hours_delta hours before or after.
        dt = timedelta(hours=hours_delta)
        t1 = doc.start_time - dt
        t2 = doc.start_time + dt
        pname = doc.program_name
        repeats = documents(
            start_time__gte=t1,
            start_time__lte=t2,
            program_name=pname
        )
        # There should only be one show in that hours_delta-hour block. Mark
        # one for removal.
        if repeats.count() > 1:
            # Check to see if one of the repeats is already marked for removal.
            repeat_recorded = False
            for repeat in repeats:
                if repeat.id in remove_list:
                    repeat_recorded = True

            # If neither repeat is marked for removal, mark the current one.
            if not repeat_recorded:
                remove_list.append(doc.id)

    # See http://docs.mongoengine.org/guide/querying.html.
    return documents(id__nin=remove_list)


class Dataset:

    def __init__(self, year=2016):
        '''
        Just pass year, since this is specific for viomet. Year is either 2012
        or 2016, for each election year.
        '''
        self.year = year
        d1 = datetime(year, 9, 1, 0, 0, 0)
        d2 = datetime(year, 11, 30, 23, 59, 59)
        self.documents = _remove_reruns(
            IatvDocument.objects(
                start_time__gte=d1,
                start_time__lte=d2,
                program_name__in=SHOWS
            )
        )

        self._date_df = None
        self._days_from_df = None
        self.final_df = None

        # self.metvi_df = ProjectExporter('Viomet Sep-Nov {}'.format(year)
        #                                 ).export_dataframe()

        # Commenting out for now. Will load this externally and pass to
        # build_final method here.
        # self.metvi_df = self.metvi_df[
        #     self.metvi_df.facet_word.isin(['attack', 'hit', 'beat'])
        # ]

    @classmethod
    def build_final(cls):
        if cls.final is not None:
            return cls.final
        else:
            cls.final = 'something TBD'
            return cls.final

    def _build_date_df(self):
        '''
        This creates a new dataset that has a row for every unique showing
        of an episode of a program of interest. The number of instances of
        metaphorical violence will be added, and will be zero if there were
        none.
        '''
        if self._date_df is None:
            columns = [
                'date',
                'network',
                'iatv_id',
                'program_name',
                'n'
            ]
            df_data = [
                (d.start_time.date(), d.network, d.iatv_id, d.program_name, 0.0)
                for d in self.documents
            ]

            self._date_df = pd.DataFrame(columns=columns, data=df_data)

        return self._date_df

    def _build_days_from_df(self):
        '''
        Given the dataframe with dates, calculate the days from (ahead/behind)
        the nearest-in-time debate. Replace the date "column" with "days_from".
        '''
        # Debate dates come from debates.org.
        if self._days_from_df is None:
            if self.year == 2016:
                debate_dates = [
                    date(2016, 9, 26), date(2016, 10, 9), date(2016, 10, 19)
                ]
            elif self.year == 2012:
                debate_dates = [
                    date(2012, 10, 3), date(2012, 10, 16), date(2012, 10, 22)
                ]
            else:
                raise ValueError('No data for that year')

            if self._date_df is None:
                self._build_date_df();

            self._days_from_df = self._date_df.copy()
            del self._days_from_df['date']

            def _closest_debate_distance(date_, debate_dates):
                return np.min([abs(debate_date - date_)
                               for debate_date in debate_dates])

            days_from = _closest_debate_distance(
                self._date_df.date, debate_dates
            )

            self._days_from_df['daysFromDebate'] = days_from

        return self._days_from_df

    def build_final(self, metvi_df):
        '''
        For starters this will add a column to the _days_from_df, namely
        the number of metvi uses. After that's working I'll add two more
        columns: the number of times the metvi subject was the Democratic
        candidate and the number of times the subject was the Republican
        candidate.
        '''
        if self.final_df is None:
            # if self._days_from_df is None:
            #     self._build_days_from_df();

            # final_df = self._days_from_df.copy()
            if self._date_df is None:
                self._build_date_df();

            final_df = self._date_df.copy()
            counts = np.zeros(len(self._date_df), dtype=int)

            for idx, iatv_id in enumerate(self._date_df.iatv_id):
                final_df['n'][idx] = sum(metvi_df.include[metvi_df.iatv_id ==
                                                          iatv_id])

            self.final_df = final_df

        return final_df


def make_subject_object_table(base_df, metvi_df, year=2016):
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
    if year == 2016:
        rep = 'Donald Trump'
        rep_short = 'trump'
        dem = 'Hillary Clinton'
        dem_short = 'clinton'

    elif year == 2012:
        rep = 'Mitt Romney'
        rep_short = 'romney'
        dem = 'Barack Obama'
        dem_short = 'obama'

    rep_ts = get_tweets_ts(rep_short)
    dem_ts = get_tweets_ts(dem_short)

    full['RepTweets'] = rep_ts[full['date']].values
    full['DemTweets'] = dem_ts[full['date']].values

    return full


def main():
    years = [2016]
    # years = [2012, 2016]
    for year in years:
        d = Dataset(year)
        pe = ProjectExporter('Viomet Sep-Nov {}'.format(year))
        metvi_df = pe.export_dataframe()
        metvi_df = metvi_df[
            metvi_df.facet_word.isin(['attack', 'hit', 'beat'])
        ]
        fdf = d.build_final(metvi_df)

        fdf.to_csv('forstatmodel.csv', index=False, header=True)


if __name__ == '__main__':
    main()
