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
