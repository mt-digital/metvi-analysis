##
# Function to create the data for statistical modeling in R.
#
# Ideally the final version of tabular data to be analyzed for Results section
# of the viomet paper being submitted to Political Communication.
#
# Date: 12/28/2018
#

from datetime import datetime, timedelta
from metacorps.app.models import IatvDocument

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

    return documents(id__nin=remove_list)


class Dataset:

    def __init__(self, year=2016):
        '''
        Just pass year, since this is specific for viomet. Year is either 2012
        or 2016, for each election year.
        '''
        self.year = year
        d1 = datetime(year, 9, 1)
        d2 = datetime(year, 11, 30, 23, 59, 59)
        self.documents = _remove_reruns(
            IatvDocument.objects(
                start_time__gte=d1,
                start_time__lte=d2,
                program_name__in=SHOWS
            )
        )

    @classmethod
    def build_final(cls):
        if cls.final is not None:
            return cls.final
        else:
            cls.final = 'something TBD'
            return cls.final

    def _build_date_n(self):
        pass
