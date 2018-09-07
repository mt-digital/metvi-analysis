import pandas as pd

from correlate_tweets import data_for_model
from projects.viomet.analysis import fit_all_networks
from projects.common.analysis import get_project_data_frame

# data = data_for_model()

csv_fields = ['state', 'trump', 'clinton', 'metvi_all']

project_df = get_project_data_frame('Data/viomet-sep-nov-2016.csv')

date_range = pd.date_range('2016-9-1', '2016-11-30', freq='D')

fit = fit_all_networks(
    project_df, date_range, by_network=False, verbose=True)

print('done!')
