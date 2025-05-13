import pandas as pd
import nfl_data_py as nfl

def get_data():
    seasons = nfl.import_seasonal_data(range(2000, 2025), 'REG')
    id_name = nfl.import_ids()
    id_name.dropna(axis=0, subset=['gsis_id', 'name'], inplace=True)
    id_name = id_name[['gsis_id', 'name', 'position']]
    seasons = seasons.merge(id_name, how='right', left_on='player_id', right_on='gsis_id')
    seasons = seasons[seasons['position'].isin(['QB', 'RB', 'WR', 'TE'])]
    seasons.rename({'season':'year'}, axis=1, inplace=True)
    seasons.dropna(axis=0, subset='year', inplace=True)
    seasons['year'] = seasons['year'].astype(int)
    return seasons

def prep_data_clustering(data, position, aggfunc, subset, panelvar):
    prep_data = data[data['position']==position]

    if subset != 'all':
        prep_data = prep_data[subset+[panelvar]]

    if prep_data[panelvar].dtype == object:
        prep_data = pd.concat([prep_data.select_dtypes(exclude=object).apply(lambda x: x.fillna(x.agg(aggfunc))), prep_data[panelvar]], axis=1)
    else:
        prep_data = prep_data.select_dtypes(exclude=object).apply(lambda x: x.fillna(x.agg(aggfunc)))

    return prep_data.groupby(panelvar).agg(aggfunc).reset_index(drop=True)
