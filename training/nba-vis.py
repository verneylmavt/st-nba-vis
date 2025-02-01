#!/usr/bin/env python
# coding: utf-8

# In[246]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[247]:


pd.set_option('display.max_columns', None)


# In[248]:


df = pd.read_csv("../data/player_totals.csv")
df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})


# In[249]:


for k, v in df.items():
    print(k, v.dtype)


# In[250]:


print(df.sample(n=10, random_state=42))


# In[251]:


print(len(df))


# ### Mean 

# ```markdown
# Average (Season)
# age                26.367686
# experience          4.597473
# g                  49.530287
# gs                 24.526553
# mp               1206.888742
# fg                193.657693
# fga               439.024726
# x3p                21.563792
# x3pa               62.191800
# x2p               179.131106
# x2pa              397.396429
# ft                106.075087
# fta               142.581804
# orb                67.757644
# drb               154.040225
# trb               237.206774
# ast               111.327424
# stl                40.245328
# blk                32.275404
# tov                76.046426
# pf                117.548830
# pts               507.917058
# fg_percent          0.441109
# e_fg_percent        0.465668
# x2p_percent         0.450762
# x3p_percent         0.346730
# ft_percent          0.743960
# ```

# ```markdown
# Weighted Average (Season)
# age                26.477356
# experience          4.978421
# g                  48.348203
# gs                 22.239065
# mp               1151.493468
# fg                186.531458
# fga               412.293035
# x3p                25.739439
# x3pa               73.250154
# x2p               165.839467
# x2pa              353.407076
# ft                 95.820079
# fta               127.730736
# orb                58.944619
# drb               143.848314
# trb               215.220935
# ast               108.780964
# stl                37.530412
# blk                23.124706
# tov                70.113653
# pf                108.905131
# pts               489.574988
# fg_percent          0.452424
# e_fg_percent        0.483639
# x2p_percent         0.469259
# x3p_percent         0.351391
# ft_percent          0.750172
# ```

# ```markdown
# Average (Career)
# age              27.169443
# experience        5.000382
# g               298.873092
# gs              100.818511
# mp             6880.173473
# fg             1153.077672
# fga            2548.663359
# x3p             127.911260
# x3pa            364.014122
# x2p            1025.166412
# x2pa           2184.649237
# ft              592.329008
# fta             789.590458
# orb             311.990267
# drb             761.380344
# trb            1293.707824
# ast             672.449046
# stl             191.705916
# blk             118.125763
# tov             358.021183
# pf              673.216603
# pts            3026.395611
# ```

# ### Age vs Peformance

# In[335]:


import pandas as pd
import altair as alt
from IPython.display import display, Markdown


# In[336]:


df = pd.read_csv("../data/player_totals.csv")
df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})
df = df[df['lg'] == 'NBA']
df = df[df['season'] >= 1955]
df = df[df['g'] >= 48.34820326006421]

df['age'] = df['age'].astype('int64')


# In[337]:


df['PPG'] = df['pts'] / df['g']
df['APG'] = df['ast'] / df['g']
df['SPG'] = df['stl'] / df['g']
df['BPG'] = df['blk'] / df['g']
df['RPG'] = df['trb'] / df['g']


# In[338]:


lebron_age_stats = df[df['player'] == 'LeBron James'].copy()
lebron_age_stats.rename(columns={'player': 'Name', 'age': 'Age'}, inplace=True)
lebron_age_stats.sort_values(by='Age', inplace=True)


# In[339]:


mj_age_stats = df[df['player'] == 'Michael Jordan'].copy()
mj_age_stats.rename(columns={'player': 'Name', 'age': 'Age'}, inplace=True)
mj_age_stats.sort_values(by='Age', inplace=True)


# In[340]:


kobe_age_stats = df[df['player'] == 'Kobe Bryant'].copy()
kobe_age_stats.rename(columns={'player': 'Name', 'age': 'Age'}, inplace=True)
kobe_age_stats.sort_values(by='Age', inplace=True)


# In[341]:


grouped = df.groupby('age')
age_stats = grouped[['pts', 'ast', 'stl', 'blk', 'trb']].sum()
games_played = grouped['g'].sum()
age_stats = age_stats.div(games_played, axis=0)
age_stats = age_stats.rename(
    columns={'pts': 'PPG', 'ast': 'APG', 'stl': 'SPG', 'blk': 'BPG', 'trb': 'RPG'}
).reset_index()


# In[342]:


age_stats.rename(columns={'age': 'Age'}, inplace=True)
df.rename(columns={'player': 'Name', 'age': 'Age'}, inplace=True)


# In[343]:


max_avg_ages = {metric: age_stats.loc[age_stats[metric].idxmax(), 'Age'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
min_avg_ages = {metric: age_stats.loc[age_stats[metric].idxmin(), 'Age'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}


# In[344]:


metrics = ['PPG', 'APG', 'RPG', 'SPG', 'BPG']
titles = [
    'Points Per Game (PPG)', 
    'Assists Per Game (APG)', 
    'Rebounds Per Game (RPG)',
    'Steals Per Game (SPG)', 
    'Blocks Per Game (BPG)', 
]


# In[345]:


display(Markdown(f"## Age vs Peformance"))
display(Markdown("---"))

for metric, title in zip(metrics, titles):
    # Top-Performer
    max_metric_stats = df.groupby('Age').apply(
        lambda x: x.loc[x[metric].idxmax()]
    ).reset_index(drop=True)
    
    max_player_points = alt.Chart(max_metric_stats).mark_circle().encode(
        x=alt.X('Age:Q', title='Age'),
        y=alt.Y(f'{metric}:Q'),
        size=alt.Size(f'{metric}:Q'),
        color=alt.Color(f'{metric}:Q'),
        tooltip=['Age', 'Name', metric]
    ).interactive()

    # League Average
    average_line = alt.Chart(age_stats).mark_line().encode(
        x=alt.X('Age:Q', title='Age'),
        y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
        tooltip=['Age', metric]
    ).properties(
        title=f'Age vs {title}',
        width='container',
        height=500
    )
    
    # LeBron James
    lebron_line = alt.Chart(lebron_age_stats).mark_line(color='yellow').encode(
        x=alt.X('Age:Q', title='Age'),
        y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
        tooltip=['Age', 'Name', f'{metric}']
    ).properties(
        title=f'LeBron James: Age vs {title}'
    )
    
    # Michael Jordan
    mj_line = alt.Chart(mj_age_stats).mark_line(color='red').encode(
        x=alt.X('Age:Q', title='Age'),
        y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
        tooltip=['Age', 'Name', f'{metric}']
    ).properties(
        title=f'Michael Jordan: Age vs {title}'
    )
    
    # Kobe Bryant
    kobe_line = alt.Chart(kobe_age_stats).mark_line(color='orange').encode(
        x=alt.X('Age:Q', title='Age'),
        y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
        tooltip=['Age', 'Name', f'{metric}']
    ).properties(
        title=f'Kobe Bryant: Age vs {title}'
    )
    
    # Maximum League Average
    max_avg_point = alt.Chart(age_stats[age_stats['Age'] == max_avg_ages[metric]]).mark_point(
        size=100, color='mediumseagreen', opacity=0.7
    ).encode(
        x='Age:Q',
        y=f'{metric}:Q',
        tooltip=['Age', metric]
    )
    
    # Minimum League Average
    min_avg_point = alt.Chart(age_stats[age_stats['Age'] == min_avg_ages[metric]]).mark_point(
        size=100, color='darkred', opacity=0.7
    ).encode(
        x='Age:Q',
        y=f'{metric}:Q',
        tooltip=['Age', metric]
    )

    final_chart = (average_line + lebron_line + mj_line + kobe_line +
                   max_avg_point + min_avg_point + max_player_points).interactive()
    final_chart.display()

    display(Markdown(f"### Average 🔵"))
    age_stats_metric = age_stats[['Age', metric]]
    display(age_stats_metric.style.hide(axis='index'))
    display(Markdown("### Michael Jordan 🔴"))
    mj_age_stats_metric = mj_age_stats[['Age', metric]]
    display(mj_age_stats_metric.style.hide(axis='index'))
    display(Markdown("### Kobe Bryant 🟠"))
    kobe_age_stats_metric = kobe_age_stats[['Age', metric]]
    display(kobe_age_stats_metric.style.hide(axis='index'))
    display(Markdown("### LeBron James 🟡"))
    lebron_age_stats_metric = lebron_age_stats[['Age', metric]]
    display(lebron_age_stats_metric.style.hide(axis='index'))
    display(Markdown("### Top-Performer"))
    max_metric_stats_metric = max_metric_stats[['Age', 'Name', metric]]
    display(max_metric_stats_metric.style.hide(axis='index'))
    display(Markdown("---"))


# ### Position vs Performance

# In[291]:


import pandas as pd
import altair as alt
from IPython.display import display, Markdown


# In[292]:


df = pd.read_csv("../data/player_totals.csv")
df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})
df = df[df['lg'] == 'NBA']
df = df[df['season'] >= 1955]


# In[293]:


columns = [
        'birth_year', 'lg', 'tm', 'fg_percent', 'x3p_percent', 
        'x2p_percent', 'e_fg_percent', 'ft_percent', 'player_id'
    ]
df = df.drop(columns=columns)

agg_rules = {
    'seas_id': 'max',
    'season': 'max',
    'age': 'max',
    'experience': 'max',
    'pos': lambda x: x.mode()[0] if not x.mode().empty else None,
    'g': 'sum', 'gs': 'sum', 'mp': 'sum', 'fg': 'sum', 'fga': 'sum',
    'x3p': 'sum', 'x3pa': 'sum', 'x2p': 'sum', 'x2pa': 'sum',
    'ft': 'sum', 'fta': 'sum', 'orb': 'sum', 'drb': 'sum', 'trb': 'sum',
    'ast': 'sum', 'stl': 'sum', 'blk': 'sum', 'tov': 'sum', 'pf': 'sum', 'pts': 'sum'
}
df = df.groupby(['player'], as_index=False).agg(agg_rules)

df['fg_percent'] = (df['fg'] / df['fga']).fillna(0).round(3)
df['x3p_percent'] = (df['x3p'] / df['x3pa']).fillna(0).round(3)
df['x2p_percent'] = (df['x2p'] / df['x2pa']).fillna(0).round(3)
df['e_fg_percent'] = ((df['fg'] + 0.5 * df['x3p']) / df['fga']).fillna(0).round(3)
df['ft_percent'] = (df['ft'] / df['fta']).fillna(0).round(3)


# In[294]:


df = df[df['g'] >= 312.79118942731276]


# In[295]:


df = df[df['pos'].isin(['PG', 'SG', 'SF', 'PF', 'C'])]
numeric_cols = ['pts', 'ast', 'stl', 'blk', 'trb', 'g']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')


# In[296]:


df['PPG'] = df['pts'] / df['g']
df['APG'] = df['ast'] / df['g']
df['SPG'] = df['stl'] / df['g']
df['BPG'] = df['blk'] / df['g']
df['RPG'] = df['trb'] / df['g']


# In[297]:


df['main_pos'] = df['pos']

grouped = df.groupby('main_pos')
position_stats = grouped[['pts', 'ast', 'stl', 'blk', 'trb']].sum()
games_played = grouped['g'].sum()
position_stats = position_stats.div(games_played, axis=0)
position_stats = position_stats.rename(
    columns={'pts': 'PPG', 
            'ast': 'APG', 
            'stl': 'SPG', 
            'blk': 'BPG', 
            'trb': 'RPG'}
).reset_index()


# In[298]:


position_stats.rename(columns={'main_pos': 'Position'}, inplace=True)
df.rename(columns={'player': 'Name', 
                'pts': 'PTS', 
                'ast': 'AST', 
                'trb': 'REB',
                'stl': 'STL', 
                'blk': 'BLK'}, inplace=True)


# In[299]:


position_order = ['PG', 'SG', 'SF', 'PF', 'C']

max_avg_positions = {metric: position_stats.loc[position_stats[metric].idxmax(), 'Position'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
min_avg_positions = {metric: position_stats.loc[position_stats[metric].idxmin(), 'Position'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}


# In[300]:


metrics = ['PPG', 'APG', 'RPG', 'SPG', 'BPG']
performances = ['PTS', 'AST', 'REB', 'STL', 'BLK']
titles = [
    'Points Per Game (PPG)', 
    'Assists Per Game (APG)', 
    'Rebounds Per Game (RPG)',
    'Steals Per Game (SPG)', 
    'Blocks Per Game (BPG)', 
]


# In[301]:


display(Markdown(f"## Position vs Peformance"))
display(Markdown("---"))

for metric, performance, title in zip(metrics, performances, titles):
    base = alt.Chart(position_stats).mark_bar(opacity=0.7).encode(
        x=alt.X('Position:N', title='Position', sort=position_order),
        y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
        tooltip=['Position', f'{metric}:Q']
    ).properties(
        title=f'Position vs {title}',
        width='container',
        height=500
    )

    max_pos = max_avg_positions[metric]
    max_highlight = alt.Chart(position_stats[position_stats['Position'] == max_pos]).mark_bar(
        color='mediumseagreen', opacity=0.7
    ).encode(
        x=alt.X('Position:N', sort=position_order),
        y=alt.Y(f'{metric}:Q'),
        tooltip=['Position', f'{metric}:Q']
    )

    min_pos = min_avg_positions[metric]
    min_highlight = alt.Chart(position_stats[position_stats['Position'] == min_pos]).mark_bar(
        color='darkred', opacity=0.7
    ).encode(
        x=alt.X('Position:N', sort=position_order),
        y=alt.Y(f'{metric}:Q'),
        tooltip=['Position', f'{metric}:Q']
    )

    max_metric_stats = df.groupby('main_pos').apply(
        lambda x: x.loc[x[metric].idxmax()]
    ).reset_index(drop=True)
    max_metric_stats.rename(columns={'main_pos': 'Position'}, inplace=True)

    metric_annotations = alt.Chart(max_metric_stats).mark_text(
        dy=-10, size=10, opacity=0.7, color="black"
    ).encode(
        x=alt.X('Position:N', sort=position_order),
        y=alt.Y(f'{metric}:Q'),
        text='Name:N',
        tooltip=['Position', 'Name', f'{metric}:Q', f'{performance}:Q']
    )
    
    max_performance_stats = df.groupby('main_pos').apply(
        lambda x: x.loc[x[performance].idxmax()]
    ).reset_index(drop=True)
    max_performance_stats.rename(columns={'main_pos': 'Position'}, inplace=True)

    performance_annotations = alt.Chart(max_performance_stats).mark_text(
        dy=-10, size=10, opacity=0.7, color="black"
    ).encode(
        x=alt.X('Position:N', sort=position_order),
        y=alt.Y(f'{metric}:Q'),
        text='Name:N',
        tooltip=['Position', 'Name', f'{metric}:Q', f'{performance}:Q']
    )

    final_chart = (base + max_highlight + min_highlight + metric_annotations + performance_annotations).interactive()
    final_chart.display()
    display(Markdown("---"))


# ### Era vs Peformance

# In[363]:


import pandas as pd
import altair as alt
from IPython.display import display, Markdown


# In[364]:


df = pd.read_csv("../data/player_totals.csv")
df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})
df = df[df['lg'] == 'NBA']


# In[365]:


df = df[df['season'] >= 1970]
df['Era'] = ((df['season'] // 10) * 10).astype(int).astype(str) + 's'


# In[366]:


era_midpoints = {
    '1970s': 1975, '1980s': 1985, '1990s': 1995,
    '2000s': 2005, '2010s': 2015, '2020s': 2025
}


# In[367]:


grouped = df.groupby(['player', 'Era'])
player_era_stats = grouped[['pts', 'ast', 'stl', 'blk', 'trb']].sum()
games_played = grouped['g'].sum()
player_era_stats = player_era_stats.div(games_played, axis=0)
player_era_stats['TG'] = games_played
player_era_stats = player_era_stats.rename(
    columns={'pts': 'PPG', 'ast': 'APG', 'stl': 'SPG', 'blk': 'BPG', 'trb': 'RPG'}
).reset_index()


# In[368]:


player_era_stats['EraMidpoint'] = player_era_stats['Era'].map(era_midpoints)

player_era_stats.rename(columns={'player': 'Name'}, inplace=True)

player_era_stats = player_era_stats[
    (player_era_stats['TG'] >= (48.34820326006421 * 10)/2) | 
    ((player_era_stats['Era'] == '2020s') & (player_era_stats['TG'] >= (48.34820326006421 * 4)/2))
]


# In[369]:


grouped_era = df.groupby('Era')
era_stats = grouped_era[['pts', 'ast', 'stl', 'blk', 'trb']].sum()
games_played_era = grouped_era['g'].sum()
era_stats = era_stats.div(games_played_era, axis=0)
era_stats = era_stats.rename(
    columns={'pts': 'PPG', 'ast': 'APG', 'stl': 'SPG', 'blk': 'BPG', 'trb': 'RPG'}
).reset_index()


# In[370]:


era_stats['EraMidpoint'] = era_stats['Era'].map(era_midpoints)

era_stats.drop(era_stats[(era_stats[['Era', 'PPG', 'APG', 'SPG', 'BPG', 'RPG']] == 0).any(axis=1)].index, inplace=True)
era_stats.reset_index(drop=True, inplace=True)


# In[371]:


max_avg_eras = {metric: era_stats.loc[era_stats[metric].idxmax(), 'Era'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
min_avg_eras = {metric: era_stats.loc[era_stats[metric].idxmin(), 'Era'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}


# In[372]:


grouped_season = df.groupby('season')
season_stats = grouped_season[['pts', 'ast', 'stl', 'blk', 'trb']].sum()
games_played_season = grouped_season['g'].sum()
season_stats = season_stats.div(games_played_season, axis=0)
season_stats = season_stats.rename(
    columns={'pts': 'PPG', 'ast': 'APG', 'stl': 'SPG', 'blk': 'BPG', 'trb': 'RPG'}
).reset_index()


# In[373]:


season_stats.rename(columns={'season': 'Season'}, inplace=True)

max_avg_seasons = {metric: season_stats.loc[season_stats[metric].idxmax(), 'Season'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
min_avg_seasons = {metric: season_stats.loc[season_stats[metric].idxmin(), 'Season'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}


# In[374]:


metrics = ['PPG', 'APG', 'RPG', 'SPG', 'BPG']
titles = [
    'Points Per Game (PPG)', 
    'Assists Per Game (APG)', 
    'Rebounds Per Game (RPG)',
    'Steals Per Game (SPG)', 
    'Blocks Per Game (BPG)', 
]


# In[375]:


display(Markdown(f"## Era vs Peformance"))
display(Markdown("---"))

for metric, title in zip(metrics, titles):
    base = alt.Chart(era_stats).mark_bar(opacity=0.7).encode(
        x=alt.X('EraMidpoint:Q', title='Season', scale=alt.Scale(domain=[1970, 2030]), axis=alt.Axis(format='d')),
        y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
        tooltip=['Era', f'{metric}:Q']
    ).properties(
        title=f'Era vs {title}',
        width='container',
        height=500
    )

    max_era = max_avg_eras[metric]
    max_highlight = alt.Chart(era_stats[era_stats['Era'] == max_era]).mark_bar(
        color='mediumseagreen', opacity=0.7
    ).encode(
        x=alt.X('EraMidpoint:Q', scale=alt.Scale(domain=[1970, 2030])),
        y=alt.Y(f'{metric}:Q'),
        tooltip=['Era', f'{metric}:Q']
    )

    min_era = min_avg_eras[metric]
    min_highlight = alt.Chart(era_stats[era_stats['Era'] == min_era]).mark_bar(
        color='darkred', opacity=0.7
    ).encode(
        x=alt.X('EraMidpoint:Q', scale=alt.Scale(domain=[1970, 2030])),
        y=alt.Y(f'{metric}:Q'),
        tooltip=['Era', f'{metric}:Q']
    )
    
    averseason_line = alt.Chart(season_stats).mark_line().encode(
        x=alt.X('Season:Q', title='Season', axis=alt.Axis(format='d')),
        y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
        tooltip=['Season', metric]
    ).properties(
        title=f'Season vs {title}'
    )

    max_avg_point = alt.Chart(season_stats[season_stats['Season'] == max_avg_seasons[metric]]).mark_point(
        size=100, color='mediumseagreen', opacity=0.7
    ).encode(
        x='Season:Q',
        y=f'{metric}:Q',
        tooltip=['Season', metric]
    )
    
    min_avg_point = alt.Chart(season_stats[season_stats['Season'] == min_avg_seasons[metric]]).mark_point(
        size=100, color='darkred', opacity=0.7
    ).encode(
        x='Season:Q',
        y=f'{metric}:Q',
        tooltip=['Season', metric]
    )
    
    max_metric_stats = player_era_stats.groupby('Era').apply(
        lambda x: x.loc[x[metric].idxmax()]
    ).reset_index(drop=True)
    metric_annotations = alt.Chart(max_metric_stats).mark_text(
        dy=-10, size=10, opacity=0.7, color="black"
    ).encode(
        x=alt.X('EraMidpoint:Q', title='Season'),
        y=alt.Y(f'{metric}:Q'),
        text='Name:N',
        tooltip=['Era', 'Name', f'{metric}:Q']
    )

    final_chart = (base + max_highlight + min_highlight + averseason_line + max_avg_point + min_avg_point + metric_annotations).interactive()
    final_chart.display()
    display(Markdown("---"))


# ### Regression Model: AST → TOV

# In[214]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import optuna

import shap


# #### Data Loading

# In[215]:


df = pd.read_csv("../data/player_totals.csv")
df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})


# #### Data Preparing & Data Splitting

# In[216]:


df.dropna(subset=['ast', 'tov'], inplace=True)

X = df[['ast']]
y = df['tov']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)


# #### Hyperparameter Tuning

# In[217]:


def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'device': 'cuda',
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10)
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=0)
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    return rmse


# In[218]:


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=250)
# best_trial = study.best_trial
# best_params = best_trial.params


# In[219]:


# print("Best Hyperparameters:")
# for key, value in best_trial.params.items():
#     print(f"  {key}: {value}")


# #### Final Training

# In[220]:


best_params = {
    "max_depth": 3,
    "learning_rate": 0.011876913662158073,
    "n_estimators": 468,
    "subsample": 0.7195490366137502,
    "colsample_bytree": 0.9732519362126294,
    "lambda": 0.0012437019068289761,
    "alpha": 0.029466763625199493
}


# In[221]:


model = xgb.XGBRegressor(
    objective='reg:squarederror',
    device='cuda',
    tree_method='hist',
    **best_params
)
model.fit(X_train, y_train)


# #### Evaluation Metrics

# In[222]:


def cal_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)
    return y_pred


# In[223]:


y_pred = cal_metrics(model, X_test, y_test)


# #### Performance Visualization

# In[224]:


plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual TOV")
plt.ylabel("Predicted TOV")
plt.title("Actual vs Predicted TOV")
plt.grid(True)
plt.show()


# In[225]:


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


# #### Model Exporting

# In[226]:


X_test.to_csv('../models/ast_v_tov/X_test.csv', index=False)
y_test.to_csv('../models/ast_v_tov/y_test.csv', index=False)
pd.Series(y_pred).to_csv('../models/ast_v_tov/y_pred.csv', index=False)


# In[227]:


model.save_model("../models/ast_v_tov/model.json")


# In[228]:


model = xgb.XGBRegressor()
model.load_model("../models/ast_v_tov/model.json")


# In[229]:


# dummy_input = [('input', FloatTensorType([None, 1]))]

# onnx_model = convert_sklearn(final_model, initial_types=dummy_input)
# with open("model.onnx", "wb") as f:
#     f.write(onnx_model.SerializeToString())


# In[230]:


def predict_tov(model, ast):
    if ast:
        ast = min(ast, 30)
        ast = round(ast * 48.348203)
        ast = min(ast, 1164)
    else:
        ast = round(108.780964)
                
    input_data = pd.DataFrame({'ast': [ast]})
    output = model.predict(input_data)
    
    tov = min(output[0], 464)
    tov = round(tov / 48.348203)
    tov = min(tov, 16)
    return tov


# In[231]:


predict_tov(model, 10)


# ### Regression Model: STL & BLK → PF

# In[232]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import optuna

import shap


# #### Data Loading

# In[233]:


df = pd.read_csv("../data/player_totals.csv")
df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})


# #### Data Preparing & Data Splitting

# In[234]:


df.dropna(subset=['stl', 'blk', 'pf'], inplace=True)

X = df[['stl', 'blk']]
y = df['pf']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)


# #### Hyperparameter Tuning

# In[235]:


def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'device': 'cuda',
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10)
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=0)
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    return rmse


# In[236]:


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=250)
# best_trial = study.best_trial
# best_params = best_trial.params


# In[237]:


# print("Best Hyperparameters:")
# for key, value in best_trial.params.items():
#     print(f"  {key}: {value}")


# #### Final Training

# In[238]:


best_params = {
    "max_depth": 3,
    "learning_rate": 0.010696205798011393,
    "n_estimators": 532,
    "subsample": 0.5214618357647576,
    "colsample_bytree": 0.5392492930305198,
    "lambda": 5.417984215326458,
    "alpha": 2.929336503927947
}


# In[ ]:


model = xgb.XGBRegressor(
    objective='reg:squarederror',
    device='cuda',
    tree_method='hist',
    **best_params
)
model.fit(X_train, y_train)


# #### Evaluation Metrics

# In[240]:


def cal_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)
    return y_pred


# In[241]:


y_pred = cal_metrics(model, X_test, y_test)


# #### Performance Visualization

# In[243]:


plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual PF")
plt.ylabel("Predicted PF")
plt.title("Actual vs Predicted PF")
plt.grid(True)
plt.show()


# In[242]:


xgb.plot_importance(model)
plt.show()


# In[244]:


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


# #### Model Exporting

# In[245]:


X_test.to_csv('../models/stl_blk_v_pf/X_test.csv', index=False)
y_test.to_csv('../models/stl_blk_v_pf/y_test.csv', index=False)
pd.Series(y_pred).to_csv('../models/stl_blk_v_pf/y_pred.csv', index=False)


# In[246]:


model.save_model("../models/stl_blk_v_pf/model.json")


# In[247]:


model = xgb.XGBRegressor()
model.load_model("../models/stl_blk_v_pf/model.json")


# In[248]:


def predict_pf(model, stl, blk):
    if stl: 
        stl = min(stl, 11)
        stl = round(stl * 48.348203)
        stl = min(stl, 301)
    else:
        stl = round(37.530412)
    
    if blk:
        blk = min(blk, 17)
        blk = round(blk * 48.348203)
        blk = min(blk, 456)
    else:
        blk = round(23.124706)
        
    input_data = pd.DataFrame({'stl': [stl], 'blk': [blk]})
    output = model.predict(input_data)
    
    pf = min(output[0], 386)
    pf = round(pf / 48.348203)
    pf = min(pf, 8)
    return pf


# In[249]:


predict_pf(model, 10, 10)


# ### Classification Model: Position

# In[237]:


import joblib
import gzip
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, classification_report, confusion_matrix, ConfusionMatrixDisplay

import xgboost as xgb
import optuna

import shap
import onnxmltools
import onnxruntime as ort


# #### Data Loading

# In[211]:


df = pd.read_csv("../data/player_totals.csv")
df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})


# #### Data Preparing

# In[212]:


df = df[['pos', 
        # 'fg', 'fga', 'fg_percent', 
        # 'fg_percent', 
        # 'e_fg_percent', 
        # 'x2p', 'x2pa', 'x2p_percent',
        'x2p_percent',
        # 'x3p', 'x3pa', 'x3p_percent',
        'x3p_percent',
        # 'ft', 'fta', 'ft_percent',
        'ft_percent',
        # 'orb', 'drb', 'trb',
        'trb',
        # 'ast', 'stl', 'blk', 'tov', 'pf', 'pts',
        'ast', 'stl', 'blk', 'pts']]

df.dropna(inplace=True)


# In[213]:


df['pos'] = df['pos'].apply(lambda x: x.split('-')[0] if '-' in x else x)
df = df.astype({'pos': 'string'})

df = df[df['pos'].isin(['PG', 'SG', 'SF', 'PF', 'C'])]


# In[214]:


for k, v in df.items():
    print(k, v.dtype)


# In[215]:


print(df.sample(n=10, random_state=42))


# #### Feature Engineering

# In[216]:


plt.figure(figsize=(6,4))
sns.countplot(x='pos', data=pd.DataFrame(df))
plt.title('Label Distribution')
plt.show()


# In[217]:


label_encoder = LabelEncoder()
df['pos_encoded'] = label_encoder.fit_transform(df['pos'])

X = df.drop(columns=['pos', 'pos_encoded'])
y = df['pos_encoded']


# #### Data Splitting

# In[218]:


X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, stratify=y_val_test, random_state=42)


# #### Data Scaling

# In[219]:


scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# #### Hyperparameter Tuning

# In[220]:


# def objective(trial):
#     params = {
#         'device': 'cuda',
#         'tree_method': 'hist',
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 12),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 5),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
#     }
#     model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **params)
#     model.fit(X_train_scaled, y_train,
#             eval_set=[(X_val_scaled, y_val)], 
#             verbose=0)
#     score = model.score(X_val_scaled, y_val)
#     return score


# In[221]:


def objective(trial):
    params = {
        'objective': 'multi:softmax',
        'device': 'cuda',
        'tree_method': 'hist',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'early_stopping_rounds': 50,
    }
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **params)
    
    y_train_np = np.array(y_train)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_index, val_index in skf.split(X_train_scaled, y_train_np):
        X_fold_train, X_fold_val = X_train_scaled[train_index], X_train_scaled[val_index]
        y_fold_train, y_fold_val = y_train_np[train_index], y_train_np[val_index]
        
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=0
        )
        
        y_fold_pred = model.predict_proba(X_fold_val)
        score = log_loss(y_fold_val, y_fold_pred)
        scores.append(score)
    
    return -1 * sum(scores) / len(scores)


# In[222]:


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)
# best_trial = study.best_trial
# best_params = best_trial.params


# In[223]:


# print("Best Hyperparameters:")
# for key, value in best_trial.params.items():
#     print(f"  {key}: {value}")


# #### Final Training

# In[224]:


# best_params = {
#     "n_estimators": 998,
#     "max_depth": 4,
#     "learning_rate": 0.044988127661406536,
#     "subsample": 0.5986880261027506,
#     "colsample_bytree": 0.8840729966567867,
#     "gamma": 0.5879993012270742,
#     "reg_alpha": 0.9728734496823839,
#     "reg_lambda": 0.3493505769577766
# }

best_params = {
    "n_estimators": 699,
    "max_depth": 3,
    "learning_rate": 0.16214382011042164,
    "subsample": 0.6879927750830442,
    "colsample_bytree": 0.9552972663153627,
    "gamma": 0.08055405739393251,
    "reg_alpha": 0.46722112186774795,
    "reg_lambda": 0.605235490035828
}


# In[225]:


model = xgb.XGBClassifier(
    objective='multi:softprob',
    device = 'cuda',
    tree_method= 'hist',
    eval_metric = 'mlogloss',
    use_label_encoder = False,
    **best_params
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=0
)


# #### Evaluation Metrics

# In[226]:


def cal_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    logloss = log_loss(y_test, model.predict_proba(X_test))
    
    print("Log Loss:", logloss)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return y_pred


# In[227]:


y_pred = cal_metrics(model, X_test_scaled, y_test)


# #### Performance Visualization

# In[228]:


ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred, 
    display_labels=['PG', 'SG', 'SF', 'PF', 'C'], 
    cmap='Blues'
)

plt.title("Confusion Matrix Heatmap")
plt.show()


# In[229]:


plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(model.feature_importances_)
plt.barh(X_test.columns[sorted_idx], model.feature_importances_[sorted_idx])
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# In[ ]:


feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)


# In[245]:


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
for pos_label in np.unique(y):
    plt.scatter(
        X_pca[y == pos_label, 0],
        X_pca[y == pos_label, 1],
        label=label_encoder.inverse_transform([pos_label])[0],
        alpha=0.7
    )
plt.title("Player Positions in 2D Feature Space (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()


# #### Model Exporting

# In[195]:


X_test.to_csv('../models/mtrcs_v_pos/X_test.csv', index=False)
y_test.to_csv('../models/mtrcs_v_pos/y_test.csv', index=False)
pd.Series(y_pred).to_csv('../models/mtrcs_v_pos/y_pred.csv', index=False)
feature_importance_df.to_csv('../models/mtrcs_v_pos/feature_importance.csv', index=False)


# In[196]:


# joblib.dump(label_encoder, '../models/mtrcs_v_pos/label_encoder.pkl')


# In[197]:


label_encoder = joblib.load('../models/mtrcs_v_pos/label_encoder.pkl')


# In[198]:


# joblib.dump(scaler, '../models/mtrcs_v_pos/scaler.pkl')


# In[199]:


scaler = joblib.load('../models/mtrcs_v_pos/scaler.pkl')


# In[200]:


# model.save_model("../models/mtrcs_v_pos/model.json")

# with open("../models/mtrcs_v_pos/model.json", "rb") as f_in:
#     with gzip.open("../models/mtrcs_v_pos/model.json.gz", "wb") as f_out:
#         shutil.copyfileobj(f_in, f_out)


# In[201]:


# model = xgb.XGBClassifier()

# with gzip.open("../models/mtrcs_v_pos/model.json.gz", "rb") as f_in:
#     with open("../models/mtrcs_v_pos/model.json", "wb") as f_out:
#         shutil.copyfileobj(f_in, f_out)

# model.load_model("../models/mtrcs_v_pos/model.json")


# In[202]:


# def predict_pos(model, scaler, label_encoder, ast, blk, trb, stl, pts):
    
#     full_features = np.array([[ast, blk, trb, stl, pts]
#                             + [0.44157819853343555, 0.48112257852686874, 0.4740650651198424, 0.26494139214184087, 0.7438722775528073, 77.27333917040605, 110.99025938491846]])

#     input_scaled = scaler.transform(full_features)

#     pos_encoded = model.predict(input_scaled)

#     predicted_pos = label_encoder.inverse_transform(pos_encoded)

#     return predicted_pos[0]


# In[203]:


# initial_type = [('input', onnxmltools.convert.common.data_types.FloatTensorType([None, X_train.shape[1]]))]
# onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

# with open("../models/mtrcs_v_pos/model.onnx", "wb") as f:
#     f.write(onnx_model.SerializeToString())


# In[204]:


model = ort.InferenceSession("../models/mtrcs_v_pos/model.onnx")


# In[205]:


def predict_pos(model, scaler, label_encoder, x2p_percent, x3p_percent, ft_percent, trb, ast, stl, blk, pts):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    # print(input_name)
    # print(output_name)

    full_features = np.array([[x2p_percent, x3p_percent, ft_percent, trb, ast, stl, blk, pts]])

    input_scaled = scaler.transform(full_features)

    pos_encoded = model.run([output_name], {input_name: input_scaled.astype(np.float32)})[0]
    if len(pos_encoded.shape) == 1:  # 1D
        predicted_class_index = int(pos_encoded[0])
    else:  # 2D (e.g., Probabilities)
        predicted_class_index = np.argmax(pos_encoded, axis=1)[0]
    predicted_pos = label_encoder.inverse_transform([predicted_class_index])

    return predicted_pos[0]


# In[206]:


predicted_position = predict_pos(model, scaler, label_encoder, 0.44157819853343555, 0.48112257852686874, 0.48112257852686874, 10, 1000, 100, 10, 100)
print(f"Predicted Position: {predicted_position}")


# In[208]:


print(type(y_test))


# ### Player Similarity

# In[1]:


import os
import random
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import optuna

import shap


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()


# #### Data Loading

# In[4]:


df = pd.read_csv("../data/player_totals.csv")
df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})


# In[5]:


print(len(df))


# #### Data Preparing

# In[5]:


df['pos'] = df['pos'].apply(lambda x: x.split('-')[0] if '-' in x else x)
df = df.astype({'pos': 'string'})

df['pos'] = df['pos'].replace({
    'F': 'SF',
    'G': 'PG'
})


# In[6]:


# print(df[~df['pos'].isin(['PG', 'SG', 'SF', 'PF', 'C'])])


# In[7]:


stats_columns = [
    'player', 'season', 
    'pos', 'g', 'gs', 'mp', 
    'fg', 'fga', 'fg_percent',
    'x3p', 'x3pa', 'x3p_percent', 
    'x2p', 'x2pa', 'x2p_percent',
    'e_fg_percent', 
    'ft', 'fta', 'ft_percent', 
    'orb', 'drb', 'trb',
    'ast', 'stl', 'blk', 'tov', 'pf', 'pts'
]

df = df[stats_columns]


# In[8]:


# for k, v in df.items():
#     # print(k, v.dtype)
#     print(k)


# #### Missing Value Handling

# In[9]:


print(df[df[['player', 'season', 'pos']].isnull().any(axis=1)])


# In[10]:


df['g'] = df['g'].fillna(0)
print(df[df[['g']].isnull().any(axis=1)])


# In[11]:


df['gs'] = df['gs'].fillna((df['g'] * (24.526553 / 49.530287)).astype(int))
print(df[df[['gs']].isnull().any(axis=1)])


# In[12]:


def fill_missing_mp(group):
    group['mp'] = group['mp'].fillna(group['mp'].mean())
    return group

df = df.groupby('g').apply(fill_missing_mp).reset_index(drop=True)

print(df[df[['mp']].isnull().any(axis=1)])


# In[13]:


df['x2p'] = df['x2p'].fillna(0)
df['x2pa'] = df['x2pa'].fillna(0)
df['x2p_percent'] = df.apply(lambda row: row['x2p'] / row['x2pa'] if row['x2pa'] != 0 else 0, axis=1)


print(df[df[['x2p']].isnull().any(axis=1)])
print(df[df[['x2pa']].isnull().any(axis=1)])
print(df[df[['x2p_percent']].isnull().any(axis=1)])


# In[14]:


df['x3p'] = df['x3p'].fillna(0)
df['x3pa'] = df['x3pa'].fillna(0)
df['x3p_percent'] = df.apply(lambda row: row['x3p'] / row['x3pa'] if row['x3pa'] != 0 else 0, axis=1)

print(df[df[['x3p']].isnull().any(axis=1)])
print(df[df[['x3pa']].isnull().any(axis=1)])
print(df[df[['x3p_percent']].isnull().any(axis=1)])


# In[15]:


mismatched_fg = df[df['fg'] != (df['x2p'] + df['x3p'])]
print(mismatched_fg)
print(df[df[['fg']].isnull().any(axis=1)])


# In[16]:


mismatched_fga = df[df['fga'] != (df['x2pa'] + df['x3pa'])]
print(mismatched_fga)
print(df[df[['fga']].isnull().any(axis=1)])


# In[17]:


df['fg_percent'] = df.apply(lambda row: row['fg'] / row['fga'] if row['fga'] != 0 else 0, axis=1)
print(df[df[['fg_percent']].isnull().any(axis=1)])

mismatched_fg_percent = df[df['fg_percent'] != ((df['x2p'] + df['x3p'])/(df['x2pa'] + df['x3pa']))]
print(mismatched_fga)


# In[18]:


df['e_fg_percent'] = df.apply(
    lambda row: (row['fg'] + 0.5 * row['x3p']) / row['fga'] if row['fga'] != 0 else 0,
    axis=1
)
print(df[df[['e_fg_percent']].isnull().any(axis=1)])


# In[19]:


df['ft'] = df['ft'].fillna(0)
df['fta'] = df['fta'].fillna(0)
df['ft_percent'] = df.apply(lambda row: row['ft'] / row['fta'] if row['fta'] != 0 else 0, axis=1)

print(df[df[['ft']].isnull().any(axis=1)])
print(df[df[['fta']].isnull().any(axis=1)])
print(df[df[['ft_percent']].isnull().any(axis=1)])


# In[20]:


def fill_missing_orb(row, df):
    if np.isnan(row['orb']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['orb'].isna())
        ]
        if not filtered.empty:
            return filtered['orb'].mean()
        else:
            return np.nan
    else:
        return row['orb']

df['orb'] = df.apply(lambda row: fill_missing_orb(row, df), axis=1)

# for col in ['orb']:
#     for idx, row in df[df[col].isnull()].iterrows():
#         pos = row['pos']
#         mp = row['mp']

#         filtered = df[
#             (df['pos'] == pos) &
#             (df['mp'] >= mp - 10) & 
#             (df['mp'] <= mp + 10) &
#             (~df[col].isnull())
#         ]
        
#         if not filtered.empty:
#             mean_value = filtered[col].mean()
#             df.at[idx, col] = mean_value
            
print(df[df[['orb']].isnull().any(axis=1)])


# In[21]:


def fill_missing_drb(row, df):
    if np.isnan(row['drb']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['drb'].isna())
        ]
        if not filtered.empty:
            return filtered['drb'].mean()
        else:
            return np.nan
    else:
        return row['drb']

df['drb'] = df.apply(lambda row: fill_missing_drb(row, df), axis=1)

# for col in ['drb']:
#     for idx, row in df[df[col].isnull()].iterrows():
#         pos = row['pos']
#         mp = row['mp']

#         filtered = df[
#             (df['pos'] == pos) &
#             (df['mp'] >= mp - 10) & 
#             (df['mp'] <= mp + 10) &
#             (~df[col].isnull())
#         ]
        
#         if not filtered.empty:
#             mean_value = filtered[col].mean()
#             df.at[idx, col] = mean_value
            
print(df[df[['drb']].isnull().any(axis=1)])


# In[22]:


df['trb'] = df.apply(lambda row: row['orb'] + row['drb'], axis=1)
print(df[df[['trb']].isnull().any(axis=1)])


# In[23]:


def fill_missing_ast(row, df):
    if np.isnan(row['ast']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['ast'].isna())
        ]
        if not filtered.empty:
            return filtered['ast'].mean()
        else:
            return np.nan
    else:
        return row['ast']

df['ast'] = df.apply(lambda row: fill_missing_ast(row, df), axis=1)

# for col in ['ast']:
#     for idx, row in df[df[col].isnull()].iterrows():
#         pos = row['pos']
#         mp = row['mp']

#         filtered = df[
#             (df['pos'] == pos) &
#             (df['mp'] >= mp - 10) & 
#             (df['mp'] <= mp + 10) &
#             (~df[col].isnull())
#         ]
        
#         if not filtered.empty:
#             mean_value = filtered[col].mean()
#             df.at[idx, col] = mean_value
            
print(df[df[['ast']].isnull().any(axis=1)])


# In[24]:


def fill_missing_stl(row, df):
    if np.isnan(row['stl']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['stl'].isna())
        ]
        if not filtered.empty:
            return filtered['stl'].mean()
        else:
            return np.nan
    else:
        return row['stl']

df['stl'] = df.apply(lambda row: fill_missing_stl(row, df), axis=1)

# for col in ['stl']:
#     for idx, row in df[df[col].isnull()].iterrows():
#         pos = row['pos']
#         mp = row['mp']

#         filtered = df[
#             (df['pos'] == pos) &
#             (df['mp'] >= mp - 10) & 
#             (df['mp'] <= mp + 10) &
#             (~df[col].isnull())
#         ]
        
#         if not filtered.empty:
#             mean_value = filtered[col].mean()
#             df.at[idx, col] = mean_value
            
print(df[df[['stl']].isnull().any(axis=1)])


# In[25]:


def fill_missing_blk(row, df):
    if np.isnan(row['blk']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['blk'].isna())
        ]
        if not filtered.empty:
            return filtered['blk'].mean()
        else:
            return np.nan
    else:
        return row['blk']

df['blk'] = df.apply(lambda row: fill_missing_blk(row, df), axis=1)

# for col in ['blk']:
#     for idx, row in df[df[col].isnull()].iterrows():
#         pos = row['pos']
#         mp = row['mp']

#         filtered = df[
#             (df['pos'] == pos) &
#             (df['mp'] >= mp - 10) & 
#             (df['mp'] <= mp + 10) &
#             (~df[col].isnull())
#         ]
        
#         if not filtered.empty:
#             mean_value = filtered[col].mean()
#             df.at[idx, col] = mean_value
            
print(df[df[['blk']].isnull().any(axis=1)])


# In[26]:


def fill_missing_tov(row, df):
    if np.isnan(row['tov']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['tov'].isna())
        ]
        if not filtered.empty:
            return filtered['tov'].mean()
        else:
            return np.nan
    else:
        return row['tov']

df['tov'] = df.apply(lambda row: fill_missing_tov(row, df), axis=1)

# for col in ['tov']:
#     for idx, row in df[df[col].isnull()].iterrows():
#         pos = row['pos']
#         mp = row['mp']

#         filtered = df[
#             (df['pos'] == pos) &
#             (df['mp'] >= mp - 10) & 
#             (df['mp'] <= mp + 10) &
#             (~df[col].isnull())
#         ]
        
#         if not filtered.empty:
#             mean_value = filtered[col].mean()
#             df.at[idx, col] = mean_value
            
print(df[df[['tov']].isnull().any(axis=1)])


# In[27]:


def fill_missing_pf(row, df):
    if np.isnan(row['pf']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['pf'].isna())
        ]
        if not filtered.empty:
            return filtered['pf'].mean()
        else:
            return np.nan
    else:
        return row['pf']

df['pf'] = df.apply(lambda row: fill_missing_pf(row, df), axis=1)

# for col in ['pf']:
#     for idx, row in df[df[col].isnull()].iterrows():
#         pos = row['pos']
#         mp = row['mp']

#         filtered = df[
#             (df['pos'] == pos) &
#             (df['mp'] >= mp - 10) & 
#             (df['mp'] <= mp + 10) &
#             (~df[col].isnull())
#         ]
        
#         if not filtered.empty:
#             mean_value = filtered[col].mean()
#             df.at[idx, col] = mean_value
            
print(df[df[['pf']].isnull().any(axis=1)])


# In[28]:


def fill_missing_pts(row, df):
    if np.isnan(row['pts']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['pts'].isna())
        ]
        if not filtered.empty:
            return filtered['pts'].mean()
        else:
            return np.nan
    else:
        return row['pts']

df['pts'] = df.apply(lambda row: fill_missing_pts(row, df), axis=1)

# for col in ['pts']:
#     for idx, row in df[df[col].isnull()].iterrows():
#         pos = row['pos']
#         mp = row['mp']

#         filtered = df[
#             (df['pos'] == pos) &
#             (df['mp'] >= mp - 10) & 
#             (df['mp'] <= mp + 10) &
#             (~df[col].isnull())
#         ]
        
#         if not filtered.empty:
#             mean_value = filtered[col].mean()
#             df.at[idx, col] = mean_value
            
print(df[df[['pts']].isnull().any(axis=1)])


# In[29]:


for col in ['orb', 'drb', 'stl', 'blk', 'tov']:
    for idx, row in df[df[col].isnull()].iterrows():
        pos = row['pos']
        mp = row['mp']

        relevant_rows = df[(df['pos'] == pos) & 
                           (df['mp'] >= mp - 200) & 
                           (df['mp'] <= mp + 200) & 
                           (~df[col].isnull())]
        
        mean_value = relevant_rows[col].mean()
        
        df.at[idx, col] = mean_value


# In[30]:


df[['season', 'g', 'gs', 'mp', 'x3p', 'x3pa', 'x2p', 'x2pa', 'ft', 'fta', 'orb', 'drb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts']] = df[['season', 'g', 'gs', 'mp', 'x3p', 'x3pa', 'x2p', 'x2pa', 'ft', 'fta', 'orb', 'drb', 'ast', 'stl','blk', 'tov', 'pf', 'pts']].astype(np.int64)


# In[31]:


# df['x2p_percent'] = df['x2p'] / df['x2pa']
df['x2p_percent'] = df.apply(lambda row: row['x2p'] / row['x2pa'] if row['x2pa'] != 0 else 0, axis=1)
# df['x3p_percent'] = df['x3p'] / df['x3pa']
df['x3p_percent'] = df.apply(lambda row: row['x3p'] / row['x3pa'] if row['x3pa'] != 0 else 0, axis=1) 
df['fg'] = df['x2p'] + df['x3p']           
df['fga'] = df['x2pa'] + df['x3pa']        
# df['fg_percent'] = df['fg'] / df['fga']   
df['fg_percent'] = df.apply(lambda row: row['fg'] / row['fga'] if row['fga'] != 0 else 0, axis=1)
# df['e_fg_percent'] = (df['fg'] + (0.5 * df['x3p'])) / df['fga']
df['e_fg_percent'] = df.apply(
    lambda row: (row['fg'] + 0.5 * row['x3p']) / row['fga'] if row['fga'] != 0 else 0,
    axis=1
)
# df['ft_percent'] = df['ft'] / df['fta']
df['ft_percent'] = df.apply(lambda row: row['ft'] / row['fta'] if row['fta'] != 0 else 0, axis=1)
df['trb'] = df['orb'] + df['drb']


# In[32]:


for k, v in df.items():
    print(k, v.dtype)


# In[33]:


print(df[df.isnull().any(axis=1)])


# In[34]:


X = df.drop(['player', 'season'], axis=1)
y = df[['player', 'season']]


# #### Data Encoding

# ```markdown
# df_clean = X
# ```

# In[35]:


pos_dummies = pd.get_dummies(X['pos'], prefix='pos')

print(pos_dummies.columns.tolist())


# In[36]:


stats_columns = [
    # 'player', 'season',
    'pos', 'g', 'gs', 'mp', 
    'fg', 'fga', 'fg_percent',
    'x3p', 'x3pa', 'x3p_percent', 
    'x2p', 'x2pa', 'x2p_percent',
    'e_fg_percent', 
    'ft', 'fta', 'ft_percent', 
    'orb', 'drb', 'trb',
    'ast', 'stl', 'blk', 'tov', 'pf', 'pts'
]

num_columns = [col for col in stats_columns if col != 'pos']
X_num = X[num_columns].copy()


# In[37]:


X_combined = pd.concat([pos_dummies, X_num.reset_index(drop=True)], axis=1)


# #### Data Scaling

# In[38]:


plt.figure(figsize=(20, 10))
sns.boxplot(data=X_combined)
plt.show()


# In[39]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)


# In[40]:


pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Original Dimension:", X_scaled.shape[1], 
    "Reduced Dimension:", X_pca.shape[1])


# In[41]:


X['identifier'] = y['player'] + "_" + y['season'].astype(str)


# #### Dataset and DataLoader

# In[42]:


X_tensor = torch.tensor(X_scaled, dtype=torch.float32)


# In[43]:


dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


# #### Model Definition

# In[44]:


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder Module for Compressing Input Features
        self.encoder = nn.Sequential(
            # Fully Connected Layer for Dimensionality Reduction (Input → 64)
            nn.Linear(input_dim, 64),
            # Activation Layer for Non-Linear Transformation
            nn.ReLU(),
            # Fully Connected Layer for Dimensionality Reduction (64 → 32)
            nn.Linear(64, 32),
            # Activation Layer for Non-Linear Transformation
            nn.ReLU(),
            # Fully Connected Layer for Latent Space Representation (32 → Latent Dim)
            nn.Linear(32, latent_dim)
        )
        # Decoder Module for Reconstructing Input Features
        self.decoder = nn.Sequential(
            # Fully Connected Layer for Dimensionality Expansion (Latent Dim → 32)
            nn.Linear(latent_dim, 32),
            # Activation Layer for Non-Linear Transformation
            nn.ReLU(),
            # Fully Connected Layer for Dimensionality Expansion (32 → 64)
            nn.Linear(32, 64),
            # Activation Layer for Non-Linear Transformation
            nn.ReLU(),
            # Fully Connected Layer for Final Reconstruction (64 → Input Dim)
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        # Encoding of Input Features to Latent Representation
        latent = self.encoder(x)
        # Decoding of Latent Representation to Reconstruct Input Features
        reconstruction = self.decoder(latent)
        return reconstruction, latent


# #### Training Function

# In[45]:


def train_model(model, X_tensor, dataloader, criterion, optimizer, num_epochs=50):
    history = []
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            reconstruction, _ = model(inputs)
            loss = criterion(reconstruction, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= 32392
        history.append(epoch_loss)
        print(f"epoch: {epoch+1}/{num_epochs}, train loss: {epoch_loss:.4f}")
    
    model.eval()
    with torch.no_grad():
        _, latent_reps = model(X_tensor.to(device))
    latent_reps = latent_reps.cpu().numpy()
    
    return latent_reps, history


# #### KFold Evaluation Function

# In[46]:


def evaluate_kfold(X_scaled, input_dim, latent_dim, num_epochs=50, learning_rate=1e-3, batch_size=128, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_losses = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_scaled), 1):
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        
        train_dataset = TensorDataset(train_tensor)
        val_dataset = TensorDataset(val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        fold_model = Autoencoder(input_dim, latent_dim).to(device)
        fold_optimizer = optim.Adam(fold_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            fold_model.train()
            for batch in train_loader:
                inputs = batch[0]
                fold_optimizer.zero_grad()
                reconstruction, _ = fold_model(inputs)
                loss = criterion(reconstruction, inputs)
                loss.backward()
                fold_optimizer.step()
        
        fold_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0]
                reconstruction, _ = fold_model(inputs)
                loss = criterion(reconstruction, inputs)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_dataset)
        cv_losses.append(val_loss)
        # print(f"[CV] Fold {fold}, Validation Loss: {val_loss:.4f}")
    
    return cv_losses


# #### Hyperparameter Tuning

# In[47]:


def objective(trial):
    latent_dim = trial.suggest_int("latent_dim", 8, 32)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    num_epochs = trial.suggest_int("num_epochs", 30, 70)
    input_dim = X_scaled.shape[1]
    
    cv_losses = evaluate_kfold(X_scaled, input_dim, latent_dim,
                                     num_epochs=num_epochs,
                                     learning_rate=learning_rate,
                                     batch_size=128,
                                     n_splits=3)
    mean_cv_loss = np.mean(cv_losses)
    return mean_cv_loss


# In[48]:


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=20)
# best_trial = study.best_trial
# best_params = best_trial.params


# In[49]:


# print("Best Hyperparameters:")
# for key, value in best_trial.params.items():
#     print(f"  {key}: {value}")


# In[50]:


best_params = {
    "latent_dim": 25,
    "learning_rate": 0.007262494401279974,
    "num_epochs": 59
}


# #### Final Training

# In[51]:


latent_dim = best_params["latent_dim"]
lr = best_params["learning_rate"]
num_epochs = best_params["num_epochs"]
input_dim = X_scaled.shape[1]

model = Autoencoder(input_dim, latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# In[52]:


latent_reps, history = train_model(model, X_tensor, dataloader, criterion, optimizer, num_epochs=num_epochs)


# In[53]:


epochs_range = range(1, num_epochs + 1)
plt.figure(figsize=(6, 4))

plt.plot(epochs_range, history, label='train loss', linestyle='-', color='#2a7db8')
plt.xlabel('epoch')
plt.ylabel('value')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

plt.show()


# #### Performance Visualization

# #### Model Exporting

# In[666]:


# joblib.dump(latent_reps, '../models/player_sims/latent_reps.pkl')


# In[6]:


latent_reps = joblib.load('../models/player_sims/latent_reps.pkl')


# In[668]:


# player_season = X[['player', 'season']]
# player_season.to_csv('../models/player_sims/player_season.csv', index=False)


# In[669]:


# player_season = pd.read_csv('../models/player_sims/player_season.csv')


# In[670]:


# X['player'] = y['player']
# X['season'] = y['season']

# X = X.drop('identifier', axis=1)


# In[671]:


# X.to_csv('../models/player_sims/X.csv', index=False)


# In[7]:


X = pd.read_csv('../models/player_sims/X.csv')


# In[8]:


knn = NearestNeighbors(n_neighbors=75, metric='euclidean', algorithm='auto')
knn.fit(latent_reps)


# In[13]:


# def find_similar_players(player_name, season, embeddings, knn, X, top_n, allow_same_player, max_other_player):
#     query_df = X[(X['player'] == player_name) & (X['season'] == season)]
#     if query_df.empty:
#         print(f"No record found for {player_name} in season {season}.")
#         return pd.DataFrame(), pd.DataFrame()
    
#     query_idx = query_df.index[0]
#     current_neighbors = max(top_n + 5, 50)
#     similar = []
#     player_count = {}

#     while len(similar) < top_n:
#         knn.set_params(n_neighbors=current_neighbors)
#         distances, indices = knn.kneighbors(embeddings[query_idx].reshape(1, -1))
        
#         for i, dist in zip(indices[0], distances[0]):
#             if len(similar) >= top_n:
#                 break
#             if i == query_idx:
#                 continue
#             if not allow_same_player and X.iloc[i]['player'] == player_name:
#                 continue
            
#             player_name_i = X.iloc[i]['player']
#             season_i = X.iloc[i]['season']
#             constructed_identifier = f"{player_name_i}_{season_i}"
            
#             if constructed_identifier in [s[0] for s in similar]:
#                 continue
#             if max_other_player is not None:
#                 if player_name_i != player_name and player_count.get(player_name_i, 0) >= max_other_player:
#                     continue
#             similar.append((i, dist))
#             if player_name_i != player_name:
#                 player_count[player_name_i] = player_count.get(player_name_i, 0) + 1
        
#         current_neighbors += 5
    
#     result_df = X.iloc[[idx for idx, _ in similar]].copy()
#     result_df['Distance'] = [dist for _, dist in similar]
    
#     n_samples = len(similar) + 1
#     perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
#     all_embeddings = np.vstack([embeddings[query_idx], embeddings[[idx for idx, _ in similar]]])
    
#     tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
#     reduced_embeddings = tsne.fit_transform(all_embeddings)
    
#     visualization_df = pd.DataFrame({
#         'x': reduced_embeddings[:, 0],
#         'y': reduced_embeddings[:, 1],
#         'Title': pd.concat(
#             [pd.Series([f"{player_name} ({season})"]), 
#              X.iloc[[idx for idx, _ in similar]]['player'] + " (" + X.iloc[[idx for idx, _ in similar]]['season'].astype(str)+ ")"]
#         ).reset_index(drop=True),
#         'Similarity': pd.concat(
#             [pd.Series([100.0]), 
#              pd.Series([100 - dist for _, dist in similar])]
#         ).reset_index(drop=True)
#     })
    
#     return result_df.reset_index(drop=True), visualization_df


# In[20]:


# similar_list, vis_df = find_similar_players("Magic Johnson", 1987, latent_reps, knn, X, top_n=20, allow_same_player=False, max_other_player=2)


# In[17]:


def predict_sim_players(player_name, season, embeddings, knn, X, top_n, allow_same_player=False, max_other_player=1):
    query_df = X[(X['player'] == player_name) & (X['season'] == season)]
    if query_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    query_idx = query_df.index[0]
    current_neighbors = max(top_n + 5, 50)
    similar = []
    player_count = {}

    while len(similar) < top_n:
        knn.set_params(n_neighbors=current_neighbors)
        distances, indices = knn.kneighbors(embeddings[query_idx].reshape(1, -1))
        
        for i, dist in zip(indices[0], distances[0]):
            if len(similar) >= top_n:
                break
            if i == query_idx:
                continue
            if not allow_same_player and X.iloc[i]['player'] == player_name:
                continue
            
            player_name_i = X.iloc[i]['player']
            season_i = X.iloc[i]['season']
            constructed_identifier = f"{player_name_i}_{season_i}"
            
            if constructed_identifier in [s[0] for s in similar]:
                continue
            if max_other_player is not None:
                if player_name_i != player_name and player_count.get(player_name_i, 0) >= max_other_player:
                    continue
            similar.append((i, dist))
            if player_name_i != player_name:
                player_count[player_name_i] = player_count.get(player_name_i, 0) + 1
        
        current_neighbors += 5
    
    result_df = X.iloc[[idx for idx, _ in similar]].copy()
    result_df['distance'] = [dist for _, dist in similar]
    
    return result_df.reset_index(drop=True), similar, query_idx


# In[28]:


def visualize_sim_players(player_name, season, embeddings, similar, query_idx):
    n_samples = len(similar) + 1
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    all_embeddings = np.vstack([embeddings[query_idx], embeddings[[idx for idx, _ in similar]]])
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_embeddings = tsne.fit_transform(all_embeddings)
    
    visualization_df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'Title': pd.concat(
            [pd.Series([f"{player_name} ({season})"]), 
            X.iloc[[idx for idx, _ in similar]]['player'] + " (" + X.iloc[[idx for idx, _ in similar]]['season'].astype(str)+ ")"]
        ).reset_index(drop=True),
        'Similarity': pd.concat(
            [pd.Series([100.0]), 
            pd.Series([100 - dist for _, dist in similar])]
        ).reset_index(drop=True)
    })
    return visualization_df


# In[19]:


result_df, similar, query_idx = predict_sim_players("Michael Jordan", 1990, latent_reps, knn, X, 10)


# In[29]:


visualization_df = visualize_sim_players("Michael Jordan", 1990, latent_reps, similar, query_idx)


# In[11]:


display(result_df)


# In[12]:


display(visualization_df)


# ### Player Archetypes

# In[108]:


import json
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import onnxruntime as ort


# In[ ]:


sns.set(style="whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


def set_seed(seed=42):
    # random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
set_seed()


# #### Data Loading

# In[55]:


df = pd.read_csv("../data/player_totals.csv")
df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})


# #### Data Preparing

# In[56]:


df['pos'] = df['pos'].apply(lambda x: x.split('-')[0] if '-' in x else x)
df = df.astype({'pos': 'string'})


# In[57]:


label_encoder = joblib.load('../models/mtrcs_v_pos/label_encoder.pkl')
scaler = joblib.load('../models/mtrcs_v_pos/scaler.pkl')
model = ort.InferenceSession("../models/mtrcs_v_pos/model.onnx")


def predict_pos(model, scaler, label_encoder, x2p_percent, x3p_percent, ft_percent, trb, ast, stl, blk, pts):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    full_features = np.array([[x2p_percent, x3p_percent, ft_percent, trb, ast, stl, blk, pts]])

    input_scaled = scaler.transform(full_features)

    pos_encoded = model.run([output_name], {input_name: input_scaled.astype(np.float32)})[0]
    if len(pos_encoded.shape) == 1:  # 1D
        predicted_class_index = int(pos_encoded[0])
    else:  # 2D (e.g., Probabilities)
        predicted_class_index = np.argmax(pos_encoded, axis=1)[0]
    predicted_pos = label_encoder.inverse_transform([predicted_class_index])

    return predicted_pos[0]


def update_pos(df, model, scaler, label_encoder):
    mask = df['pos'].isin(['F', 'G'])
    df.loc[mask, 'pos'] = df[mask].apply(
        lambda row: predict_pos(
            model,
            scaler,
            label_encoder,
            row['x2p_percent'],
            row['x3p_percent'],
            row['ft_percent'],
            row['trb'],
            row['ast'],
            row['stl'],
            row['blk'],
            row['pts']
        ),
        axis=1
    )
    return df


df = update_pos(df, model, scaler, label_encoder)


# In[58]:


# print(df[~df['pos'].isin(['PG', 'SG', 'SF', 'PF', 'C'])])


# In[59]:


stats_columns = [
    'player', 'season', 
    'pos', 'g', 'gs', 'mp', 
    'fg', 'fga', 'fg_percent',
    'x3p', 'x3pa', 'x3p_percent', 
    'x2p', 'x2pa', 'x2p_percent',
    'e_fg_percent', 
    'ft', 'fta', 'ft_percent', 
    'orb', 'drb', 'trb',
    'ast', 'stl', 'blk', 'tov', 'pf', 'pts'
]

df = df[stats_columns]


# #### Missing Value Handling

# In[60]:


df['g'] = df['g'].fillna(0)
df['gs'] = df['gs'].fillna((df['g'] * (24.526553 / 49.530287)).astype(int))

def fill_missing_mp(group):
    group['mp'] = group['mp'].fillna(group['mp'].mean())
    return group
df = df.groupby('g').apply(fill_missing_mp).reset_index(drop=True)


# In[61]:


df['x2p'] = df['x2p'].fillna(0)
df['x2pa'] = df['x2pa'].fillna(0)
df['x2p_percent'] = df.apply(lambda row: row['x2p'] / row['x2pa'] if row['x2pa'] != 0 else 0, axis=1)


# In[62]:


df['x3p'] = df['x3p'].fillna(0)
df['x3pa'] = df['x3pa'].fillna(0)
df['x3p_percent'] = df.apply(lambda row: row['x3p'] / row['x3pa'] if row['x3pa'] != 0 else 0, axis=1)


# In[63]:


df['fg_percent'] = df.apply(lambda row: row['fg'] / row['fga'] if row['fga'] != 0 else 0, axis=1)

df['e_fg_percent'] = df.apply(
    lambda row: (row['fg'] + 0.5 * row['x3p']) / row['fga'] if row['fga'] != 0 else 0,
    axis=1
)


# In[64]:


df['ft'] = df['ft'].fillna(0)
df['fta'] = df['fta'].fillna(0)
df['ft_percent'] = df.apply(lambda row: row['ft'] / row['fta'] if row['fta'] != 0 else 0, axis=1)


# In[65]:


def fill_missing_orb(row, df):
    if np.isnan(row['orb']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['orb'].isna())
        ]
        if not filtered.empty:
            return filtered['orb'].mean()
        else:
            return np.nan
    else:
        return row['orb']

df['orb'] = df.apply(lambda row: fill_missing_orb(row, df), axis=1)


# In[66]:


def fill_missing_drb(row, df):
    if np.isnan(row['drb']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['drb'].isna())
        ]
        if not filtered.empty:
            return filtered['drb'].mean()
        else:
            return np.nan
    else:
        return row['drb']

df['drb'] = df.apply(lambda row: fill_missing_drb(row, df), axis=1)


# In[67]:


df['trb'] = df.apply(lambda row: row['orb'] + row['drb'], axis=1)


# In[68]:


def fill_missing_ast(row, df):
    if np.isnan(row['ast']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['ast'].isna())
        ]
        if not filtered.empty:
            return filtered['ast'].mean()
        else:
            return np.nan
    else:
        return row['ast']

df['ast'] = df.apply(lambda row: fill_missing_ast(row, df), axis=1)


# In[69]:


def fill_missing_stl(row, df):
    if np.isnan(row['stl']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['stl'].isna())
        ]
        if not filtered.empty:
            return filtered['stl'].mean()
        else:
            return np.nan
    else:
        return row['stl']

df['stl'] = df.apply(lambda row: fill_missing_stl(row, df), axis=1)


# In[70]:


def fill_missing_blk(row, df):
    if np.isnan(row['blk']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['blk'].isna())
        ]
        if not filtered.empty:
            return filtered['blk'].mean()
        else:
            return np.nan
    else:
        return row['blk']

df['blk'] = df.apply(lambda row: fill_missing_blk(row, df), axis=1)


# In[71]:


def fill_missing_tov(row, df):
    if np.isnan(row['tov']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['tov'].isna())
        ]
        if not filtered.empty:
            return filtered['tov'].mean()
        else:
            return np.nan
    else:
        return row['tov']

df['tov'] = df.apply(lambda row: fill_missing_tov(row, df), axis=1)


# In[72]:


def fill_missing_pf(row, df):
    if np.isnan(row['pf']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['pf'].isna())
        ]
        if not filtered.empty:
            return filtered['pf'].mean()
        else:
            return np.nan
    else:
        return row['pf']

df['pf'] = df.apply(lambda row: fill_missing_pf(row, df), axis=1)


# In[73]:


def fill_missing_pts(row, df):
    if np.isnan(row['pts']):
        filtered = df[
            (df['pos'] == row['pos']) &
            (df['mp'] >= row['mp'] - 10) & 
            (df['mp'] <= row['mp'] + 10) &
            (~df['pts'].isna())
        ]
        if not filtered.empty:
            return filtered['pts'].mean()
        else:
            return np.nan
    else:
        return row['pts']

df['pts'] = df.apply(lambda row: fill_missing_pts(row, df), axis=1)


# In[74]:


print(df[df.isnull().any(axis=1)])


# In[75]:


for col in ['orb', 'drb', 'stl', 'blk', 'tov']:
    for idx, row in df[df[col].isnull()].iterrows():
        pos = row['pos']
        mp = row['mp']

        relevant_rows = df[(df['pos'] == pos) & 
                           (df['mp'] >= mp - 200) & 
                           (df['mp'] <= mp + 200) & 
                           (~df[col].isnull())]
        
        mean_value = relevant_rows[col].mean()
        
        df.at[idx, col] = mean_value


# In[76]:


df[['season', 'g', 'gs', 'mp', 'x3p', 'x3pa', 'x2p', 'x2pa', 'ft', 'fta', 'orb', 'drb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts']] = df[['season', 'g', 'gs', 'mp', 'x3p', 'x3pa', 'x2p', 'x2pa', 'ft', 'fta', 'orb', 'drb', 'ast', 'stl','blk', 'tov', 'pf', 'pts']].astype(np.int64)


# In[77]:


# df['x2p_percent'] = df['x2p'] / df['x2pa']
df['x2p_percent'] = df.apply(lambda row: row['x2p'] / row['x2pa'] if row['x2pa'] != 0 else 0, axis=1)
# df['x3p_percent'] = df['x3p'] / df['x3pa']
df['x3p_percent'] = df.apply(lambda row: row['x3p'] / row['x3pa'] if row['x3pa'] != 0 else 0, axis=1) 
df['fg'] = df['x2p'] + df['x3p']           
df['fga'] = df['x2pa'] + df['x3pa']        
# df['fg_percent'] = df['fg'] / df['fga']   
df['fg_percent'] = df.apply(lambda row: row['fg'] / row['fga'] if row['fga'] != 0 else 0, axis=1)
# df['e_fg_percent'] = (df['fg'] + (0.5 * df['x3p'])) / df['fga']
df['e_fg_percent'] = df.apply(
    lambda row: (row['fg'] + 0.5 * row['x3p']) / row['fga'] if row['fga'] != 0 else 0,
    axis=1
)
# df['ft_percent'] = df['ft'] / df['fta']
df['ft_percent'] = df.apply(lambda row: row['ft'] / row['fta'] if row['fta'] != 0 else 0, axis=1)
df['trb'] = df['orb'] + df['drb']


# In[78]:


for k, v in df.items():
    print(k, v.dtype)


# In[79]:


print(df[df.isnull().any(axis=1)])


# #### Data Preparing (Derived Metric)

# In[80]:


base_columns = [
    'player', 'season', 'mp', 'pts', 'fg', 'fga', 'fg_percent', 'x3p', 'x3pa',
    'x3p_percent', 'x2p', 'x2pa', 'x2p_percent', 'e_fg_percent', 'ft', 'fta',
    'ft_percent', 'ast', 'tov', 'stl', 'blk', 'orb', 'drb', 'trb'
]

data = df[base_columns].copy()


# In[81]:


data = data[data['mp'] > 0].reset_index(drop=True)


# In[82]:


data['ppm'] = data['pts'] / data['mp']
data['fgm_pm'] = data['fg'] / data['mp']
data['x3p_pm'] = data['x3p'] / data['mp']
data['x3pa_pm'] = data['x3pa'] / data['mp']
data['x2p_pm'] = data['x2p'] / data['mp']
data['x2pa_pm'] = data['x2pa'] / data['mp']
data['ast_pm'] = data['ast'] / data['mp']
data['ast_tov'] = data['ast'] / (data['tov'] + 1e-5)
data['stl_pm'] = data['stl'] / data['mp']
data['blk_pm'] = data['blk'] / data['mp']
data['trb_pm'] = data['trb'] / data['mp']
data['orb_pm'] = data['orb'] / data['mp']
data['drb_pm'] = data['drb'] / data['mp']
data['ft_pm'] = data['ft'] / data['mp']


# #### Data Scaling

# In[83]:


features = [
    'ppm', 'fgm_pm', 'x3p_pm', 'x3pa_pm', 'x2p_pm', 'x2pa_pm', 'fg_percent',
    'x3p_percent', 'x2p_percent', 'e_fg_percent', 'ft_percent',
    'ast_pm', 'ast_tov', 'stl_pm', 'blk_pm', 'trb_pm', 'orb_pm', 'drb_pm'
]

X = data[features].copy()


# In[84]:


X.fillna(X.mean(), inplace=True)


# In[99]:


y = data[['player', 'season']]


# In[85]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# #### Dimensionality Reduction

# In[86]:


pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(X_scaled)


# In[87]:


pca_vis = PCA(n_components=2, random_state=42)
X_pca_vis = pca_vis.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], alpha=0.5)
plt.title("Principal Component Analysis (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# #### Model

# In[88]:


def run_kmeans(n_clusters, data_for_cluster):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(data_for_cluster)
    return labels, kmeans.inertia_


# In[89]:


range_n_clusters = list(range(2, 26))
inertias = []
silhouette_scores = []

for n_clusters in range_n_clusters:
    labels, inertia = run_kmeans(n_clusters, X_pca)
    inertias.append(inertia)
    s_score = silhouette_score(X_pca, labels)
    silhouette_scores.append(s_score)
    print(f"n_clusters: {n_clusters}, inertia: {inertia:.2f}, silhouette: {s_score:.3f}")


# In[90]:


plt.figure(figsize=(8, 4))
plt.plot(range_n_clusters, inertias, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(range_n_clusters, silhouette_scores, marker='o', color='green')
plt.title("Silhouette Scores for Various k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.show()


# In[91]:


optimal_k = 25
final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
final_labels = final_kmeans.fit_predict(X_pca)


# In[92]:


data['cluster_label'] = final_labels

print("Cluster Label Counts:")
print(data['cluster_label'].value_counts())


# #### Performance Visualization

# In[93]:


cluster_to_archetype = {
    0:  "PG-Scoring Machine: Sharpshooting (primary), Defending (secondary)",
    1:  "PG-Offensive Threat 1: Sharpshooting, Slashing",
    2:  "PG-Offensive Threat 2: Slashing Shot, Creating",
    3:  "PG-Offensive Threat 3: Sharpshooting, Playmaking",
    4:  "PG-Playmaking Shot Creator: Playmaking, Shot Creating",
    5:  "SG-Scoring Machine: Sharpshooting (primary), Sharpshooting (secondary)",
    6:  "SG-2-Way Sharpshooter: Defending, Slashing",
    7:  "SG-Inside Out Playmaker: Sharpshooting, Playmaking",
    8:  "SG-Playmaking Shot Creator: Playmaking, Shot Creating",
    9:  "SG-Shot Creator: Defending, Shot Creating",
    10: "SF-Offensive Threat: Sharpshooting (primary), Playmaking (secondary)",
    11: "SF-3-Level Scorer: Sharpshooting, Slashing",
    12: "SF-2-Way 3-Point Facilitator 1: Defending, Sharpshooting",
    13: "SF-2-Way 3-Point Facilitator 2: Defending, Slashing",
    14: "SF-2-Way Finisher: Slashing, Rim Protector",
    15: "PF-Post Facilitator: Slashing (primary), Defending (secondary)",
    16: "PF-2-Way 3-Level Scorer: Playmaking, Slashing",
    17: "PF-Stretch Glass Cleaner: Post Scoring, Glass Cleaning",
    18: "PF-Mid-Range Facilitator: Defending, Sharpshooting",
    19: "PF-Glass Cleaning Lockdown: Glass Cleaning, Defending",
    20: "C-Paint Beast 1: Slashing Glass (primary), Cleaning (secondary)",
    21: "C-Paint Beast 2: Rim Protector, Slashing",
    22: "C-2-Way Facilitator: Post Scoring, Rim Protector",
    23: "C-Glass Cleaning Finisher: Slashing, Sharpshooting",
    24: "C-Stretch Glass Cleaner: Sharpshooting, Glass Cleaning"
}


# In[94]:


plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca_vis[:,0], y=X_pca_vis[:,1],
                hue=data['cluster_label'].astype(str),
                palette='viridis', legend='full', alpha=0.7)
plt.title("Final Clustering Visualization on PCA (2 Components)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# In[95]:


sil_score = silhouette_score(X_pca, final_labels)
db_score = davies_bouldin_score(X_pca, final_labels)
ch_score = calinski_harabasz_score(X_pca, final_labels)

print("Final Clustering Evaluation Metrics:")
print(f"Silhouette Score: {sil_score:.3f}")
print(f"Davies-Bouldin Score: {db_score:.3f}")
print(f"Calinski-Harabasz Score: {ch_score:.2f}")


# In[96]:


cluster_summary = data.groupby('cluster_label')[features + ['ppm', 'fgm_pm', 'ast_tov']].mean().T

print("Cluster Feature Averages:")
display(cluster_summary)


# In[97]:


plt.figure(figsize=(14, 10))
sns.heatmap(cluster_summary, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Cluster Feature Averages - Heatmap")
plt.xlabel("Cluster Label")
plt.ylabel("Features")
plt.show()


# #### Prediction

# In[100]:


X["player"] = y["player"]
X["season"] = y["season"]


# In[ ]:


def pred_archetypes(input_metrics):
    input_df = pd.DataFrame([input_metrics], columns=features)
    input_scaled = scaler.transform(input_df)
    input_pca = pca.transform(input_scaled)
    cluster_label = final_kmeans.predict(input_pca)[0]
    archetype = cluster_to_archetype.get(cluster_label, f"Cluster {cluster_label}")
    return archetype


# In[107]:


sample_input = {feat: X.loc[(X['player'] == "Shaquille O'Neal") & (X['season'] == 2002), feat].iloc[0] for feat in features}
print(json.dumps(sample_input, indent=4))
predicted_archetype = pred_archetypes(sample_input)
print("Predicted Archetype:", predicted_archetype)

