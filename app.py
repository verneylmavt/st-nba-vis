import os
import joblib

import streamlit as st
from streamlit_extras.mention import mention
from streamlit_extras.echo_expander import echo_expander
# from streamlit_searchbox import st_searchbox

import numpy as np
import pandas as pd
import altair as alt

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import xgboost as xgb
import torch
import torch.nn as nn

# import shap

import onnxruntime as ort



# ----------------------
# Page Function (Charts)
# ----------------------

def load_df_time():
    df = pd.read_csv("data/player_totals.csv")
    df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})
    df = df[df['lg'] == 'NBA']
    df = df[df['season'] >= 1955]
    return df



def load_df_player():
    df = pd.read_csv("data/player_totals.csv")
    df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})
    df = df[df['lg'] == 'NBA']
    df = df[df['season'] >= 1955]

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
    return df



def pos_v_perf():
    st.header("Position vs Performance")
    
    df = load_df_player().copy()
    df = df[df['g'] >= 312.79118942731276]
    
    df = df[df['pos'].isin(['PG', 'SG', 'SF', 'PF', 'C'])]
    numeric_cols = ['pts', 'ast', 'stl', 'blk', 'trb', 'g']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df['PPG'] = df['pts'] / df['g']  # Points per Game
    df['APG'] = df['ast'] / df['g']  # Assists per Game
    df['SPG'] = df['stl'] / df['g']  # Steals per Game
    df['BPG'] = df['blk'] / df['g']  # Blocks per Game
    df['RPG'] = df['trb'] / df['g']  # Rebounds per Game

    df['main_pos'] = df['pos']

    # position_stats = df.groupby('main_pos').apply(
    #     lambda group: pd.Series({
    #         'PPG': group['pts'].sum() / group['g'].sum(),
    #         'APG': group['ast'].sum() / group['g'].sum(),
    #         'SPG': group['stl'].sum() / group['g'].sum(),
    #         'BPG': group['blk'].sum() / group['g'].sum(),
    #         'RPG': group['trb'].sum() / group['g'].sum(),
    #     })
    # ).reset_index()
    
    grouped = df.groupby('main_pos')
    position_stats = grouped[['pts', 'ast', 'stl', 'blk', 'trb']].sum()
    games_played = grouped['g'].sum()
    position_stats = position_stats.div(games_played, axis=0)
    position_stats = position_stats.rename(
        columns={'pts': 'PPG', 'ast': 'APG', 'stl': 'SPG', 'blk': 'BPG', 'trb': 'RPG'}
    ).reset_index()


    position_stats.rename(columns={'main_pos': 'Position'}, inplace=True)
    df.rename(columns={'player': 'Name', 'pts': 'PTS', 'ast': 'AST', 'trb': 'REB', 'stl': 'STL', 'blk': 'BLK'}, inplace=True)

    position_order = ['PG', 'SG', 'SF', 'PF', 'C']

    max_avg_positions = {metric: position_stats.loc[position_stats[metric].idxmax(), 'Position'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
    min_avg_positions = {metric: position_stats.loc[position_stats[metric].idxmin(), 'Position'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}

    metrics = ['PPG', 'APG', 'RPG', 'SPG', 'BPG']
    performances = ['PTS', 'AST', 'REB', 'STL', 'BLK']
    titles = [
        'Points Per Game (PPG)', 
        'Assists Per Game (APG)', 
        'Rebounds Per Game (RPG)',
        'Steals Per Game (SPG)', 
        'Blocks Per Game (BPG)', 
    ]
    
    for metric, performance, title in zip(metrics, performances, titles):
        with st.container(border=True):
            base = alt.Chart(position_stats).mark_bar(opacity=0.7).encode(
                x=alt.X('Position:N', title='Position', sort=position_order),
                y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
                tooltip=['Position', f'{metric}:Q']
            ).properties(
                title=f'Position vs {title}'
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
                dy=-10, size=10, opacity=0.7, color="whitesmoke"
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
                dy=-10, size=10, opacity=0.7, color="whitesmoke"
            ).encode(
                x=alt.X('Position:N', sort=position_order),
                y=alt.Y(f'{metric}:Q'),
                text='Name:N',
                tooltip=['Position', 'Name', f'{metric}:Q', f'{performance}:Q']
            )

            final_chart = (base + max_highlight + min_highlight + metric_annotations + performance_annotations).interactive()
            st.altair_chart(final_chart, use_container_width=True, theme="streamlit")



# def age_v_perf():
#     st.header("Age vs Performance")
    
#     df = load_df_time().copy()
#     df = df[df['g'] >= 48.34820326006421]
    
#     df['PPG'] = df['pts'] / df['g']  # Points per Game
#     df['APG'] = df['ast'] / df['g']  # Assists per Game
#     df['SPG'] = df['stl'] / df['g']  # Steals per Game
#     df['BPG'] = df['blk'] / df['g']  # Blocks per Game
#     df['RPG'] = df['trb'] / df['g']  # Rebounds per Game
#     # df.dropna(subset=['Age', 'PPG', 'APG', 'SPG', 'BPG', 'RPG', 'player'], inplace=True)
    
#     lebron_age_stats = (df[df['player'] == 'LeBron James']).copy()
#     lebron_age_stats.rename(columns={'player': 'Name', 'age': 'Age'}, inplace=True)
    
    
#     # age_stats = df.groupby('age').apply(
#     #     lambda group: pd.Series({
#     #         'PPG': group['pts'].sum() / group['g'].sum(),  # Points per Game
#     #         'APG': group['ast'].sum() / group['g'].sum(),  # Assists per Game
#     #         'SPG': group['stl'].sum() / group['g'].sum(),  # Steals per Game
#     #         'BPG': group['blk'].sum() / group['g'].sum(),  # Blocks per Game
#     #         'RPG': group['trb'].sum() / group['g'].sum(),  # Rebounds per Game
#     #     })
#     # ).reset_index()
    
#     grouped = df.groupby('age')
#     age_stats = grouped[['pts', 'ast', 'stl', 'blk', 'trb']].sum()
#     games_played = grouped['g'].sum()
#     age_stats = age_stats.div(games_played, axis=0)
#     age_stats = age_stats.rename(
#         columns={'pts': 'PPG', 'ast': 'APG', 'stl': 'SPG', 'blk': 'BPG', 'trb': 'RPG'}
#     ).reset_index()

    
#     age_stats.rename(columns={'age': 'Age'}, inplace=True)
#     df.rename(columns={'player': 'Name', 'age': 'Age'}, inplace=True)
    
#     max_avg_ages = {metric: age_stats.loc[age_stats[metric].idxmax(), 'Age'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
#     min_avg_ages = {metric: age_stats.loc[age_stats[metric].idxmin(), 'Age'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}

#     metrics = ['PPG', 'APG', 'RPG', 'SPG', 'BPG']
#     titles = [
#         'Points Per Game (PPG)', 
#         'Assists Per Game (APG)', 
#         'Rebounds Per Game (RPG)',
#         'Steals Per Game (SPG)', 
#         'Blocks Per Game (BPG)', 
#     ]

#     for metric, title in zip(metrics, titles):
#         max_metric_stats = df.groupby('Age').apply(
#             lambda x: x.loc[x[metric].idxmax()]
#         ).reset_index(drop=True)

#         average_line = alt.Chart(age_stats).mark_line().encode(
#             x=alt.X('Age:Q', title='Age'),
#             y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
#             tooltip=['Age', metric]
#         ).properties(
#             title=f'Age vs {title}'
#         )
        
#         lebron_line = alt.Chart(lebron_age_stats).mark_line(color='purple').encode(
#             x=alt.X('Age:Q', title='Age'),
#             y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
#             tooltip=['Age', f'{metric}']
#         ).properties(
#             title=f'LeBron James: Age vs {title}'
#         )

#         # max_avg_value = age_stats.loc[age_stats['Age'] == max_avg_ages[metric], metric].values[0]
#         max_avg_point = alt.Chart(age_stats[age_stats['Age'] == max_avg_ages[metric]]).mark_point(
#             size=100, color='mediumseagreen', opacity=0.7
#         ).encode(
#             x='Age:Q',
#             y=f'{metric}:Q',
#             tooltip=['Age', metric]
#         )
        
#         # min_avg_value = age_stats.loc[age_stats['Age'] == min_avg_ages[metric], metric].values[0]
#         min_avg_point = alt.Chart(age_stats[age_stats['Age'] == min_avg_ages[metric]]).mark_point(
#             size=100, color='darkred', opacity=0.7
#         ).encode(
#             x='Age:Q',
#             y=f'{metric}:Q',
#             tooltip=['Age', metric]
#         )
        
#         max_player_points = alt.Chart(max_metric_stats).mark_circle().encode(
#             x=alt.X('Age:Q', title='Age'),
#             y=alt.Y(f'{metric}:Q'),
#             size=alt.Size(f'{metric}:Q'),
#             color=alt.Color(f'{metric}:Q'),
#             tooltip=['Age', 'Name', metric]
#         ).interactive()

#         final_chart = (average_line + lebron_line + max_avg_point + min_avg_point + max_player_points).interactive()
#         st.altair_chart(final_chart, use_container_width=True, theme="streamlit")



def age_v_perf():
    st.header("Age vs Performance")
    
    df = load_df_time().copy()
    df = df[df['g'] >= 48.34820326006421]
    
    df['PPG'] = df['pts'] / df['g']  # Points per Game
    df['APG'] = df['ast'] / df['g']  # Assists per Game
    df['SPG'] = df['stl'] / df['g']  # Steals per Game
    df['BPG'] = df['blk'] / df['g']  # Blocks per Game
    df['RPG'] = df['trb'] / df['g']  # Rebounds per Game
    
    lebron_age_stats = df[df['player'] == 'LeBron James'].copy()
    lebron_age_stats.rename(columns={'player': 'Name', 'age': 'Age'}, inplace=True)
    lebron_age_stats.sort_values(by='Age', inplace=True)
    
    mj_age_stats = df[df['player'] == 'Michael Jordan'].copy()
    mj_age_stats.rename(columns={'player': 'Name', 'age': 'Age'}, inplace=True)
    mj_age_stats.sort_values(by='Age', inplace=True)
    
    kobe_age_stats = df[df['player'] == 'Kobe Bryant'].copy()
    kobe_age_stats.rename(columns={'player': 'Name', 'age': 'Age'}, inplace=True)
    kobe_age_stats.sort_values(by='Age', inplace=True)
    
    grouped = df.groupby('age')
    age_stats = grouped[['pts', 'ast', 'stl', 'blk', 'trb']].sum()
    games_played = grouped['g'].sum()
    age_stats = age_stats.div(games_played, axis=0)
    age_stats = age_stats.rename(
        columns={'pts': 'PPG', 'ast': 'APG', 'stl': 'SPG', 'blk': 'BPG', 'trb': 'RPG'}
    ).reset_index()
    
    age_stats.rename(columns={'age': 'Age'}, inplace=True)
    df.rename(columns={'player': 'Name', 'age': 'Age'}, inplace=True)
    
    max_avg_ages = {metric: age_stats.loc[age_stats[metric].idxmax(), 'Age'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
    min_avg_ages = {metric: age_stats.loc[age_stats[metric].idxmin(), 'Age'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
    
    metrics = ['PPG', 'APG', 'RPG', 'SPG', 'BPG']
    titles = [
        'Points Per Game (PPG)', 
        'Assists Per Game (APG)', 
        'Rebounds Per Game (RPG)',
        'Steals Per Game (SPG)', 
        'Blocks Per Game (BPG)', 
    ]
    
    for metric, title in zip(metrics, titles):
        with st.container(border=True):
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
                title=f'Age vs {title}'
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
            
            final_chart = (average_line + lebron_line + mj_line + kobe_line + max_avg_point + min_avg_point + max_player_points).interactive()
            st.altair_chart(final_chart, use_container_width=True, theme="streamlit")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.write("Average 🔵")
                age_stats_metric = age_stats[['Age', metric]]
                st.dataframe(age_stats_metric, hide_index=True, use_container_width=True)
            with col2:
                st.write("Michael Jordan 🔴")
                mj_age_stats_metric = mj_age_stats[['Age', metric]]
                st.dataframe(mj_age_stats_metric, hide_index=True, use_container_width=True)
            with col3:
                st.write("Kobe Bryant 🟠")
                kobe_age_stats_metric = kobe_age_stats[['Age', metric]]
                st.dataframe(kobe_age_stats_metric, hide_index=True, use_container_width=True)
            with col4:
                st.write("LeBron James 🟡")
                lebron_age_stats_metric = lebron_age_stats[['Age', metric]]
                st.dataframe(lebron_age_stats_metric, hide_index=True, use_container_width=True)
            with col5:
                st.write("Top-Performer ⭐")
                max_metric_stats_metric = max_metric_stats[['Age', 'Name', metric]]
                st.dataframe(max_metric_stats_metric, hide_index=True, use_container_width=True)



# def era_v_perf():
#     st.header("Era vs Performance")

#     df = load_df_time().copy()

#     df['Era'] = ((df['season'] // 10) * 10).astype(int).astype(str) + 's'
#     df = df[df['Era'].isin(['1970s', '1980s', '1990s', '2000s', '2010s', '2020s'])]

#     player_era_stats = df.groupby(['player', 'Era']).apply(
#         lambda group: pd.Series({
#             'PPG': group['pts'].sum() / group['g'].sum(),
#             'APG': group['ast'].sum() / group['g'].sum(),
#             'SPG': group['stl'].sum() / group['g'].sum(),
#             'BPG': group['blk'].sum() / group['g'].sum(),
#             'RPG': group['trb'].sum() / group['g'].sum(),
#             'TG': group['g'].sum()
#         })
#     ).reset_index()

#     player_era_stats.rename(columns={'player': 'Name'}, inplace=True)

#     player_era_stats = player_era_stats[
#         (player_era_stats['TG'] >= (48.34820326006421 * 10)/2) | 
#         ((player_era_stats['Era'] == '2020s') & (player_era_stats['TG'] >= (48.34820326006421 * 4)/2))
#     ]

#     era_stats = df.groupby('Era').apply(
#         lambda group: pd.Series({
#             'PPG': group['pts'].sum() / group['g'].sum(),
#             'APG': group['ast'].sum() / group['g'].sum(),
#             'SPG': group['stl'].sum() / group['g'].sum(),
#             'BPG': group['blk'].sum() / group['g'].sum(),
#             'RPG': group['trb'].sum() / group['g'].sum(),
#         })
#     ).reset_index()

#     era_stats.drop(era_stats[(era_stats[['Era', 'PPG', 'APG', 'SPG', 'BPG', 'RPG']] == 0).any(axis=1)].index, inplace=True)
#     era_stats.reset_index(drop=True, inplace=True)
#     max_avg_eras = {metric: era_stats.loc[era_stats[metric].idxmax(), 'Era'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
#     min_avg_eras = {metric: era_stats.loc[era_stats[metric].idxmin(), 'Era'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
    
#     season_stats = df.groupby('season').apply(
#         lambda group: pd.Series({
#             'PPG': group['pts'].sum() / group['g'].sum(),  # Points per Game
#             'APG': group['ast'].sum() / group['g'].sum(),  # Assists per Game
#             'SPG': group['stl'].sum() / group['g'].sum(),  # Steals per Game
#             'BPG': group['blk'].sum() / group['g'].sum(),  # Blocks per Game
#             'RPG': group['trb'].sum() / group['g'].sum(),  # Rebounds per Game
#         })
#     ).reset_index()
    
#     season_stats.rename(columns={'season': 'Season'}, inplace=True)
#     df.rename(columns={'player': 'Name', 'season': 'Season'}, inplace=True)
    
#     max_avg_seasons = {metric: season_stats.loc[season_stats[metric].idxmax(), 'Season'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
#     min_avg_seasons = {metric: season_stats.loc[season_stats[metric].idxmin(), 'Season'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}

#     metrics = ['PPG', 'APG', 'RPG', 'SPG', 'BPG']
#     titles = [
#         'Points Per Game (PPG)', 
#         'Assists Per Game (APG)', 
#         'Rebounds Per Game (RPG)',
#         'Steals Per Game (SPG)', 
#         'Blocks Per Game (BPG)', 
#     ]

#     for metric, title in zip(metrics, titles):
#         base = alt.Chart(era_stats).mark_bar(opacity=0.7).encode(
#             x=alt.X('Era:N', title='Era', sort=sorted(era_stats['Era'].unique())),
#             y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
#             tooltip=['Era', f'{metric}:Q']
#         ).properties(
#             title=f'Era vs {title}'
#         )

#         max_era = max_avg_eras[metric]
#         max_highlight = alt.Chart(era_stats[era_stats['Era'] == max_era]).mark_bar(
#             color='mediumseagreen', opacity=0.7
#         ).encode(
#             x=alt.X('Era:N', sort=sorted(era_stats['Era'].unique())),
#             y=alt.Y(f'{metric}:Q'),
#             tooltip=['Era', f'{metric}:Q']
#         )

#         min_era = min_avg_eras[metric]
#         min_highlight = alt.Chart(era_stats[era_stats['Era'] == min_era]).mark_bar(
#             color='darkred', opacity=0.7
#         ).encode(
#             x=alt.X('Era:N', sort=sorted(era_stats['Era'].unique())),
#             y=alt.Y(f'{metric}:Q'),
#             tooltip=['Era', f'{metric}:Q']
#         )
        
#         averseason_line = alt.Chart(season_stats).mark_line().encode(
#             x=alt.X('Season:Q', title='Season'),
#             y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
#             tooltip=['Season', metric]
#         ).properties(
#             title=f'Season vs {title}'
#         )

#         max_avg_point = alt.Chart(season_stats[season_stats['Season'] == max_avg_seasons[metric]]).mark_point(
#             size=100, color='mediumseagreen', opacity=0.7
#         ).encode(
#             x='Season:Q',
#             y=f'{metric}:Q',
#             tooltip=['Season', metric]
#         )
        
#         min_avg_point = alt.Chart(season_stats[season_stats['Season'] == min_avg_seasons[metric]]).mark_point(
#             size=100, color='darkred', opacity=0.7
#         ).encode(
#             x='Season:Q',
#             y=f'{metric}:Q',
#             tooltip=['Season', metric]
#         )

#         max_metric_stats = player_era_stats.groupby('Era').apply(
#             lambda x: x.loc[x[metric].idxmax()]
#         ).reset_index(drop=True)
#         metric_annotations = alt.Chart(max_metric_stats).mark_text(
#             dy=-10, size=10, opacity=0.7, color="whitesmoke"
#         ).encode(
#             x=alt.X('Era:N', sort=sorted(era_stats['Era'].unique())),
#             y=alt.Y(f'{metric}:Q'),
#             text='Name:N',
#             tooltip=['Era', 'Name', f'{metric}:Q']
#         )

#         final_chart = (base + max_highlight + min_highlight + metric_annotations).interactive()
#         st.altair_chart(final_chart, use_container_width=True, theme="streamlit")



def era_v_perf():
    st.header("Era vs Performance")

    df = load_df_time().copy()
    df = df[df['season'] >= 1970]
    df['Era'] = ((df['season'] // 10) * 10).astype(int).astype(str) + 's'

    era_midpoints = {
        '1970s': 1975, '1980s': 1985, '1990s': 1995,
        '2000s': 2005, '2010s': 2015, '2020s': 2025
    }

    # player_era_stats = df.groupby(['player', 'Era']).apply(
    #     lambda group: pd.Series({
    #         'PPG': group['pts'].sum() / group['g'].sum(),
    #         'APG': group['ast'].sum() / group['g'].sum(),
    #         'SPG': group['stl'].sum() / group['g'].sum(),
    #         'BPG': group['blk'].sum() / group['g'].sum(),
    #         'RPG': group['trb'].sum() / group['g'].sum(),
    #         'TG': group['g'].sum()
    #     })
    # ).reset_index()
    
    grouped = df.groupby(['player', 'Era'])
    player_era_stats = grouped[['pts', 'ast', 'stl', 'blk', 'trb']].sum()
    games_played = grouped['g'].sum()
    player_era_stats = player_era_stats.div(games_played, axis=0)
    player_era_stats['TG'] = games_played
    player_era_stats = player_era_stats.rename(
        columns={'pts': 'PPG', 'ast': 'APG', 'stl': 'SPG', 'blk': 'BPG', 'trb': 'RPG'}
    ).reset_index()
    
    
    player_era_stats['EraMidpoint'] = player_era_stats['Era'].map(era_midpoints)

    player_era_stats.rename(columns={'player': 'Name'}, inplace=True)

    player_era_stats = player_era_stats[
        (player_era_stats['TG'] >= (48.34820326006421 * 10)/2) | 
        ((player_era_stats['Era'] == '2020s') & (player_era_stats['TG'] >= (48.34820326006421 * 4)/2))
    ]

    # era_stats = df.groupby('Era').apply(
    #     lambda group: pd.Series({
    #         'PPG': group['pts'].sum() / group['g'].sum(),
    #         'APG': group['ast'].sum() / group['g'].sum(),
    #         'SPG': group['stl'].sum() / group['g'].sum(),
    #         'BPG': group['blk'].sum() / group['g'].sum(),
    #         'RPG': group['trb'].sum() / group['g'].sum(),
    #     })
    # ).reset_index()
    
    grouped_era = df.groupby('Era')
    era_stats = grouped_era[['pts', 'ast', 'stl', 'blk', 'trb']].sum()
    games_played_era = grouped_era['g'].sum()
    era_stats = era_stats.div(games_played_era, axis=0)
    era_stats = era_stats.rename(
        columns={'pts': 'PPG', 'ast': 'APG', 'stl': 'SPG', 'blk': 'BPG', 'trb': 'RPG'}
    ).reset_index()


    era_stats['EraMidpoint'] = era_stats['Era'].map(era_midpoints)

    era_stats.drop(era_stats[(era_stats[['Era', 'PPG', 'APG', 'SPG', 'BPG', 'RPG']] == 0).any(axis=1)].index, inplace=True)
    era_stats.reset_index(drop=True, inplace=True)

    max_avg_eras = {metric: era_stats.loc[era_stats[metric].idxmax(), 'Era'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
    min_avg_eras = {metric: era_stats.loc[era_stats[metric].idxmin(), 'Era'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
    
    # season_stats = df.groupby('season').apply(
    #     lambda group: pd.Series({
    #         'PPG': group['pts'].sum() / group['g'].sum(),
    #         'APG': group['ast'].sum() / group['g'].sum(),
    #         'SPG': group['stl'].sum() / group['g'].sum(),
    #         'BPG': group['blk'].sum() / group['g'].sum(),
    #         'RPG': group['trb'].sum() / group['g'].sum(),
    #     })
    # ).reset_index()
    
    grouped_season = df.groupby('season')
    season_stats = grouped_season[['pts', 'ast', 'stl', 'blk', 'trb']].sum()
    games_played_season = grouped_season['g'].sum()
    season_stats = season_stats.div(games_played_season, axis=0)
    season_stats = season_stats.rename(
        columns={'pts': 'PPG', 'ast': 'APG', 'stl': 'SPG', 'blk': 'BPG', 'trb': 'RPG'}
    ).reset_index()


    season_stats.rename(columns={'season': 'Season'}, inplace=True)

    max_avg_seasons = {metric: season_stats.loc[season_stats[metric].idxmax(), 'Season'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
    min_avg_seasons = {metric: season_stats.loc[season_stats[metric].idxmin(), 'Season'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}

    metrics = ['PPG', 'APG', 'RPG', 'SPG', 'BPG']
    titles = [
        'Points Per Game (PPG)', 
        'Assists Per Game (APG)', 
        'Rebounds Per Game (RPG)',
        'Steals Per Game (SPG)', 
        'Blocks Per Game (BPG)', 
    ]

    for metric, title in zip(metrics, titles):
        with st.container(border=True):
            base = alt.Chart(era_stats).mark_bar(opacity=0.7).encode(
                x=alt.X('EraMidpoint:Q', title='Season', scale=alt.Scale(domain=[1970, 2030]), axis=alt.Axis(format='d')),
                y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
                tooltip=['Era', f'{metric}:Q']
            ).properties(
                title=f'Era vs {title}'
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
                dy=-10, size=10, opacity=0.7, color="whitesmoke"
            ).encode(
                x=alt.X('EraMidpoint:Q', title='Season'),
                y=alt.Y(f'{metric}:Q'),
                text='Name:N',
                tooltip=['Era', 'Name', f'{metric}:Q']
            )

            final_chart = (base + max_highlight + min_highlight + averseason_line + max_avg_point + min_avg_point + metric_annotations).interactive()
            st.altair_chart(final_chart, use_container_width=True, theme="streamlit")
        
        

# ----------------------
# Loading Function
# ----------------------

def load_xgb_reg_model(model_name):
    model = xgb.XGBRegressor()
    model.load_model(os.path.join("models", model_name, "model.json"))
    return model

def load_onnx_model(model_name):
    model =  ort.InferenceSession(os.path.join("models", model_name, "model.onnx"))
    return model

def load_joblib_object(object_name, model_name):
    object = joblib.load(os.path.join("models", model_name, object_name))
    return object

def load_xgb_data(model_name):
    X_test = pd.read_csv(os.path.join("models", model_name, "X_test.csv"))
    y_test = pd.read_csv(os.path.join("models", model_name, "y_test.csv"))
    y_pred = pd.read_csv(os.path.join("models", model_name, "y_pred.csv"))
    return X_test, y_test, y_pred

def load_df_csv(csv_name, model_name):
    df = pd.read_csv(os.path.join("models", model_name, csv_name))
    return df

def load_test_pred_chart(y_test, y_pred, text):
    data = pd.concat([y_test, y_pred], axis=1)
    actual_col = y_test.columns[0]
    predicted_col = y_pred.columns[0]

    data = data.rename(columns={
        actual_col: f'Actual {text}',
        predicted_col: f'Predicted {text}'
    })
    
    scatter = alt.Chart(data).mark_circle(opacity=0.5).encode(
        x=alt.X(f'Actual {text}', title=f'Actual {text}'),
        y=alt.Y(f'Predicted {text}', title=f'Predicted {text}'),
        tooltip=[f'Actual {text}', f'Predicted {text}']
    ).properties(
        # title=f"Prediction",
        width=600,
        height=400
    )

    y_equals_x = alt.Chart(pd.DataFrame({
        f'Actual {text}': [data[f'Actual {text}'].min(), data[f'Actual {text}'].max()],
        f'Predicted {text}': [data[f'Actual {text}'].min(), data[f'Actual {text}'].max()]
    })).mark_line(color='white').encode(
        x=f'Actual {text}',
        y=f'Predicted {text}'
    )

    chart = (scatter + y_equals_x).configure_axis(grid=True)
    return chart

def load_tree_explainer_chart(shap_df, X_test):
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_test)
    
    # shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    
    shap_melted = shap_df.melt(var_name="Feature", value_name="SHAP Value")
    shap_melted["Feature Value"] = pd.concat([X_test[col] for col in X_test.columns]).reset_index(drop=True)
    shap_melted["Feature Value"] = pd.to_numeric(shap_melted["Feature Value"], errors='coerce')

    chart = alt.Chart(shap_melted).mark_circle(size=50, opacity=0.7).encode(
        x=alt.X("SHAP Value", title="SHAP Value"),
        y=alt.Y("Feature:N", title="Feature", sort=None),
        color=alt.Color("Feature Value", title="Feature Value", scale=alt.Scale(scheme='redblue')),
        tooltip=["Feature", "Feature Value", "SHAP Value"]
    ).properties(
        # title="Tree Explainer",
        width=800,
        height=400
    ).configure_axis(
        grid=False
    )
    return chart

def load_cm_chart(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels).reset_index().melt(
        id_vars='index', var_name='Predicted', value_name='Count'
    ).rename(columns={'index': 'Actual'})

    heatmap = alt.Chart(cm_df).mark_rect().encode(
        x=alt.X('Predicted:O', title='Predicted', sort=labels),
        y=alt.Y('Actual:O', title='Actual', sort=labels),
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), title='Count'),
        tooltip=['Actual', 'Predicted', 'Count']
    ).properties(
        width=500,
        height=500
    )
    text = heatmap.mark_text(baseline='middle').encode(
        text=alt.Text('Count:Q', format='.0f'),
        color=alt.condition(
            alt.datum.Count > 0.5 * max(cm_df['Count']), 
            alt.value('white'), 
            alt.value('black')
        )
    )
    return heatmap, text

def load_feat_imp_chart(feature_importance):
    bar_chart = alt.Chart(feature_importance).mark_bar().encode(
        x=alt.X('Importance:Q', title='Importance'),
        y=alt.Y('Feature:N', sort='-x', title='Feature'),
        tooltip=['Feature', 'Importance']
    ).properties(
        # title="Feature Importances",
        width=600,
        height=400
    )
    return bar_chart

def load_pca_chart(pca):
    chart = alt.Chart(pca).mark_circle(size=60).encode(
        x="Principal Component 1",
        y="Principal Component 2",
        color=alt.Color("Position:N", sort=['PG', 'SG', 'SF', 'PF', 'C']),
        tooltip=["Principal Component 1", "Principal Component 2", "Position"]
    ).properties(
        # title="Player Positions in 2D Feature Space (PCA)",
        width=700,
        height=500
    )
    return chart



# ----------------------
# Page Function (Models)
# ----------------------

def ast_v_tov():
    # st.divider()
    st.header("Assist → Turnover")
    
    model = load_xgb_reg_model("ast_v_tov")
    X_test, y_test, y_pred = load_xgb_data("ast_v_tov")
    X_test = X_test.rename(columns={'ast': 'AST'})
    test_pred_chart = load_test_pred_chart(y_test, y_pred, "Turnovers")
    shap_df = load_df_csv("shap_df.csv", "ast_v_tov")
    tree_explainer_chart = load_tree_explainer_chart(shap_df, X_test)
    
    
    # ----------------------
    # Prediction Function
    # ----------------------
    
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
    
    with st.form(key='ast_v_tov_form'):
        ast = st.slider("Assists", 0, 30, 5)
        # st.info("""NBA Record: Scott Skiles (Orlando Magic) vs Denver Nuggets on 30 December 1990 w/ 30 Assists""", icon="💡")
        st.info("""NBA Record (1 Game): Scott Skiles w/ 30 Assists""", icon="💡")
        submit_button = st.form_submit_button(label='Predict')
        
        if submit_button:
            tov = predict_tov(model, ast)
            st.write(f"Predicted Turnovers: {tov}")
    
    st.feedback("thumbs")
    mention(
            label="GitHub Repo: verneylmavt/st-nba-vis",
            icon="github",
            url="https://github.com/verneylmavt/st-nba-vis"
        )
    mention(
            label="Other ML Tasks",
            icon="streamlit",
            url="https://verneylogyt.streamlit.app/"
        )
    st.divider()
    
    st.subheader("""Model""")
    with echo_expander(code_location="below", label="Code"):
        import xgboost as xgb
        
        xgb.XGBRegressor(
            objective='reg:squarederror',
            device='cuda',
            tree_method='hist',
            max_depth=3,
            learning_rate=0.011876913662158073,
            n_estimators=468,
            subsample=0.7195490366137502,
            colsample_bytree=0.9732519362126294,
            reg_alpha=0.029466763625199493,
            reg_lambda=0.0012437019068289761
        )
        
    st.subheader("""Prediction""")
    st.altair_chart(test_pred_chart, use_container_width=True, theme="streamlit")
    
    st.subheader("""Tree Explainer""")
    st.altair_chart(tree_explainer_chart, use_container_width=True, theme="streamlit")
    
    st.subheader("""Evaluation Metrics""")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", "34.9254", border=True)
    col2.metric("MAE", "23.5824", border=True)
    col3.metric("R²", "0.7258", border=True)
    
    st.divider()
    st.subheader("""Top 100 Assist-to-Turnover Players""")
    top_100_ast_tov_1980 = pd.read_csv(os.path.join("models", "ast_v_tov", "top_100_ast-tov_1980.csv"))
    top_100_ast_tov_1980.rename(columns={'player': 'Name', 'pos':'Position', 'g':'Games', 'ast': 'AST', 'tov': 'TO', 'ast/tov':'AST/TO Ratio'}, inplace=True)
    st.dataframe(top_100_ast_tov_1980, hide_index=True, use_container_width=True)



def stl_blk_v_pf():
    st.header("Steal & Block → Personal Foul")
    
    model = load_xgb_reg_model("stl_blk_v_pf")
    X_test, y_test, y_pred = load_xgb_data("stl_blk_v_pf")
    X_test = X_test.rename(columns={'stl': 'STL', 'blk': 'BLK'})
    test_pred_chart = load_test_pred_chart(y_test, y_pred, "Personal Fouls")
    shap_df = load_df_csv("shap_df.csv", "stl_blk_v_pf")
    tree_explainer_chart = load_tree_explainer_chart(shap_df, X_test)
    
    
    # ----------------------
    # Prediction Function
    # ----------------------
    
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
    
    with st.form(key='stl_blk_v_pf_form'):
        stl = st.slider("Steals", 0, 11, 3)
        # st.info("""NBA Record: Kendall Gill (New Jersey Nets) vs Miami Heat on 3 April 1999 w/ 11 Steals""", icon="💡")
        st.info("""NBA Record (1 Game): Kendall Gill w/ 11 Steals""", icon="💡")
        blk = st.slider("Blocks", 0, 17, 4)
        # st.info("""NBA Record: Elmore Smith (Los Angeles Lakers) vs Portland Trail Blazers on 28 October 1973 w/ 17 Blocks""", icon="💡")
        st.info("""NBA Record (1 Game): Elmore Smith w/ 17 Blocks""", icon="💡")
        submit_button = st.form_submit_button(label='Predict')
        
        if submit_button:
            pf = predict_pf(model, stl, blk)
            st.write(f"Predicted Personal Fouls: {pf}")
            
    st.feedback("thumbs")
    mention(
            label="GitHub Repo: verneylmavt/st-nba-vis",
            icon="github",
            url="https://github.com/verneylmavt/st-nba-vis"
        )
    mention(
            label="Other ML Tasks",
            icon="streamlit",
            url="https://verneylogyt.streamlit.app/"
        )
    st.divider()
    
    st.subheader("""Model""")
    with echo_expander(code_location="below", label="Code"):
        import xgboost as xgb
        
        xgb.XGBRegressor(
            objective='reg:squarederror',
            device='cuda',
            tree_method='hist',
            max_depth=3,
            learning_rate=0.010696205798011393,
            n_estimators=532,
            subsample=0.5214618357647576,
            colsample_bytree=0.5392492930305198,
            reg_alpha=2.929336503927947,
            reg_lambda=5.417984215326458
        )
        
    st.subheader("""Prediction""")
    st.altair_chart(test_pred_chart, use_container_width=True, theme="streamlit")
    st.subheader("""Tree Explainer""")
    st.altair_chart(tree_explainer_chart, use_container_width=True, theme="streamlit")
    
    st.subheader("""Evaluation Metrics""")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", "36.2078", border=True)
    col2.metric("MAE", "26.0388", border=True)
    col3.metric("R²", "0.7931", border=True)
    
    # st.divider()
    


def mtrcs_v_pos():
    st.header("Performance → Position")
    
    model = load_onnx_model("mtrcs_v_pos")
    
    X_test, y_test, y_pred = load_xgb_data("mtrcs_v_pos")
    y_test['pos_encoded'] = y_test['pos_encoded'].map({0: 'PG', 1: 'SG', 2: 'SF', 3: 'PF', 4: 'C'})
    y_pred['0'] = y_pred['0'].map({0: 'PG', 1: 'SG', 2: 'SF', 3: 'PF', 4: 'C'})
    
    feature_importance = load_df_csv("feature_importance.csv", "mtrcs_v_pos")
    feature_importance['Feature'] = feature_importance['Feature'].replace({
        'ast': 'AST',
        'blk': 'BLK',
        'trb': 'REB',
        'stl': 'STL',
        'pts': 'PTS',
        'x3p_percent': '3P%',
        'x2p_percent': '2P%',
        'ft_percent': 'FT%'
    })
    pca = load_df_csv("pca.csv", "mtrcs_v_pos")
    
    label_encoder = load_joblib_object("label_encoder.pkl", "mtrcs_v_pos")
    scaler = load_joblib_object("scaler.pkl", "mtrcs_v_pos")
    
    pca_chart = load_pca_chart(pca)
    cm_chart = load_cm_chart(y_test, y_pred, ['PG', 'SG', 'SF', 'PF', 'C'])
    feat_imp_chart = load_feat_imp_chart(feature_importance)
    
    # df = load_df_time().copy()
    # df = df[['pos', 'x2p_percent', 'x3p_percent', 'ft_percent', 'trb', 'ast', 'stl', 'blk', 'pts']]
    # df.dropna(inplace=True)
    # df['pos'] = df['pos'].apply(lambda x: x.split('-')[0] if '-' in x else x)
    # df = df.astype({'pos': 'string'})
    # df = df[df['pos'].isin(['PG', 'SG', 'SF', 'PF', 'C'])]
    # df['pos_encoded'] = label_encoder.fit_transform(df['pos'])
    # X = df.drop(columns=['pos', 'pos_encoded'])
    # y = df['pos_encoded']
    
    
    # ----------------------
    # Prediction Function
    # ----------------------
    
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
    
    
    with st.form(key='mtrcs_v_pos_form'):
        pts = st.slider("Points", 500, 4029, 1000)
        st.info("""NBA Record (1 Season): Wilt Chamberlain w/ 4,029 Points""", icon="💡")
        ast = st.slider("Assists", 125, 1164, 200)
        st.info("""NBA Record (1 Season): John Stockton w/ 1,164 Assists""", icon="💡")
        trb = st.slider("Rebounds", 250, 2149, 300)
        st.info("""NBA Record (1 Season): Wilt Chamberlain w/ 2,149 Rebounds""", icon="💡")
        stl = st.slider("Steals", 50, 301, 100)
        st.info("""NBA Record (1 Season): Alvin Robertson w/ 301 Steals""", icon="💡")
        blk = st.slider("Blocks", 50, 456, 100)
        st.info("""NBA Record (1 Season): Mark Eaton w/ 456 Blocks""", icon="💡")
        x2p_percent = st.slider("2-Pt Field Goal Percentage (%)", 0.000, 72.727, 45.000)
        st.info("""NBA Record (1 Season): Wilt Chamberlain w/ 72.7%""", icon="💡")
        x3p_percent = st.slider("3-Pt Field Goal Percentage (%)", 0.000, 53.645, 30.000)
        st.info("""NBA Record (1 Season): Kyle Korver w/ 53.6%""", icon="💡")
        ft_percent = st.slider("Free Throw Percentage (%)", 0.000, 98.100, 75.000)
        st.info("""NBA Record (1 Season): José Calderón w/ 98.1%""", icon="💡")
        submit_button = st.form_submit_button(label='Predict')
        
        if submit_button:
            pos = predict_pos(model, scaler, label_encoder, x2p_percent, x3p_percent, ft_percent, trb, ast, stl, blk, pts)
            st.write(f"Predicted Position: {pos}")
    
    st.feedback("thumbs")
    mention(
            label="GitHub Repo: verneylmavt/st-nba-vis",
            icon="github",
            url="https://github.com/verneylmavt/st-nba-vis"
        )
    mention(
            label="Other ML Tasks",
            icon="streamlit",
            url="https://verneylogyt.streamlit.app/"
        )
    st.divider()
    
    st.subheader("""Model""")
    with echo_expander(code_location="below", label="Code"):
        import xgboost as xgb
        
        xgb.XGBClassifier(
            objective='multi:softmax',
            eval_metric = 'mlogloss',
            device='cuda',
            tree_method='hist',
            n_estimators=699,
            max_depth=3,
            learning_rate=0.16214382011042164,
            subsample=0.6879927750830442,
            colsample_bytree=0.9552972663153627,
            gamma=0.08055405739393251,
            reg_alpha=0.46722112186774795,
            reg_lambda=0.605235490035828
        )
        
    # st.subheader("""Prediction""")
    # st.altair_chart(test_pred_chart, use_container_width=True, theme="streamlit")
    # st.subheader("""Tree Explainer""")
    # st.altair_chart(tree_explainer_chart, use_container_width=True, theme="streamlit")
    
    
    # st.subheader("""Feature Spaces""")
    # st.altair_chart(pca_chart.interactive(), use_container_width=True, theme="streamlit")
    st.subheader("""Feature Importances""")
    st.altair_chart(feat_imp_chart.interactive(), use_container_width=True, theme="streamlit")
    st.subheader("""Confusion Matrix""")
    st.altair_chart((cm_chart[0] + cm_chart[1]).interactive(), use_container_width=True, theme="streamlit")
    
    st.subheader("""Evaluation Metrics""")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", "0.65", border=True)
    col2.metric("Recall", "0.65", border=True)
    col3.metric("F1-Score", "0.65", border=True)
    col4.metric("Log Loss", "0.8145", border=True)
    # st.divider()



def player_sims():
    st.header("Player Similarity")
    
    latent_reps = load_joblib_object("latent_reps.pkl", "player_sims")
    X = load_df_csv("X.csv", "player_sims")
    
    knn = NearestNeighbors(n_neighbors=75, metric='euclidean', algorithm='auto')
    knn.fit(latent_reps)
    
    player_list = X['player'].unique().tolist()
    player_list.sort()
    season_list = X['season'].unique().tolist()
    season_list.sort()
    
    # st.write("Done!")
    
    # def search_players(searchterm: str) -> list:
    #     return [movie for movie in player_list if searchterm.lower() in movie.lower()] if searchterm else []

    # player = st_searchbox(
    #     search_function=search_players,
    #     placeholder="Search Players...", 
    #     key="player_search"
    # )
    
    
    # ----------------------
    # Prediction Function
    # ----------------------
    
    def predict_sim_players(player_name, season, embeddings, knn, X, top_n, allow_same_player=False, max_other_player=1):
        query_df = X[(X['player'] == player_name) & (X['season'] == season)]
        # st.write(query_df)
        
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
        
        # query_df["distance"] = 0
        query_df.loc[:, "distance"] = 0
        result_df = pd.concat([query_df, result_df])
        
        return result_df.reset_index(drop=True), similar, query_idx
    
    
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
                [pd.Series([100]), 
                pd.Series([100 - dist for _, dist in similar])]
            ).reset_index(drop=True)
        })
        return visualization_df
    
    
    with st.form(key='player_sims_form'):
        player = st.selectbox(
            "Player",
            player_list,
        )
        
        # def search_players(searchterm: str) -> list:
        #     return [movie for movie in player_list if searchterm.lower() in movie.lower()] if searchterm else []
        
        # player = st_searchbox(
        #     search_function=search_players,
        #     placeholder="Michael Jordan", 
        #     key="player_search",
        #     label="Player"
        # )
        
        season = st.selectbox(
            "Season",
            season_list,
        )
        top_n = st.slider("Number of Recommendations", 0, 50, 10)
        # same = st.checkbox("Allow Same Players")
        # same = st.toggle("Allow Same Players")
        
        submit_button = st.form_submit_button(label='Find')
        if submit_button:
            # if same:
            #     same = 3
            # else:
            #     same = 0
            if player and season:
                result_df, similar, query_idx = predict_sim_players(player, season, latent_reps, knn, X, top_n)
                if not result_df.empty:
                    result_df = result_df[['player', 'season', 'pos', 'pts', 'ast', 'trb', 'stl', 'blk', 'x2p_percent', 'x3p_percent', 'ft_percent', 'distance']]
                    result_df['season'] = result_df['season'].astype(str)
                    result_df = result_df.rename(columns={
                        'player': 'Name',
                        'season': 'Season',
                        'pos': 'Position',
                        'pts': 'PTS',
                        'ast': 'AST',
                        'trb': 'REB',
                        'stl': 'STL',
                        'blk': 'BLK',
                        'x2p_percent': '2P%',
                        'x3p_percent': '3P%',
                        'ft_percent': 'FT%',
                        'distance': 'Euclidean Distance'
                    })
                    st.dataframe(result_df, hide_index=True, use_container_width=True)
                    
                visualization_df = visualize_sim_players(player, season, latent_reps, similar, query_idx)
                visualization_df['Similarity (Normalized)'] = (visualization_df['Similarity'] - visualization_df['Similarity'].min()) / \
                                            (visualization_df['Similarity'].max() - visualization_df['Similarity'].min())
                if not visualization_df.empty:
                    st.altair_chart(
                        alt.Chart(visualization_df).mark_circle().encode(
                            x='x',
                            y='y',
                            size = alt.Size('Similarity (Normalized):Q' , scale=alt.Scale(range=[100, 500]), legend=None),
                            color=alt.Color('Similarity:Q', 
                                            # scale=alt.Scale(scheme='viridis'), 
                                            title='Similarity (%)'),
                            tooltip=['Title', 'Similarity']
                        ).properties(
                            width=800,
                            height=600
                        ).interactive(),
                        use_container_width=True,
                        theme="streamlit"
                    )
    st.feedback("thumbs")
    mention(
            label="GitHub Repo: verneylmavt/st-nba-vis",
            icon="github",
            url="https://github.com/verneylmavt/st-nba-vis"
        )
    mention(
            label="Other ML Tasks",
            icon="streamlit",
            url="https://verneylogyt.streamlit.app/"
        )
    st.divider()
    
    st.subheader("""Parameters""")
    st.code("""
            Batch Size = 128
            Input Dimension = 30
            Latent Dimension = 25
            
            Epochs = 59
            Learning Rate = 0.007262494401279974
                    """, 
            language="None")
    
    
    st.subheader("""Model""")
    with echo_expander(code_location="below", label="Code"):
        import torch
        import torch.nn as nn
        
        class Model(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super(Model, self).__init__()
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



# ----------------------
# Page UI
# ----------------------

def main():
    st.set_page_config(layout="wide")
    
    st.title("NBA Analysis and Visualization")
    st.divider()
    # st.warning('THE WORK IS STILL IN PROGRESS!!!', icon="⚒️")
    
    with st.sidebar:
        st.title("Pages")
        category = st.radio("Category", ["Charts", "Supervised Learning", "Unsupervised Learning"])
        if category == "Charts":
            option = st.radio("Options", ["Age vs Performance", "Position vs Performance", "Era vs Performance"])
        elif category == "Supervised Learning":
            option = st.radio("Options", ["Assist → Turnover", "Steal & Block → Personal Foul", "Performance → Position"])
        elif category == "Unsupervised Learning":
            option = st.radio("Options", ["Player Similarity"])
    
    if category == "Charts":
        if option == "Age vs Performance":
            age_v_perf()
        elif option == "Position vs Performance":
            pos_v_perf()
        elif option == "Era vs Performance":
            era_v_perf()

    elif category == "Supervised Learning":
        if option == "Assist → Turnover":
            ast_v_tov()
        elif option == "Steal & Block → Personal Foul":
            stl_blk_v_pf()
        elif option == "Performance → Position":
            mtrcs_v_pos()
    
    elif category == "Unsupervised Learning":
        if option == "Player Similarity":
            player_sims()

if __name__ == "__main__":
    main()
