import pandas as pd
import altair as alt
import streamlit as st

def load_df_time():
    df = pd.read_csv("data/player_totals.csv")
    df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})
    df = df[df['season'] >= 1955]
    df = df.drop(columns=['birth_year'])
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

    position_stats = df.groupby('main_pos').apply(
        lambda group: pd.Series({
            'PPG': group['pts'].sum() / group['g'].sum(),
            'APG': group['ast'].sum() / group['g'].sum(),
            'SPG': group['stl'].sum() / group['g'].sum(),
            'BPG': group['blk'].sum() / group['g'].sum(),
            'RPG': group['trb'].sum() / group['g'].sum(),
        })
    ).reset_index()
    position_order = ['PG', 'SG', 'SF', 'PF', 'C']

    max_avg_positions = {metric: position_stats.loc[position_stats[metric].idxmax(), 'main_pos'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
    min_avg_positions = {metric: position_stats.loc[position_stats[metric].idxmin(), 'main_pos'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}

    metrics = ['PPG', 'APG', 'RPG', 'SPG', 'BPG']
    performances = ['pts', 'ast', 'trb', 'stl', 'blk']
    titles = [
        'Points Per Game (PPG)', 
        'Assists Per Game (APG)', 
        'Rebounds Per Game (RPG)',
        'Steals Per Game (SPG)', 
        'Blocks Per Game (BPG)', 
    ]
    
    for metric, performance, title in zip(metrics, performances, titles):
        base = alt.Chart(position_stats).mark_bar(opacity=0.7).encode(
            x=alt.X('main_pos:N', title='Position', sort=position_order),
            y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
            tooltip=['main_pos', f'{metric}:Q']
        ).properties(
            title=f'Position vs {title}'
        )

        max_pos = max_avg_positions[metric]
        max_highlight = alt.Chart(position_stats[position_stats['main_pos'] == max_pos]).mark_bar(
            color='mediumseagreen', opacity=0.7
        ).encode(
            x=alt.X('main_pos:N', sort=position_order),
            y=alt.Y(f'{metric}:Q'),
            tooltip=['main_pos', f'{metric}:Q']
        )

        min_pos = min_avg_positions[metric]
        min_highlight = alt.Chart(position_stats[position_stats['main_pos'] == min_pos]).mark_bar(
            color='darkred', opacity=0.7
        ).encode(
            x=alt.X('main_pos:N', sort=position_order),
            y=alt.Y(f'{metric}:Q'),
            tooltip=['main_pos', f'{metric}:Q']
        )

        max_metric_stats = df.groupby('main_pos').apply(
            lambda x: x.loc[x[metric].idxmax()]
        ).reset_index(drop=True)
        metric_annotations = alt.Chart(max_metric_stats).mark_text(
            dy=-10, size=10, opacity=0.7, color="whitesmoke"
        ).encode(
            x=alt.X('main_pos:N', sort=position_order),
            y=alt.Y(f'{metric}:Q'),
            text='player:N',
            tooltip=['main_pos', 'player', f'{metric}:Q']
        )
        
        max_performance_stats = df.groupby('main_pos').apply(
            lambda x: x.loc[x[performance].idxmax()]
        ).reset_index(drop=True)
        performance_annotations = alt.Chart(max_performance_stats).mark_text(
            dy=-10, size=10, opacity=0.7, color="whitesmoke"
        ).encode(
            x=alt.X('main_pos:N', sort=position_order),
            y=alt.Y(f'{metric}:Q'),
            text='player:N',
            tooltip=['main_pos', 'player', f'{performance}:Q']
        )

        final_chart = (base + max_highlight + min_highlight + metric_annotations + performance_annotations).interactive()

        st.altair_chart(final_chart, use_container_width=True, theme="streamlit")



def age_v_perf():
    st.header("Age vs Performance")
    
    df = load_df_time().copy()
    df = df[df['lg'] == 'NBA']
    df = df[df['g'] >= 48.34820326006421]
    
    df['PPG'] = df['pts'] / df['g']  # Points per Game
    df['APG'] = df['ast'] / df['g']  # Assists per Game
    df['SPG'] = df['stl'] / df['g']  # Steals per Game
    df['BPG'] = df['blk'] / df['g']  # Blocks per Game
    df['RPG'] = df['trb'] / df['g']  # Rebounds per Game
    # df.dropna(subset=['age', 'PPG', 'APG', 'SPG', 'BPG', 'RPG', 'player'], inplace=True)
    
    age_stats = df.groupby('age').apply(
        lambda group: pd.Series({
            'PPG': group['pts'].sum() / group['g'].sum(),  # Points per Game
            'APG': group['ast'].sum() / group['g'].sum(),  # Assists per Game
            'SPG': group['stl'].sum() / group['g'].sum(),  # Steals per Game
            'BPG': group['blk'].sum() / group['g'].sum(),  # Blocks per Game
            'RPG': group['trb'].sum() / group['g'].sum(),  # Rebounds per Game
        })
    ).reset_index()
    
    
    max_avg_ages = {metric: age_stats.loc[age_stats[metric].idxmax(), 'age'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
    min_avg_ages = {metric: age_stats.loc[age_stats[metric].idxmin(), 'age'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}

    metrics = ['PPG', 'APG', 'RPG', 'SPG', 'BPG']
    titles = [
        'Points Per Game (PPG)', 
        'Assists Per Game (APG)', 
        'Rebounds Per Game (RPG)',
        'Steals Per Game (SPG)', 
        'Blocks Per Game (BPG)', 
    ]

    for metric, title in zip(metrics, titles):
        max_metric_stats = df.groupby('age').apply(
            lambda x: x.loc[x[metric].idxmax()]
        ).reset_index(drop=True)

        average_line = alt.Chart(age_stats).mark_line().encode(
            x=alt.X('age:Q', title='Age'),
            y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
            tooltip=['age', metric]
        ).properties(
            title=f'Age vs {title}'
        )

        # max_avg_value = age_stats.loc[age_stats['age'] == max_avg_ages[metric], metric].values[0]
        max_avg_point = alt.Chart(age_stats[age_stats['age'] == max_avg_ages[metric]]).mark_point(
            size=100, color='mediumseagreen', opacity=0.7
        ).encode(
            x='age:Q',
            y=f'{metric}:Q',
            tooltip=['age', metric]
        )
        
        # min_avg_value = age_stats.loc[age_stats['age'] == min_avg_ages[metric], metric].values[0]
        min_avg_point = alt.Chart(age_stats[age_stats['age'] == min_avg_ages[metric]]).mark_point(
            size=100, color='darkred', opacity=0.7
        ).encode(
            x='age:Q',
            y=f'{metric}:Q',
            tooltip=['age', metric]
        )
        
        max_player_points = alt.Chart(max_metric_stats).mark_circle().encode(
            x=alt.X('age:Q', title='Age'),
            y=alt.Y(f'{metric}:Q'),
            size=alt.Size(f'{metric}:Q'),
            color=alt.Color(f'{metric}:Q'),
            tooltip=['age', 'player', metric]
        ).interactive()

        final_chart = (average_line + max_avg_point + min_avg_point + max_player_points).interactive()
        st.altair_chart(final_chart, use_container_width=True, theme="streamlit")



def era_v_perf():
    st.header("Era vs Performance")

    df = load_df_time().copy()
    df = df[df['g'] >= 48.34820326006421]
    
    df['era'] = ((df['season'] // 10) * 10).astype(int).astype(str) + 's'
    
    df['PPG'] = df['pts'] / df['g']  # Points per Game
    df['APG'] = df['ast'] / df['g']  # Assists per Game
    df['SPG'] = df['stl'] / df['g']  # Steals per Game
    df['BPG'] = df['blk'] / df['g']  # Blocks per Game
    df['RPG'] = df['trb'] / df['g']  # Rebounds per Game
    
    df.dropna(subset=['era', 'PPG', 'APG', 'SPG', 'BPG', 'RPG', 'player'], inplace=True)
    
    era_stats = df.groupby('era').apply(
        lambda group: pd.Series({
            'PPG': group['pts'].sum(skipna=True) / group['g'].sum(skipna=True),
            'APG': group['ast'].sum(skipna=True) / group['g'].sum(skipna=True),
            'SPG': group['stl'].sum(skipna=True) / group['g'].sum(skipna=True),
            'BPG': group['blk'].sum(skipna=True) / group['g'].sum(skipna=True),
            'RPG': group['trb'].sum(skipna=True) / group['g'].sum(skipna=True),
        })
    ).reset_index()
    era_stats.drop(era_stats[(era_stats[['era', 'PPG', 'APG', 'SPG', 'BPG', 'RPG']] == 0).any(axis=1)].index, inplace=True)
    era_stats.reset_index(drop=True, inplace=True)
    
    max_avg_eras = {metric: era_stats.loc[era_stats[metric].idxmax(), 'era'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
    min_avg_eras = {metric: era_stats.loc[era_stats[metric].idxmin(), 'era'] for metric in ['PPG', 'APG', 'SPG', 'BPG', 'RPG']}
    
    metrics = ['PPG', 'APG', 'RPG', 'SPG', 'BPG']
    titles = [
        'Points Per Game (PPG)', 
        'Assists Per Game (APG)', 
        'Rebounds Per Game (RPG)',
        'Steals Per Game (SPG)', 
        'Blocks Per Game (BPG)', 
    ]
    
    for metric, title in zip(metrics, titles):
        # Base bar chart for the metric
        base = alt.Chart(era_stats).mark_bar(opacity=0.7).encode(
            x=alt.X('era:N', title='Era', sort=sorted(era_stats['era'].unique())),
            y=alt.Y(f'{metric}:Q', title=f'{metric.upper()}'),
            tooltip=['era', f'{metric}:Q']
        ).properties(
            title=f'Era vs {title}'
        )

        # Highlight the era with the maximum average value
        max_era = max_avg_eras[metric]
        max_highlight = alt.Chart(era_stats[era_stats['era'] == max_era]).mark_bar(
            color='mediumseagreen', opacity=0.7
        ).encode(
            x=alt.X('era:N', sort=sorted(era_stats['era'].unique())),
            y=alt.Y(f'{metric}:Q'),
            tooltip=['era', f'{metric}:Q']
        )

        # Highlight the era with the minimum average value
        min_era = min_avg_eras[metric]
        min_highlight = alt.Chart(era_stats[era_stats['era'] == min_era]).mark_bar(
            color='darkred', opacity=0.7
        ).encode(
            x=alt.X('era:N', sort=sorted(era_stats['era'].unique())),
            y=alt.Y(f'{metric}:Q'),
            tooltip=['era', f'{metric}:Q']
        )

        # Annotate the players with the highest stats for this specific metric
        max_metric_stats = df.groupby('era').apply(
            lambda x: x.loc[x[metric].idxmax()]
        ).reset_index(drop=True)
        metric_annotations = alt.Chart(max_metric_stats).mark_text(
            dy=-10, size=10, opacity=0.7, color="whitesmoke"
        ).encode(
            x=alt.X('era:N', sort=sorted(era_stats['era'].unique())),
            y=alt.Y(f'{metric}:Q'),
            text='player:N',
            tooltip=['era', 'player', f'{metric}:Q']
        )

        # Combine charts
        final_chart = (base + max_highlight + min_highlight + metric_annotations).interactive()

        # Display in Streamlit
        st.altair_chart(final_chart, use_container_width=True, theme="streamlit")

age_v_perf()
pos_v_perf()
# era_v_perf()
