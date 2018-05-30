import pandas as pd
import psycopg2
import datetime

sports = [0, 1, 2, 3, 4, 5, 6]
countries = [0, 1, 2, 3, 4, 5, 6]
leagues = [0, 1, 2, 3, 4, 5, 6]
types_de_prono = [0, 1, 2, 3, 4, 5, 6]
pronos = [0, 1, 2, 3, 4, 5, 6]
datefrom = datetime.date(2016,12,16)
dateto = datetime.date(2017,12,31)
NUMBER_OF_BETS_MIN = 10
#methods : score, ratio, bilan, number_of_wins, number_of_bets
method = 'score'


def create_bets_df():
    cursor = psycopg2.connect(dbname='SYB', password='admin').cursor()
    cursor.execute("SELECT * FROM bets_view_betsinfo order by date")
    rows = cursor.fetchall()
    cursor.execute("SELECT column_name FROM information_schema.columns where table_name='bets_view_betsinfo'")
    columns = [x[0] for x in cursor.fetchall()]
    return pd.DataFrame(rows, columns=columns)

def ranking(bets, sports, countries, leagues, types_de_prono, pronos,  NUMBER_OF_BETS_MIN, method, date_from=0, date_to=0, date = 0):
    if date!=0:
        date_from = date
        date_to = date
    profiles = pd.DataFrame(
        bets.loc[(bets.date <= date_to) & (bets.date >= date_from)].profile_id.value_counts())
    profiles.columns = ['number_of_bets']
    profiles = profiles.loc[profiles['number_of_bets'] > NUMBER_OF_BETS_MIN]
    profiles['number_of_wins'] = bets.loc[
        (bets.date <= date_to) & (bets.date >= date_from) & (bets.result == True) & (
        bets.sport_id.isin(sports)) & (bets.country_id.isin(countries)) & (
        bets.competition_id.isin(leagues)) & (bets.type_prono_id.isin(types_de_prono)) & (
        bets.pronostic_id.isin(pronos))].profile_id.value_counts()
    temp = pd.DataFrame(bets.loc[(bets.date <= date_to) & (bets.date >= date_from) & (bets.result == True)][
                            ['profile_id', 'odd_value']].apply(pd.to_numeric).groupby('profile_id')[
                            'odd_value'].mean()).reset_index()
    temp2 = profiles.reset_index().apply(pd.to_numeric)
    temp2.columns = ['profile_id', 'number_of_bets', 'number_of_wins']
    profiles = temp2.merge(temp, on='profile_id', how='left')
    profiles['bilan'] = profiles.number_of_wins * profiles.odd_value - profiles.number_of_bets
    profiles['ratio'] = profiles.number_of_wins / profiles.number_of_bets
    score = []
    for bilan, ratio in zip(profiles.bilan, profiles.ratio):
        score.append(bilan * ratio if bilan > 0 else (bilan * (1 - ratio)))
    profiles['score'] = score
    ranking = profiles.sort_values(method, ascending=False).reset_index(drop=True).reset_index()
    ranking = ranking.rename(columns={'index': 'rank'})
    return ranking

bets = create_bets_df()
print(ranking(bets, sports, countries, leagues, types_de_prono, pronos, datefrom, dateto, NUMBER_OF_BETS_MIN, method))