import pandas as pd
import datetime

class Featurization:
    def create_features(self, db, match_list, NUMBER_OF_BETS_MIN, NUMBER_OF_EXPERTS, RATIO_TRAIN_TEST):
        match_vectors = []
        robots_id = [1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1104]
        datefrom = datetime.date(2016,12,16)
        bet_list = db.bets
        bet_list['profile_id'] = bet_list['profile_id'].apply(pd.to_numeric)
        bet_list = bet_list.loc[~bet_list.profile_id.isin(robots_id)]
        for match in match_list.values:
            dateto = match[5]
            if datefrom < dateto:
                ranking = self.scoring(db, 0, 0, 0, datefrom, dateto, NUMBER_OF_BETS_MIN)
                match_vector = list(match)
                match_bet_list = bet_list.loc[(bet_list.game_id == match[0]) & (bet_list.type_prono_id == match[4])]
                match_bet_list = match_bet_list.merge(ranking[['profile_id', 'rank', 'number_of_bets', 'bilan', 'ratio', 'score']], on='profile_id', how='left')
                data = match_bet_list.dropna(subset=['rank'])
                if data['rank'].count() >= NUMBER_OF_EXPERTS:
                    data = data.sort_values(['rank']).head(NUMBER_OF_EXPERTS)
                    for values in data.values:
                        #match_vector.append(values[-5])
                        match_vector.append(values[14])
                        match_vector.append(values[-4])
                        match_vector.append(values[-3])
                        match_vector.append(values[-2])
                        match_vector.append(values[-1])
                    match_vectors.append(match_vector)
        columns = ['game_id', 'sport_id', 'country_id', 'competition_id', 'type_prono_id', 'date', 'result_id', '1', '2']
        # columns = ['game_id', 'sport_id', 'country_id', 'competition_id', 'type_prono_id', 'date', 'result_id', '1', 'N', '2']
        normalize = []
        for index in range(NUMBER_OF_EXPERTS):
            columns.append('rank'+str(index+1))
            normalize.append('rank' + str(index + 1))
            columns.append('prono' + str(index + 1))
            columns.append('number_of_bets' + str(index + 1))
            normalize.append('number_of_bets' + str(index + 1))
            columns.append('bilan' + str(index + 1))
            normalize.append('bilan' + str(index + 1))
            columns.append('ratio' + str(index + 1))
            columns.append('score' + str(index + 1))
            normalize.append('score' + str(index + 1))
        df = pd.DataFrame(match_vectors, columns=columns)
        df = df.dropna()
        df['year'] = pd.DatetimeIndex(df['date']).year
        df['month'] = pd.DatetimeIndex(df['date']).month
        df['day'] = pd.DatetimeIndex(df['date']).day
        # country_list = db.create_country_list(db.cursor, df.country_id.unique())
        # competition_list = db.create_competition_list(db.cursor, df.competition_id.unique())
        # df[country_list] = pd.get_dummies(df.country_id)
        # df[competition_list] = pd.get_dummies(df.competition_id)
        for index in range(NUMBER_OF_EXPERTS):
                df[['v1'+str(index+1), 'v2'+str(index+1)]] = pd.get_dummies(df['prono'+str(index+1)])
                # df[['v1'+str(index+1), 'n'+str(index+1), 'v2'+str(index+1)]] = pd.get_dummies(df['prono'+str(index+1)])
        df[normalize] = (df[normalize] - df[normalize].mean()) / (df[normalize].max() - df[normalize].min())
        useless_columns = ['game_id', 'sport_id', 'type_prono_id', 'result_id', 'country_id', 'competition_id', 'date']
        for index in range(NUMBER_OF_EXPERTS):
            useless_columns.append('prono' + str(index + 1))
        winning_odd = []
        df_pred = db.next_matchs_list.game_id
        df_predict = df.loc[df.game_id.isin(df_pred)]
        df = df.head(df.shape[0]-df_predict.shape[0]-1)
        for element in df.values:
            if element[6] == 1:
                winning_odd.append(element[7])
            if element[6] == 2:
                winning_odd.append(element[8])
            if element[6] == 3:
                winning_odd.append(element[9])
        train_size = int(df.shape[0] * RATIO_TRAIN_TEST)
        X_id = df.tail(df.shape[0]-train_size)
        X_train = df.head(train_size).drop(useless_columns, axis=1)
        X_test = df.tail(df.shape[0]-train_size).drop(useless_columns, axis=1)
        y_train = df.head(train_size)[['result_id']]
        y_test = df.tail(df.shape[0]-train_size)[['result_id']]
        X_pred = df_predict.drop(useless_columns, axis=1)
        X_id_pred = df_predict
        winning_odd_train = winning_odd[:train_size]
        winning_odd_test = winning_odd[train_size:]
        return X_id, X_train, X_test, y_train, y_test, winning_odd_train, winning_odd_test, X_pred, X_id_pred

    def scoring(self, db, sport, country, league, date_from, date_to, NUMBER_OF_BETS_MIN):
        profiles = pd.DataFrame(db.bets.loc[db.bets.date < date_to].profile_id.value_counts())
        profiles.columns = ['number_of_bets']
        profiles = profiles.loc[profiles['number_of_bets'] > NUMBER_OF_BETS_MIN]
        profiles['number_of_wins'] = db.bets.loc[(db.bets.date < date_to) & (db.bets.result == True)].profile_id.value_counts()
        temp = pd.DataFrame(db.bets.loc[(db.bets.date < date_to) & (db.bets.result == True)][['profile_id', 'odd_value']].apply(pd.to_numeric).groupby('profile_id')['odd_value'].mean()).reset_index()
        temp2 = profiles.reset_index().apply(pd.to_numeric)
        temp2.columns = ['profile_id', 'number_of_bets', 'number_of_wins']
        profiles = temp2.merge(temp, on='profile_id', how='left')
        profiles['bilan'] = profiles.number_of_wins * profiles.odd_value - profiles.number_of_bets
        profiles['ratio'] = profiles.number_of_wins / profiles.number_of_bets
        profiles['score'] = [bilan * ratio if bilan > 0 else (bilan * (1-ratio)) for (bilan, ratio) in zip(profiles.bilan, profiles.ratio)]
        ranking = profiles.sort_values('score', ascending=False).reset_index(drop=True).reset_index()
        ranking = ranking.rename(columns={'index': 'rank'})
        return ranking
