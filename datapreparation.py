import psycopg2
import pandas as pd
import numpy as np
import datetime

class DataPreparation:
    def __init__(self, players=5, base="local", init=True, type_prono_id=240):
        self.cursor = self.connexion(base)
        print("connection ...")
        if init:
            self.PLAYERS = str(players)
            self.bets = self.create_bets_df(self.cursor)
            self.odds = self.create_odds_df(self.cursor)
            self.league = self.create_leagues_df(self.cursor)
            self.type_prono_id = str(type_prono_id)
            self.match_list = self.create_match_list_df(self.cursor)
            self.match_list = self.create_match_list_odds(self.cursor, self.match_list)
            self.next_matchs_list = self.create_next_matchs_list_df(self.cursor)
        print("connected !")

    def connexion(self, base):
        if base == "local":
            return psycopg2.connect(dbname='SYB', password='admin').cursor()
        if base == "prod":
            return psycopg2.connect(dbname='sayourbet', password='marcvero001', user='postgres', host='saydb.cefq6s1ya72a.us-west-2.rds.amazonaws.com', port='5432').cursor()
        if base == "dev":
            return psycopg2.connect(dbname='sayourbet', password='marcvero001', user='postgres', host='saydb.cefq6s1ya72a.us-west-2.rds.amazonaws.com', port='5432').cursor()


    def create_bets_df(self, cursor):
        cursor.execute("SELECT * FROM bets_view_betsinfo order by date")
        rows = cursor.fetchall()
        cursor.execute("SELECT column_name FROM information_schema.columns where table_name='bets_view_betsinfo'")
        columns = [x[0] for x in cursor.fetchall()]
        return pd.DataFrame(rows, columns=columns)

    def create_odds_df(self, cursor):
        cursor.execute("SELECT * FROM bets_bet")
        rows = cursor.fetchall()
        cursor.execute("SELECT column_name FROM information_schema.columns where table_name='bets_bet'")
        columns = [x[0] for x in cursor.fetchall()]
        return pd.DataFrame(rows, columns=columns)

    def create_leagues_df(self, cursor):
        cursor.execute("SELECT * FROM competitions_competition")
        rows = cursor.fetchall()
        cursor.execute("SELECT column_name FROM information_schema.columns where table_name='competitions_competition'")
        columns = [x[0] for x in cursor.fetchall()]
        return pd.DataFrame(rows, columns=columns)

    def create_match_list_df(self, cursor):
        cursor.execute("""  SELECT DISTINCT x.game_id, sport_id, country_id, x.competition_id, x.type_prono_id, date, x.pronostic_id
                            FROM (
                                SELECT DISTINCT COUNT(*), game_id, sport_id, country_id, competition_id, type_prono_id, date, pronostic_id, result
                                FROM bets_view_betsinfo 
                                where type_prono_id = """+self.type_prono_id+""" 
                                group by game_id, sport_id, country_id, competition_id, type_prono_id, date, pronostic_id, result
                                order by date
                                ) as x 
                            JOIN bets_bet ON bets_bet.game_id = x.game_id
                            where x.type_prono_id =  """+self.type_prono_id+""" and (x.result=true OR x.result IS NULL) and COUNT >= """+self.PLAYERS)
        rows = cursor.fetchall()
        columns = ['game_id', 'sport_id', 'country_id', 'competition_id', 'type_prono_id', 'date', 'result_id']
        return pd.DataFrame(rows, columns=columns).sort_values('date')

    def create_match_list_odds(self, cursor, match_list):
        cursor.execute(""" select game_id, value, pronostic_id from bets_bet
                    join bets_betoddpair on bets_betoddpair.bet_id = bets_bet.id
                    join bets_odd on bets_betoddpair.odd_id = bets_odd.id
                    where bets_bet.type_prono_id = 1""")
        rows = cursor.fetchall()
        columns = ['game_id', 'odd_value', 'pronostic_id']
        raw_df = pd.DataFrame(rows, columns = columns)
        net_df = pd.DataFrame(raw_df.game_id.unique(), columns = ['game_id'])
        # v1s, ns, v2s = [], [], []
        v1s, v2s = [], []
        cond2 = raw_df['pronostic_id'] == 1
        # cond3 = raw_df['pronostic_id'] == 2
        cond4 = raw_df['pronostic_id'] == 3
        for element in net_df.game_id:
            cond1 = raw_df['game_id'] == element
            v1 = np.mean(raw_df[cond1 & cond2].odd_value)
            # n = np.mean(raw_df[cond1 & cond3].odd_value)
            v2 = np.mean(raw_df[cond1 & cond4].odd_value)
            v1s.append(v1)
            # ns.append(n)
            v2s.append(v2)
        net_df['1'] = v1s
        # net_df['N'] = ns
        net_df['2'] = v2s
        match_list_odds = match_list.merge(net_df, on='game_id').dropna()
        return match_list_odds

    def create_country_list(self, cursor, indexes):
        if len(indexes) == 1:
            condition = "id = "+str(indexes[0])
        else:
            condition = "id in "+str(tuple(indexes))
        cursor.execute("SELECT name FROM public.competitions_country WHERE "+condition+" order by id;")
        rows = cursor.fetchall()
        return [x[0] for x in rows]

    def create_competition_list(self, cursor, indexes):
        if len(indexes) == 1:
            condition = "id = "+str(indexes[0])
        else:
            condition = "id in "+str(tuple(indexes))
        cursor.execute("SELECT name FROM public.competitions_competition WHERE "+condition+" order by id;")
        rows = cursor.fetchall()
        return [x[0] for x in rows]

    def create_next_matchs_list_df(self, cursor):
        cursor.execute("""  SELECT DISTINCT x.game_id, sport_id, country_id, x.competition_id, x.type_prono_id, date
                            FROM (
                                SELECT DISTINCT COUNT(*), game_id, sport_id, country_id, competition_id, type_prono_id, date, pronostic_id, result
                                FROM bets_view_betsinfo 
                                where type_prono_id = 1 
                                group by game_id, sport_id, country_id, competition_id, type_prono_id, date, pronostic_id, result
                                order by date
                                ) as x 
                            JOIN bets_bet ON bets_bet.game_id = x.game_id
                            where x.type_prono_id = 1 and x.result IS NULL and COUNT >= """+self.PLAYERS)
        rows = cursor.fetchall()
        columns = ['game_id', 'sport_id', 'country_id', 'competition_id', 'type_prono_id', 'date']
        return self.create_match_list_odds(self.cursor, pd.DataFrame(rows, columns=columns).sort_values('date'))

    def create_daily_match_list_df(self, cursor):
        date = str(datetime.datetime.now()+datetime.timedelta(days=1))
        cursor.execute("SELECT id, datetime FROM games_game where datetime > date '"+date+"' and datetime < date '"+date+"' + integer '1'""")
        rows = cursor.fetchall()
        columns = ['game_id', 'datetime']
        return pd.DataFrame(rows, columns=columns)




