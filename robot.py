import os
import pytz
from datetime import timedelta
import datapreparation

class Robot:
    def __init__(self, db):
        self.db = db
        self.daily_match_list = self.get_daily_match_list(self.db)

    def get_daily_match_list(self, db):
        return db.create_daily_match_list_df(db.cursor)

    def set_tasks(self):
        tz = pytz.timezone('CET')
        dict = {}
        base_command = '(echo source ~/.bashrc; echo cd /home/ec2-user/Virtualenvs/p35/;echo source bin/activate;echo cd prediction;'
        for match in self.daily_match_list.values:
            if match[1] not in dict.keys():
                dict[match[1]] = [match[0]]
            else:
                dict[match[1]].append(match[0])
        for dt, game_ids in dict.items():
            dt = dt.astimezone(tz)+timedelta(minutes=-20)
            game_ids = str(game_ids).replace(' ', '')
            command1 = 'echo python Predict.py "'+game_ids+'" 1 6 6 1.6 0.90 prod 1) | at '+str(dt.hour)+":"+str(dt.minute)+" tomorrow"
            command1 = base_command+command1
            os.system(command1)
            command2 = 'echo python Predict.py "'+game_ids+'" 1 6 6 1.7 0.93 prod 2) | at '+str(dt.hour)+":"+str(dt.minute+1)+" tomorrow"
            command2 = base_command+command2
            os.system(command2)
            command3 = 'echo python Predict.py "'+game_ids+'" 1 8 8 1.6 0.92 prod 3) | at '+str(dt.hour)+":"+str(dt.minute+2)+" tomorrow"
            command3 = base_command+command3
            os.system(command3)
            command4 = 'echo python Predict.py "'+game_ids+'" 1 8 8 1.8 0.95 prod 4) | at '+str(dt.hour)+":"+str(dt.minute+3)+" tomorrow"
            command4 = base_command+command4
            os.system(command4)
            print(command1, command2, command3, command4)

db = datapreparation.DataPreparation(base="prod", init=False)
robot = Robot(db)
robot.set_tasks()
