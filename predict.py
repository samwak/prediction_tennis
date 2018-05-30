import datapreparation
import featurization
import model
import sys
import requests
import ast

game_id = sys.argv[1]
RATIO_TRAIN_TEST = float(sys.argv[2])
NUMBER_OF_BETS_MIN = int(sys.argv[3])
NUMBER_OF_EXPERTS = int(sys.argv[4])
VALUE_MIN = float(sys.argv[5])
CONFIANCE = float(sys.argv[6])
BASE = sys.argv[7]
# if len(sys.argv) == 9:
#     if sys.argv[8] == "1":
#         token1 = "Token 5b443bfbdeefbefcf3587e581e336f2564940dac"
#         token2 = "Token 5a151db3efe0f3e94dda98cd500767d37e0c652d"
#         token3 = "Token aaa9d03030f038f77005e70f3156568004584818"
#     if sys.argv[8] == "2":
#         token1 = "Token f8f6e169ada4cf82331073beb2d2eae7f194fe21"
#         token2 = "Token f7ca7716e30c678d3d7d4eb06c2bbcd4ab0d9dfd"
#         token3 = "Token 1e6517001c01fed6fcb50f4281d814137872f379"
#     if sys.argv[8] == "3":
#         token1 = "Token 9d547fa99a1c36954a6ecd1c727a9958ec942a54"
#         token2 = "Token 065b9c1e4f20b97e0c8d9d54bce34f80b2cce949"
#         token3 = "Token e436f0f2bb98b5e0cc95441989949ed3298072c5"
#     if sys.argv[8] == "4":
#         token1 = "Token 2b0c27c54ef3836c3590d1340d5602369ed5e7d7"
#         token2 = "Token 3118226059a6426b3439b7e27fd61c250fe5a51d"
#         token3 = "Token d953198b0fad3f022c22ff39245dc98ef51509ac"
BASE_URL = "https://api.sayourbet.com"

"""Data Pipeline"""
db = datapreparation.DataPreparation(NUMBER_OF_EXPERTS, BASE)
featurization = featurization.Featurization()
X_id, X_train, X_test, y_train, y_test, winning_odd_train, winning_odd_test, X_pred, X_id_pred = featurization.create_features(db, db.match_list, NUMBER_OF_BETS_MIN, NUMBER_OF_EXPERTS, RATIO_TRAIN_TEST)

"""Model fit"""
model = model.Model(VALUE_MIN, CONFIANCE)
model.fit(X_train, y_train)

if game_id == "all":
    """Classifier Model predict => predict issue of the games"""
    y_pred_logreg, y_pred_forest, y_pred_boost = model.predict_class(X_test)

    """Regression Model predict => specific profile prediction"""
    profil_wr, profil_v, profil_opt = model.predict_probas(X_test, X_id)

    """Calculate and display results"""
    accuracy_score_logreg, accuracy_score_forest, accuracy_score_boost = model.accuracy_score(y_test, y_pred_logreg, y_pred_forest, y_pred_boost)
    model.confusion_matrix(y_test, y_pred_logreg, y_pred_forest, y_pred_boost)
    won_odds_logreg, won_odds_forest, won_odds_boost = model.won_odds(y_test, y_pred_logreg, y_pred_forest, y_pred_boost, winning_odd_test)
    model.display_results(accuracy_score_logreg, accuracy_score_forest, accuracy_score_boost, won_odds_logreg, won_odds_forest, won_odds_boost, len(winning_odd_test))
    accuracy_score_wr, accuracy_score_v, accuracy_score_opt = model.accuracy_score_probas(y_test, profil_wr, profil_v, profil_opt)
    won_odds_wr, won_odds_v, won_odds_opt = model.won_odds_probas(y_test, profil_wr, profil_v, profil_opt, X_id)
    model.display_results_probas(accuracy_score_wr, accuracy_score_v, accuracy_score_opt, won_odds_wr, won_odds_v, won_odds_opt, profil_wr, profil_v, profil_opt)
    if X_pred.shape[0] > 0:
        profil_wr_predictive, profil_v_predictive, profil_opt_predictive = model.predict_probas(X_pred, X_id_pred)
        model.display_results_predictions(X_id_pred, profil_wr_predictive, profil_v_predictive, profil_opt_predictive, display=True)

# else:
    # """Prediction"""
    # if X_pred.shape[0]>0:
    #     profil_wr_predictive, profil_v_predictive, profil_opt_predictive = model.predict_probas(X_pred, X_id_pred)
    #     results = model.display_results_predictions(X_id_pred, profil_wr_predictive, profil_v_predictive, profil_opt_predictive, display=False)
    #     for element in ast.literal_eval(game_id):
    #         if element in list(results.game_id):
    #             result = results.loc[results.game_id == element][['game_id', 'profil_wr', 'profil_v', 'profil_opt']].iloc[0].values
    #             print(result)
    #             if len(sys.argv) == 9:
    #                 if result[1] != 0:
    #                     bop = db.bets.loc[(db.bets.game_id == result[0]) & (db.bets.type_prono_id == 1) & (db.bets.pronostic_id == result[1])].sort_values(['date'], ascending=False).iloc[0].bop_id
    #                     ticket_id = requests.post(BASE_URL + "/v1/tickets/add/", headers={'Authorization': token1}, data={'betoddpair_set': [int(bop)]}).json()['id']
    #                     print(ticket_id)
    #                     post = requests.post(BASE_URL + "/v1/posts/add/", headers={'Authorization': token1}, data={'ticket': int(ticket_id)})
    #                     print(post.content)
    #                 if result[2] != 0:
    #                     bop = db.bets.loc[(db.bets.game_id == result[0]) & (db.bets.type_prono_id == 1) & (db.bets.pronostic_id == result[2])].sort_values(['date'], ascending=False).iloc[0].bop_id
    #                     ticket_id = requests.post(BASE_URL + "/v1/tickets/add/", headers={'Authorization': token2}, data={'betoddpair_set': [int(bop)]}).json()['id']
    #                     post = requests.post(BASE_URL + "/v1/posts/add/", headers={'Authorization': token2}, data={'ticket': int(ticket_id)})
    #                     print(post.content)
    #                 if result[3] != 0:
    #                     bop = db.bets.loc[(db.bets.game_id == result[0]) & (db.bets.type_prono_id == 1) & (db.bets.pronostic_id == result[3])].sort_values(['date'], ascending=False).iloc[0].bop_id
    #                     ticket_id = requests.post(BASE_URL + "/v1/tickets/add/", headers={'Authorization': token3}, data={'betoddpair_set': [int(bop)]}).json()['id']
    #                     post = requests.post(BASE_URL + "/v1/posts/add/", headers={'Authorization': token3}, data={'ticket': int(ticket_id)})
    #                     print(post.content)
    #         else:
    #             print("pas assez de données pour ce match")
    # else:
    #     print("va te faire enculer negro")