from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost
import pandas as pd
import numpy as np
# import matplotlib.pylab as plt

class Model:
    def __init__(self, VALUE_MIN, CONFIANCE):
        self.logreg = linear_model.LogisticRegression(C=0.01, solver='lbfgs', multi_class='multinomial')
        self.clf = RandomForestClassifier(n_estimators = 100, max_depth=7, random_state=20)
        self.xgb = xgboost.XGBClassifier(learning_rate=0.01, n_estimators= 300, max_depth=4, gamma = 2, min_child_weight=1)
        self.VALUE_MIN = VALUE_MIN
        self.CONFIANCE = CONFIANCE

    def fit(self, X_train, y_train):
        self.logreg.fit(X_train, y_train.result_id)
        self.clf.fit(X_train, y_train.result_id)
        self.xgb.fit(X_train, y_train.result_id)
        # plt.figure(1)
        # plt.subplot(121)
        # plt.barh(range(X_train.shape[1]), self.clf.feature_importances_, color="r", align="center")
        # plt.yticks(range(X_train.shape[1]), X_train.columns)
        # plt.subplot(122)
        # plt.barh(range(X_train.shape[1]), self.xgb.feature_importances_, color="r", align="center")
        # plt.yticks(range(X_train.shape[1]), X_train.columns)
        # plt.show()


    def predict_class(self, X_test):
        return self.logreg.predict(X_test), self.clf.predict(X_test), self.xgb.predict(X_test)


    def predict_probas(self, X_test, X_id):
        df = pd.concat([pd.DataFrame(X_id[['game_id', '1', 'N', '2']]).reset_index(drop=True), pd.DataFrame(self.logreg.predict_proba(X_test), columns=['P1L', 'PNL', 'P2L']), pd.DataFrame(self.logreg.predict_proba(X_test), columns=['P1F', 'PNF', 'P2F']), pd.DataFrame(self.logreg.predict_proba(X_test), columns=['P1B', 'PNB', 'P2B'])], axis=1)
        df['G1L'] = df['1'] * df['P1L']
        df['GNL'] = df['N'] * df['PNL']
        df['G2L'] = df['2'] * df['P2L']
        df['G1F'] = df['1'] * df['P1F']
        df['GNF'] = df['N'] * df['PNF']
        df['G2F'] = df['2'] * df['P2F']
        df['G1B'] = df['1'] * df['P1B']
        df['GNB'] = df['N'] * df['PNB']
        df['G2B'] = df['2'] * df['P2B']
        probas_logreg, probas_forest, probas_boost = self.logreg.predict_proba(X_test), self.clf.predict_proba(X_test), self.xgb.predict_proba(X_test)
        profil_wr, profil_v, profil_opt = [], [], []
        for i in range(df.shape[0]):
            if (np.argmax(probas_logreg[i]) == np.argmax(probas_forest[i])) and (np.argmax(probas_logreg[i]) == np.argmax(probas_boost[i])) and (((max(probas_logreg[i]) + max(probas_forest[i]) + max(probas_boost[i])) / 3) > self.CONFIANCE):
                profil_wr.append(np.argmax(probas_logreg[i])+1)
                if df['G1L'].iloc[i] > self.VALUE_MIN and df['G1F'].iloc[i] > self.VALUE_MIN and df['G1B'].iloc[i] > self.VALUE_MIN and np.argmax(probas_logreg[i]) == 0:
                    profil_opt.append(1)
                elif df['GNL'].iloc[i] > self.VALUE_MIN and df['GNF'].iloc[i] > self.VALUE_MIN and df['GNB'].iloc[i] > self.VALUE_MIN and np.argmax(probas_logreg[i]) == 1:
                    profil_opt.append(2)
                elif df['G2L'].iloc[i] > self.VALUE_MIN and df['G2F'].iloc[i] > self.VALUE_MIN and df['G2B'].iloc[i] > self.VALUE_MIN and np.argmax(probas_logreg[i]) == 2:
                    profil_opt.append(3)
                else:
                    profil_opt.append(0)
            else:
                profil_wr.append(0)
                profil_opt.append(0)
            if df['G1L'].iloc[i] > self.VALUE_MIN and df['G1F'].iloc[i] > self.VALUE_MIN and df['G1B'].iloc[i] > self.VALUE_MIN:
                profil_v.append(1)
            elif df['GNL'].iloc[i] > self.VALUE_MIN and df['GNF'].iloc[i] > self.VALUE_MIN and df['GNB'].iloc[i] > self.VALUE_MIN:
                profil_v.append(2)
            elif df['G2L'].iloc[i] > self.VALUE_MIN and df['G2F'].iloc[i] > self.VALUE_MIN and df['G2B'].iloc[i] > self.VALUE_MIN:
                profil_v.append(3)
            else:
                profil_v.append(0)
        return profil_wr,profil_v,profil_opt

    def won_odds(self, y_test,y_pred_logreg, y_pred_forest, y_pred_boost, winning_odds):
        logreg = [winning_odds[index] if y_test.iloc[index, 0] == y_pred_logreg[index] else 0 for index in range(len(winning_odds))]
        forest = [winning_odds[index] if y_test.iloc[index, 0] == y_pred_forest[index] else 0 for index in range(len(winning_odds))]
        boost = [winning_odds[index] if y_test.iloc[index, 0] == y_pred_boost[index] else 0 for index in range(len(winning_odds))]
        return logreg, forest, boost

    def won_odds_probas(self, y_test, profil_wr, profil_v, profil_opt, X_id):
        wr = []
        v = []
        opt = []
        for index, element in enumerate(profil_wr):
            if element != 0 and element == y_test.result_id.iloc[index]:
                if element == 1:
                    wr.append(X_id['1'].iloc[index])
                elif element == 2:
                    wr.append(X_id['N'].iloc[index])
                elif element == 3:
                    wr.append(X_id['2'].iloc[index])
            else:
                wr.append(0)
        for index, element in enumerate(profil_v):
            if element != 0 and element == y_test.result_id.iloc[index]:
                if element == 1:
                    v.append(X_id['1'].iloc[index])
                elif element == 2:
                    v.append(X_id['N'].iloc[index])
                elif element == 3:
                    v.append(X_id['2'].iloc[index])
            else:
                v.append(0)
        for index, element in enumerate(profil_opt):
            if element != 0 and element == y_test.result_id.iloc[index]:
                if element == 1:
                    opt.append(X_id['1'].iloc[index])
                elif element == 2:
                    opt.append(X_id['N'].iloc[index])
                elif element == 3:
                    opt.append(X_id['2'].iloc[index])
            else:
                opt.append(0)
        return wr, v, opt


    def accuracy_score(self, y_test, y_pred_logreg, y_pred_forest, y_pred_boost):
        return accuracy_score(y_test, y_pred_logreg), accuracy_score(y_test, y_pred_forest), accuracy_score(y_test, y_pred_boost)

    def accuracy_score_probas(self, y_test, profil_wr, profil_v, profil_opt):
        accuracy_score_wr, accuracy_score_v, accuracy_score_opt = "Pas de match", "Pas de match", "Pas de match"
        wr = np.sum([1 for x in profil_wr if x != 0])
        v =  np.sum([1 for x in profil_v if x != 0])
        opt = np.sum([1 for x in profil_opt if x != 0])
        if wr != 0:
            accuracy_score_wr = np.sum([1 if y_test.result_id.iloc[i] == profil_wr[i] else 0 for i in range(len(y_test))]) / wr
        if v !=0 :
            accuracy_score_v = np.sum([1 if y_test.result_id.iloc[i] == profil_v[i] else 0 for i in range(len(y_test))]) / v
        if opt != 0:
            accuracy_score_opt = np.sum([1 if y_test.result_id.iloc[i] == profil_opt[i] else 0 for i in range(len(y_test))]) / opt
        return accuracy_score_wr, accuracy_score_v, accuracy_score_opt

    def display_results(self, accuracy_score_logreg, accuracy_score_forest, accuracy_score_boost, won_odds_logreg, won_odds_forest, won_odds_boost, test_length):
        print('accuracy score Logistic Regression', accuracy_score_logreg * 100, '% === bilan de', sum(won_odds_logreg) - test_length, '€ sur', test_length, 'paris === rentabilite de', "%.2f" % (100* ((sum(won_odds_logreg)/test_length) - 1)), '%')
        print('accuracy score Random Forest', accuracy_score_forest * 100, '% === bilan de', sum(won_odds_forest) - test_length, '€ sur', test_length, 'paris === rentabilite de', "%.2f" % (100* ((sum(won_odds_forest)/test_length) - 1)), '%')
        print('accuracy score XG BOOST', accuracy_score_boost * 100, '% === bilan de', sum(won_odds_boost) - test_length, '€ sur', test_length, 'paris === rentabilite de', "%.2f" % (100* ((sum(won_odds_boost)/test_length) - 1)), '%')

    def display_results_probas(self, accuracy_score_wr, accuracy_score_v, accuracy_score_opt, won_odds_wr, won_odds_v, won_odds_opt, profil_wr, profil_v, profil_opt):
        len_wr = len([x for x in profil_wr if x != 0])
        len_v = len([x for x in profil_v if x != 0])
        len_opt = len([x for x in profil_opt if x != 0])
        if len_wr > 0:
            print('accuracy score profil winning ratio', accuracy_score_wr * 100, '% === bilan de', sum(won_odds_wr) - len_wr, '€ sur', len_wr, 'paris === rentabilite de', "%.2f" % (100* ((sum(won_odds_wr)/len_wr) - 1)), '%')
        if len_v > 0:
            print('accuracy score profil value', accuracy_score_v * 100, '% === bilan de', sum(won_odds_v) - len_v, '€ sur', len_v, 'paris === rentabilite de', "%.2f" % (100* ((sum(won_odds_v)/len_v) - 1)), '%')
        if len_opt > 0:
            print('accuracy score profil optimisation', accuracy_score_opt * 100, '% === bilan de', sum(won_odds_opt) - len_opt, '€ sur', len_opt, 'paris === rentabilite de', "%.2f" % (100* ((sum(won_odds_opt)/len_opt) - 1)), '%')

    def display_results_predictions(self, X_id_pred, profil_wr_predictive, profil_v_predictive, profil_opt_predictive, display):
        X_id_pred['profil_wr'] = profil_wr_predictive
        X_id_pred['profil_v'] = profil_v_predictive
        X_id_pred['profil_opt'] = profil_opt_predictive
        if display:
            print(X_id_pred[['game_id', 'profil_wr', 'profil_v', 'profil_opt']])
        return X_id_pred.drop(['result_id'], axis=1)

    def confusion_matrix(self, y_test, y_pred_logreg, y_pred_forest, y_pred_boost):
        print(confusion_matrix(y_test, y_pred_logreg))
        print(confusion_matrix(y_test, y_pred_forest))
        print(confusion_matrix(y_test, y_pred_boost))
