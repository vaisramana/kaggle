import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
import dataHandler as dh

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)


def recover_train_test_target(combinedData):
    train0 = pd.read_csv('../input/train.csv')
    
    targets = train0.Survived
    train = combinedData.head(891)
    test = combinedData.iloc[891:]
    
    return train, test, targets
    
def randomForest(train, test, targets):
    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf = clf.fit(train, targets)
    model = SelectFromModel(clf, prefit=True)
    
    run_gs = False
    if run_gs:
        parameter_grid = {
                     'max_depth' : [4, 6, 8],
                     'n_estimators': [50, 10],
                     'max_features': ['sqrt', 'auto', 'log2'],
                     'min_samples_split': [1, 3, 10],
                     'min_samples_leaf': [1, 3, 10],
                     'bootstrap': [True, False],
                     }
        forest = RandomForestClassifier()
        cross_validation = StratifiedKFold(targets, n_folds=5)

        grid_search = GridSearchCV(forest,
                                   scoring='accuracy',
                                   param_grid=parameter_grid,
                                   cv=cross_validation)

        grid_search.fit(train, targets)
        model = grid_search
        parameters = grid_search.best_params_

        print('Best score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))
    else: 
        parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                      'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
        
        model = RandomForestClassifier(**parameters)
        model.fit(train, targets)

    output = model.predict(test).astype(int)
    df_output = pd.DataFrame()
    aux = pd.read_csv('../input/test.csv')
    df_output['PassengerId'] = aux['PassengerId']
    df_output['Survived'] = output
    df_output[['PassengerId','Survived']].to_csv('../predition.csv',index=False)


combinedData = dh.dataHandler()
train, test, targets = recover_train_test_target(combinedData)
randomForest(train, test, targets)



