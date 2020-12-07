import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def model_train(X_train, y_train, filename):
    rf_model = RandomForestClassifier(max_depth = 4, random_state = 100)
    rf_model.fit(X_train, y_train)

    labels = []

    for i in range(1000):
        labels.append('Sample m/z {}'.format(str(i+1)))
    for i in range(1000):
        labels.append('Hit m/z {}'.format(str(i+1)))

    rf_Features = dict(zip(labels, rf_model.feature_importances_))
    rf_Importance = sorted(rf_Features, key=rf_Features.get, reverse=True)
    rf_Importance = {i: rf_Features[i] for i in rf_Importance}
    #print(rf_Importance)

    # example_tree = rf_model.estimators_[0]
    # my_tree = Source(tree.export_graphviz(example_tree, out_file = None, class_names = ['Low', 'High'], filled = True))
    # my_tree.format = 'png'
    # my_tree.render('epa_tree_intensities')
    # display(SVG(my_tree.pipe(format = 'svg')))

    saved_forest = filename
    joblib.dump(rf_model, saved_forest)


def model_test(X_test, y_test, filename):
    rf_model = joblib.load(filename)
    rf_y_pred = rf_model.predict(X_test)

    matrix = pd.DataFrame(confusion_matrix(y_test, rf_y_pred), columns = ['Predicted Low', 'Predicted High'], index = ['True Low', 'True High'])
    fig_path = os.path.join(os.getcwd(), "measures.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
    fig = plt.figure()
    sns.heatmap(matrix, annot = True, cbar = True, cmap = 'Blues', fmt = 'g')
    plt.ylabel('True Match Quality')
    plt.xlabel('Predicted Match Quality')
    pdf.savefig(fig)
    plt.close()
    print(matrix)

    probabilities = rf_model.predict_proba(X_test)
    probabilities = probabilities[:, 1]
    score = roc_auc_score(y_test, probabilities)
    fig = plt.figure()
    fp, tp, _ = roc_curve(y_test, probabilities, pos_label = 2)
    plt.plot(fp, tp, marker = '.', label = 'ROC (area = {:.3f})'.format(score))
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = 'best')
    pdf.savefig(fig)
    plt.close()

    print('Accuracy:', accuracy_score(y_test, rf_y_pred)*100)

    pdf.close()

    return rf_y_pred

def model_predict(X, filename):
    rf_model = joblib.load(filename)
    rf_y_pred = rf_model.predict(X)
    return rf_y_pred

def grid(x, y):
    rf_model = RandomForestClassifier(random_state = 100)
    params = {'bootstrap': [True, False],
    'max_depth': [4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
    clf = GridSearchCV(estimator = rf_model, param_grid = params, n_jobs = -1)
    grid_result = clf.fit(x, y)
    print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))



