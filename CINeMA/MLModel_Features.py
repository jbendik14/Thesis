value = 7
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
random.seed(value)
import numpy as np
np.random.seed(value)
import tensorflow as tf
tf.random.set_seed(value)
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from distutils.version import StrictVersion

class MLFModel:
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path

    def train(self, X_train, y_train):
        self.model = Sequential()
        self.model.add(Dense(5, input_dim = 5, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs = 300, batch_size = 10, shuffle = True, verbose = 0)

    def cross(self, X_train, y_train):
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=value)
        all_scores = []
        fig_path = os.path.join(os.getcwd(), "model_performance.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        for train, val in kfold.split(X_train, y_train):
            model = Sequential()
            model.add(Dense(5, input_dim=5, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.summary()
            history = model.fit(X_train[train], y_train[train], epochs = 300, batch_size = 10, validation_data=(X_train[val], y_train[val]), verbose = 0)
            scores = model.evaluate(X_train[val], y_train[val], verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            all_scores.append(scores[1] * 100)

            fig = plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            pdf.savefig(fig)
            plt.close()

        pdf.close()

        print('%.2f%% (+/- %.2f%%)' % (np.mean(all_scores), np.std(all_scores)))

    def save(self):
        self.model.save(self.model_path)
        print("Saved model to disk.")

    def load(self):
        self.model = load_model(self.model_path)

    def create_initial_model(self, activation = 'relu'):
        model = Sequential()
        model.add(Dense(5, input_dim=5, activation = activation))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def grid(self, x, y):
        seed = 7
        np.random.seed(seed)
        model = KerasClassifier(build_fn = self.create_initial_model, verbose = 0)
        activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid']
        batch_size = [10, 20, 40, 64, 80, 100, 128]
        epochs = [10, 50, 100, 300]
        param_grid = dict(activation = activation, batch_size = batch_size, epochs = epochs)
        grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1)
        grid_result = grid.fit(x, y)
        print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))

    def predict(self, X_test):
        pred = self.model.predict(X_test)
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 2
        return pred

    def test(self, X_test, y_test):
        classes = ['Low', 'High']
        pred = self.model.predict(X_test)

        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1

        fig_path = os.path.join(os.getcwd(), "measures.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        fig = plt.figure()

        matrix = confusion_matrix(y_test, pred)
        sns.heatmap(matrix, annot = True, cbar = True, xticklabels = classes, yticklabels = classes, cmap = 'Blues', fmt = 'g')
        plt.ylabel('True Match Quality')
        plt.xlabel('Predicted Match Quality')
        pdf.savefig(fig)
        plt.close()

        probabilities = self.model.predict(X_test)
        area = roc_auc_score(y_test, probabilities)
        fp, tp, _ = roc_curve(y_test, probabilities)
        fig = plt.figure()
        plt.plot(fp, tp, marker = '.', label = 'ROC (area = {:.3f})'.format(area))
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc = 'best')
        pdf.savefig(fig)
        plt.close()

        pdf.close()

        score = self.model.evaluate(X_test, y_test, verbose = 0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))
