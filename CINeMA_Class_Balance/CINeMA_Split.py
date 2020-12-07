"""
Usage:
  CINeMA.py tts -d <data_dir> -s <model_save_file_path> [--w <data_dir>] [--e <epochs>]
  CINeMA.py tts_features -d <data_dir> -s <model_save_file_path> [--w <data_dir>]
  CINeMA.py tts_rf -d <data_dir> -s <model_save_file_path> [--w <data_dir>]
  CINeMA.py tts_rf_features -d <data_dir> -s <model_save_file_path> [--w <data_dir>]
  CINeMA.py -h

Options:
  -h --help                     Show this screen.
  -d <data_dir>                 Data directory path.
  --w <data_dir>                Data directory path to be split
  --st <similarity_threshold>   Similarity threshold [default: 600].
  --pt <percent threshold>      Percent threshold [default: 80].
  --e <epochs>                  Number of epochs [default: 30]
  -s <model_save_file_path>     File path where model is saved
  --version                     Show version
"""

from docopt import docopt
import sys
import os

# Adding modules to python path, so that they can be imported.
ROOT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.insert(0, ROOT_PATH + "/model/algorithmic")
sys.path.insert(0, ROOT_PATH + "/model/ml")
sys.path.insert(0, ROOT_PATH + "/utils")

value = 7
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
random.seed(value)
import numpy as np
np.random.seed(value)
import DataParser as dp
import ReportGenerator as rg
import Features as ft
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

def get_valid_gt(in_gt, sample):
    out_gt = []
    for (peak_num, name, confidence) in in_gt:
        if name in hits and sample[peak_num - 1]['name'] == name:
            out_gt.append((peak_num, name, confidence))
    return out_gt

def translate_pred(pred):
    ret  = []
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1

    for p in pred:
        if (p[0] == 0 and p[1] == 1):
            ret.append(2)
        elif (p[0] == 1 and p[1] == 0):
            ret.append(0)
        else:
            ret.append(-1)
    return ret

def get_matrix(pred, gt):
    TP = np.sum(np.logical_and(pred == 2, gt == 2))
    TN = np.sum(np.logical_and(pred == 0, gt == 0))
    FP = np.sum(np.logical_and(pred == 2, gt == 0))
    FN = np.sum(np.logical_and(pred == 0, gt == 2))
    confusion_matrix = np.matrix([[TN, FP], [FN, TP]])
    return confusion_matrix

if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    if args['tts']:
        import MLModel as ml
        parser1 = dp.DataParser(args['-d'], dp.Mode.TRAIN)
        all_data = parser1.parseData()
        X = []
        Y = []
        all_names = []
        all_hits = []
        all_samples = []
        all_valid_gt = []
        for data in all_data:
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            ground_truth = data['ground_truth']
            valid_gt = get_valid_gt(ground_truth, sample)
            all_names.append(sample_name)
            all_hits.append(hits)
            all_samples.append(sample)
            all_valid_gt.append(valid_gt)
            for (peak_num, name, confidence) in valid_gt:
                if confidence == 2:
                    gt = 2
                elif confidence == 0:
                    gt = 0
                sample_spectrum = sample[peak_num - 1]['spectrum']
                sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
                hit_spectrum = hits[name]['spectrum']
                hit_spectrum = hit_spectrum / np.linalg.norm(hit_spectrum)
                hit_spectrum[0:49] = 0
                X.append(np.concatenate((sample_spectrum, hit_spectrum)))
                Y.append(gt)

        if args['--w']:
            parser2 = dp.DataParser(args['--w'], dp.Mode.TRAIN)
            all_data = parser2.parseData()
            X_2 = []
            Y_2 = []
            all_names = []
            all_hits = []
            all_samples = []
            all_valid_gt = []
            for data in all_data:
                sample_name = data['name']
                hits = data['hits']
                sample = data['sample']
                ground_truth = data['ground_truth']
                valid_gt = get_valid_gt(ground_truth, sample)
                all_names.append(sample_name)
                all_hits.append(hits)
                all_samples.append(sample)
                all_valid_gt.append(valid_gt)
                for (peak_num, name, confidence) in valid_gt:
                    if confidence == 2:
                        gt = 2
                    elif confidence == 0:
                        gt = 0
                    sample_spectrum = sample[peak_num - 1]['spectrum']
                    sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
                    hit_spectrum = hits[name]['spectrum']
                    hit_spectrum = hit_spectrum / np.linalg.norm(hit_spectrum)
                    hit_spectrum[0:49] = 0
                    X_2.append(np.concatenate((sample_spectrum, hit_spectrum)))
                    Y_2.append(gt)

            indicies = []
            for i in range(len(X_2)):
                indicies.append(i)

            X_train_2, X_test_2, y_train_2, y_test_2, indicies_train2, indicies_test2 = train_test_split(X_2, Y_2, indicies, test_size=0.20, random_state=100)

            for i in X_train_2:
                X.append(i)
            for i in y_train_2:
                Y.append(i)

            dnn = ml.DNNModel(args['-s'])

            X = np.array(X)
            Y = np.array(Y)
            i_class0 = np.where(Y == 0)[0]
            i_class2 = np.where(Y == 2)[0]
            n_class0 = len(i_class0)
            n_class2 = len(i_class2)
            if n_class0 > n_class2:
                i_class0_downsampled = np.random.choice(i_class0, size=n_class2, replace=False)
                y = np.concatenate((Y[i_class0_downsampled], Y[i_class2]))
                x_train = np.concatenate((X[i_class0_downsampled], X[i_class2]))
            if n_class2 > n_class0:
                i_class2_downsampled = np.random.choice(i_class2, size=n_class0, replace=False)
                y = np.concatenate((Y[i_class0], Y[i_class2_downsampled]))
                x_train = np.concatenate((X[i_class0], X[i_class2_downsampled]))
            y_train = []
            y_test_final = []
            for i in y:
                if i == 0:
                    y_train.append([1, 0])
                if i == 2:
                    y_train.append([0, 1])
            for i in y_test_2:
                if i == 0:
                    y_test_final.append([1, 0])
                if i == 2:
                    y_test_final.append([0, 1])

            dnn.cross(np.array(x_train), np.array(y_train), int(args['--e']))
            dnn.train(np.array(x_train), np.array(y_train), int(args['--e']))
            dnn.save()
            dnn.load()
            print(n_class0, n_class2)
            print(len(x_train))
            dnn.evaluate(np.array(X_test_2), np.array(y_test_final))
            prediction = dnn.predict(np.array(X_test_2))
            pred = translate_pred(prediction)
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv3(all_valid_gt, pred, indicies_test2)
            rp.report_pdf3(all_hits, all_samples, all_valid_gt, pred, indicies_test2)

        else:
            indicies = []
            for i in range(len(X)):
                indicies.append(i)
            X_train, X_test, y_train, y_test, indicies_train, indicies_test = train_test_split(X, Y, indicies, test_size=0.20, random_state=100)
            dnn = ml.DNNModel(args['-s'])
            X = np.array(X_train)
            Y = np.array(y_train)
            i_class0 = np.where(Y == 0)[0]
            i_class2 = np.where(Y == 2)[0]
            n_class0 = len(i_class0)
            n_class2 = len(i_class2)
            if n_class0 > n_class2:
                i_class0_downsampled = np.random.choice(i_class0, size=n_class2, replace=False)
                y = np.concatenate((Y[i_class0_downsampled], Y[i_class2]))
                x_train = np.concatenate((X[i_class0_downsampled], X[i_class2]))
            if n_class2 > n_class0:
                i_class2_downsampled = np.random.choice(i_class2, size=n_class0, replace=False)
                y = np.concatenate((Y[i_class0], Y[i_class2_downsampled]))
                x_train = np.concatenate((X[i_class0], X[i_class2_downsampled]))
            y_train = []
            y_test_final = []
            for i in y:
                if i == 0:
                    y_train.append([1, 0])
                if i == 2:
                    y_train.append([0, 1])
            for i in y_test:
                if i == 0:
                    y_test_final.append([1, 0])
                if i == 2:
                    y_test_final.append([0, 1])
            print(n_class0, n_class2)
            print(len(x_train))
            dnn.cross(np.array(x_train), np.array(y_train), int(args['--e']))
            dnn.train(np.array(x_train), np.array(y_train), int(args['--e']))
            dnn.save()
            dnn.load()
            dnn.evaluate(np.array(X_test), np.array(y_test_final))
            prediction = dnn.predict(np.array(X_test))
            pred = translate_pred(prediction)
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv3(all_valid_gt, pred, indicies_test)
            rp.report_pdf3(all_hits, all_samples, all_valid_gt, pred, indicies_test)

    if args['tts_features']:
        import MLModel_Features as mlf
        parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
        all_data = parser.parseData()
        all_names = []
        all_hits = []
        all_samples = []
        all_valid_gt = []
        df = pd.DataFrame(
            columns=['Name', 'Confidence', 'Score', '#TOP Hit Ions in Compound', '#Top Compound Ions in Hit',
                     'Molecular Ion in Compound', 'Correlation Percentage'])
        table_creator = ft.gt_table()
        for data in all_data:
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            ground_truth = data['ground_truth']
            valid_gt = get_valid_gt(ground_truth, sample)
            all_names.append(sample_name)
            all_hits.append(hits)
            all_samples.append(sample)
            all_valid_gt.append(valid_gt)
            for (peak_num, name, confidence) in valid_gt:
                s, x, y, z, m = table_creator.table(hits[name], sample[peak_num - 1])
                df.loc[-1] = [str(name), confidence, s, x, y, z, m]
                df.index = df.index + 1
                df = df.sort_index()
        df = df.iloc[::-1]
        df = df.drop(['Name'], axis=1)

        if args['--w']:
            parser = dp.DataParser(args['--w'], dp.Mode.TRAIN)
            all_data = parser.parseData()
            all_names = []
            all_hits = []
            all_samples = []
            all_valid_gt = []
            df2 = pd.DataFrame(
                columns=['Name', 'Confidence', 'Score', '#TOP Hit Ions in Compound', '#Top Compound Ions in Hit',
                         'Molecular Ion in Compound', 'Correlation Percentage'])
            table_creator = ft.gt_table()
            for data in all_data:
                sample_name = data['name']
                hits = data['hits']
                sample = data['sample']
                ground_truth = data['ground_truth']
                valid_gt = get_valid_gt(ground_truth, sample)
                all_names.append(sample_name)
                all_hits.append(hits)
                all_samples.append(sample)
                all_valid_gt.append(valid_gt)
                for (peak_num, name, confidence) in valid_gt:
                    s, x, y, z, m = table_creator.table(hits[name], sample[peak_num - 1])
                    df2.loc[-1] = [str(name), confidence, s, x, y, z, m]
                    df2.index = df2.index + 1
                    df2 = df2.sort_index()
            df2 = df2.iloc[::-1]
            df2 = df2.drop(['Name'], axis=1)

            indicies = []
            for i in range(len(df2)):
                indicies.append(i)
            X_train2, X_test2, indicies_train, indicies_test = train_test_split(df2, indicies, test_size = 0.2, random_state = 100)

            final_df = pd.concat([df, X_train2], ignore_index=True)

            df_low = final_df[final_df.Confidence == 0]
            df_high = final_df[final_df.Confidence == 2]
            if len(df_low) > len(df_high):
                df_majority_downsampled = resample(df_low, replace=False, n_samples=len(df_high), random_state=100)
                final_df = pd.concat([df_majority_downsampled, df_high])
            if len(df_high) > len(df_low):
                df_majority_downsampled = resample(df_high, replace=False, n_samples=len(df_low), random_state=100)
                final_df = pd.concat([df_low, df_majority_downsampled])

            set2 = final_df.values
            X_train = set2[:, 1:6].astype(float)
            y_train = set2[:, 0]

            test_set = X_test2.values
            X_test = test_set[:, 1:6].astype(float)
            y_test = test_set[:, 0]

            encoder = LabelEncoder()
            encoder.fit(y_train)
            y_train = encoder.transform(y_train)

            encoder2 = LabelEncoder()
            encoder2.fit(y_test)
            y_test = encoder2.transform(y_test)

            model = mlf.MLFModel(args['-s'])
            model.cross(X_train, y_train)
            model.train(X_train, y_train)
            model.save()
            model.load()
            model.test(X_test, y_test)
            pred = model.predict(X_test)
            print(len(df_low), len((df_high)))
            print(len(X_train))
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv3(all_valid_gt, pred, indicies_test)
            rp.report_pdf3(all_hits, all_samples, all_valid_gt, pred, indicies_test)

        else:
            indicies = []
            for i in range(len(df)):
                indicies.append(i)
            df_train, df_test, indicies_train, indicies_test = train_test_split(df, indicies, test_size=0.20, random_state=100)
            df_low = df_train[df_train.Confidence == 0]
            df_high = df_train[df_train.Confidence == 2]
            if len(df_low) > len(df_high):
                df_majority_downsampled = resample(df_low, replace=False, n_samples=len(df_high), random_state=100)
                df_train = pd.concat([df_majority_downsampled, df_high])
            if len(df_high) > len(df_low):
                df_majority_downsampled = resample(df_high, replace=False, n_samples=len(df_low), random_state=100)
                df_train = pd.concat([df_low, df_majority_downsampled])
            set = df_train.values
            x_train = set[:, 1:6].astype(float)
            y_train = set[:, 0]

            print(len(df_low), len(df_high))
            print(len(x_train))
            set_test = df_test.values
            x_test = set_test[:, 1:6].astype(float)
            y_test = set_test[:, 0]

            encoder = LabelEncoder()
            encoder.fit(y_train)
            y_train = encoder.transform(y_train)

            encoder2 = LabelEncoder()
            encoder2.fit(y_test)
            y_test = encoder.transform(y_test)

            model = mlf.MLFModel(args['-s'])
            model.cross(x_train, y_train)
            model.train(x_train, y_train)
            model.save()
            model.load()
            model.test(x_test, y_test)
            pred = model.predict(x_test)
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv3(all_valid_gt, pred, indicies_test)
            rp.report_pdf3(all_hits, all_samples, all_valid_gt, pred, indicies_test)

    elif args['tts_rf']:
        import Intensities_Tree as rfi
        parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
        all_data = parser.parseData()
        X = []
        Y = []
        all_names = []
        all_hits = []
        all_samples = []
        all_valid_gt = []
        for data in all_data:
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            ground_truth = data['ground_truth']
            valid_gt = get_valid_gt(ground_truth, sample)
            all_names.append(sample_name)
            all_hits.append(hits)
            all_samples.append(sample)
            all_valid_gt.append(valid_gt)
            for (peak_num, name, confidence) in valid_gt:
                if confidence == 2:
                    gt = 2
                elif confidence == 0:
                    gt = 0
                sample_spectrum = sample[peak_num - 1]['spectrum']
                sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
                hit_spectrum = hits[name]['spectrum']
                hit_spectrum = hit_spectrum / np.linalg.norm(hit_spectrum)
                hit_spectrum[0:49] = 0
                X.append(np.concatenate((sample_spectrum, hit_spectrum)))
                Y.append(gt)

        if args['--w']:
            parser2 = dp.DataParser(args['--w'], dp.Mode.TRAIN)
            all_data = parser2.parseData()
            X_2 = []
            Y_2 = []
            all_names = []
            all_hits = []
            all_samples = []
            all_valid_gt = []
            for data in all_data:
                sample_name = data['name']
                hits = data['hits']
                sample = data['sample']
                ground_truth = data['ground_truth']
                valid_gt = get_valid_gt(ground_truth, sample)
                all_names.append(sample_name)
                all_hits.append(hits)
                all_samples.append(sample)
                all_valid_gt.append(valid_gt)
                for (peak_num, name, confidence) in valid_gt:
                    if confidence == 2:
                        gt = 2
                    elif confidence == 0:
                        gt = 0
                    sample_spectrum = sample[peak_num - 1]['spectrum']
                    sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
                    hit_spectrum = hits[name]['spectrum']
                    hit_spectrum = hit_spectrum / np.linalg.norm(hit_spectrum)
                    hit_spectrum[0:49] = 0
                    X_2.append(np.concatenate((sample_spectrum, hit_spectrum)))
                    Y_2.append(gt)

            indicies = []
            for i in range(len(X_2)):
                indicies.append(i)
            X_train_2, X_test_2, y_train_2, y_test_2, indicies_train_2, indicies_test_2 = train_test_split(X_2, Y_2, indicies, test_size=0.20, random_state=100)


            for i in X_train_2:
                X.append(i)
            for i in y_train_2:
                Y.append(i)

            print(len(X))
            X_train = np.array(X)
            y_train = np.array(Y)
            i_class0 = np.where(y_train == 0)[0]
            i_class2 = np.where(y_train == 2)[0]
            n_class0 = len(i_class0)
            n_class2 = len(i_class2)
            if n_class0 > n_class2:
                i_class0_downsampled = np.random.choice(i_class0, size=n_class2, replace=False)
                y = np.concatenate((y_train[i_class0_downsampled], y_train[i_class2]))
                x_train = np.concatenate((X_train[i_class0_downsampled], X_train[i_class2]))
            if n_class2 > n_class0:
                i_class2_downsampled = np.random.choice(i_class2, size=n_class0, replace=False)
                y = np.concatenate((y_train[i_class0], y_train[i_class2_downsampled]))
                x_train = np.concatenate((X_train[i_class0], X_train[i_class2_downsampled]))

            print(n_class0, n_class2)
            print(len(x_train))
            model = rfi.model_train(x_train, y, args['-s'])
            pred = rfi.model_test(X_test_2, y_test_2, args['-s'])
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv3(all_valid_gt, pred, indicies_test_2)
            rp.report_pdf3(all_hits, all_samples, all_valid_gt, pred, indicies_test_2)

        else:
            indicies = []
            for i in range(len(X)):
                indicies.append(i)
            X_train, X_test, y_train, y_test, indicies_train, indicies_test = train_test_split(X, Y, indicies, test_size=0.20, random_state=100)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            i_class0 = np.where(y_train == 0)[0]
            i_class2 = np.where(y_train == 2)[0]
            n_class0 = len(i_class0)
            n_class2 = len(i_class2)
            if n_class0 > n_class2:
                i_class0_downsampled = np.random.choice(i_class0, size=n_class2, replace=False)
                y = np.concatenate((y_train[i_class0_downsampled], y_train[i_class2]))
                x_train = np.concatenate((X_train[i_class0_downsampled], X_train[i_class2]))
            if n_class2 > n_class0:
                i_class2_downsampled = np.random.choice(i_class2, size=n_class0, replace=False)
                y = np.concatenate((y_train[i_class0], y_train[i_class2_downsampled]))
                x_train = np.concatenate((X_train[i_class0], X_train[i_class2_downsampled]))
            print(n_class0, n_class2)
            print(len(x_train))
            model = rfi.model_train(x_train, y, args['-s'])
            pred = rfi.model_test(X_test, y_test, args['-s'])
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv3(all_valid_gt, pred, indicies_test)
            rp.report_pdf3(all_hits, all_samples, all_valid_gt, pred, indicies_test)


    elif args['tts_rf_features']:
        import DT_Model as dt
        parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
        all_data = parser.parseData()
        all_names = []
        all_hits = []
        all_samples = []
        all_valid_gt = []
        df = pd.DataFrame(
            columns=['Name', 'Confidence', 'Score', '#TOP Hit Ions in Compound', '#Top Compound Ions in Hit',
                     'Molecular Ion in Compound', 'Correlation Percentage'])
        table_creator = ft.gt_table()
        for data in all_data:
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            ground_truth = data['ground_truth']
            valid_gt = get_valid_gt(ground_truth, sample)
            all_names.append(sample_name)
            all_hits.append(hits)
            all_samples.append(sample)
            all_valid_gt.append(valid_gt)
            for (peak_num, name, confidence) in valid_gt:
                s, x, y, z, m = table_creator.table(hits[name], sample[peak_num - 1])
                df.loc[-1] = [str(name), confidence, s, x, y, z, m]
                df.index = df.index + 1
                df = df.sort_index()
        df = df.iloc[::-1]
        df = df.drop(['Name'], axis=1)

        if args['--w']:
            parser2 = dp.DataParser(args['--w'], dp.Mode.TRAIN)
            all_data = parser2.parseData()
            all_names = []
            all_hits = []
            all_samples = []
            all_valid_gt = []
            df2 = pd.DataFrame(
                columns=['Name', 'Confidence', 'Score', '#TOP Hit Ions in Compound', '#Top Compound Ions in Hit',
                         'Molecular Ion in Compound', 'Correlation Percentage'])
            table_creator = ft.gt_table()
            for data in all_data:
                sample_name = data['name']
                hits = data['hits']
                sample = data['sample']
                ground_truth = data['ground_truth']
                valid_gt = get_valid_gt(ground_truth, sample)
                all_names.append(sample_name)
                all_hits.append(hits)
                all_samples.append(sample)
                all_valid_gt.append(valid_gt)
                for (peak_num, name, confidence) in valid_gt:
                    s, x, y, z, m = table_creator.table(hits[name], sample[peak_num - 1])
                    df2.loc[-1] = [str(name), confidence, s, x, y, z, m]
                    df2.index = df2.index + 1
                    df2 = df2.sort_index()
            df2 = df2.iloc[::-1]
            df2 = df2.drop(['Name'], axis=1)

            indicies = []
            for i in range(len(df2['Confidence'])):
                indicies.append(i)
            X_train2, X_test2, indicies_train, indicies_test = train_test_split(df2, indicies, test_size=0.20, random_state=100)

            final_df = pd.concat([df, X_train2], ignore_index=True)

            df_low = final_df[final_df.Confidence == 0]
            df_high = final_df[final_df.Confidence == 2]
            if len(df_low) > len(df_high):
                df_majority_downsampled = resample(df_low, replace=False, n_samples=len(df_high), random_state=100)
                final_df = pd.concat([df_majority_downsampled, df_high])
            if len(df_high) > len(df_low):
                df_majority_downsampled = resample(df_high, replace=False, n_samples=len(df_low), random_state=100)
                final_df = pd.concat([df_low, df_majority_downsampled])

            X_train = final_df.drop(['Confidence'], axis = 1)
            y_train = final_df['Confidence']

            X_test = X_test2.drop(['Confidence'], axis = 1)
            y_test = X_test2['Confidence']

            y_train = y_train.astype('int')
            y_test = y_test.astype('int')

            print(len(df_high), len(df_low))
            print(len(X_train))
            model = dt.model_train(X_train, y_train, args['-s'])
            pred = dt.model_test(X_test, y_test, args['-s'])
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv3(all_valid_gt, pred, indicies_test)
            rp.report_pdf3(all_hits, all_samples, all_valid_gt, pred, indicies_test)

        else:
            indicies = []
            for i in range(len(df['Confidence'])):
                indicies.append(i)
            df_train, df_test, indicies_train, indicies_test = train_test_split(df, indicies, test_size=0.20, random_state=100)
            df_low = df_train[df_train.Confidence == 0]
            df_high = df_train[df_train.Confidence == 2]
            print(len(df_low), len(df_high))
            if len(df_low) > len(df_high):
                df_majority_downsampled = resample(df_low, replace=False, n_samples=len(df_high), random_state=100)
                df_train = pd.concat([df_majority_downsampled, df_high])
            if len(df_high) > len(df_low):
                df_majority_downsampled = resample(df_high, replace=False, n_samples=len(df_low), random_state=100)
                df_train = pd.concat([df_low, df_majority_downsampled])
            x_train = df_train.drop(['Confidence'], axis = 1)
            x_test = df_test.drop(['Confidence'], axis = 1)
            print(len(x_train))
            y_train = df_train['Confidence'].astype('int')
            y_test = df_test['Confidence'].astype('int')
            model = dt.model_train(x_train, y_train, args['-s'])
            pred = dt.model_test(x_test, y_test, args['-s'])
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv3(all_valid_gt, pred, indicies_test)
            rp.report_pdf3(all_hits, all_samples, all_valid_gt, pred, indicies_test)
