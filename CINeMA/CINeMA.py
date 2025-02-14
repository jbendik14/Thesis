"""
Usage:
  CINeMA.py ml (train|test|predict) -d <data_dir> -s <model_save_file_path> [--e <epochs>]
  CINeMA.py ml_features (train|test|predict) -d <data_dir> -s <model_save_file_path> [--e <epochs>]
  CINeMA.py rf (train|test|predict) -d <data_dir> -s <model_save_file_path> [--e <epochs>]
  CINeMA.py rf_features (train|test|predict) -d <data_dir> -s <model_save_file_path> [--e <epochs>]
  CINeMA.py algo -d <data_dir> (test|predict) [--st <similarity_threshold>] [--pt <percent_threshold>]
  CINeMA.py scatter -d <data_dir> -w <path_to_csv>
  CINeMA.py -h

Options:
  -h --help                     Show this screen.
  -d <data_dir>                 Data directory path.
  -w <path_to_csv>              Path to csv.
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

import DataParser as dp
import ReportGenerator as rg
import MWvsRT as wr
import numpy as np
import Features as ft
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    #print(args)
    if args['ml']:
        import MLModel as ml
        if args['train']:
            parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
            all_data = parser.parseData()
            X = []
            Y = []
            for data in all_data:
                sample_name = data['name']
                hits = data['hits']
                sample = data['sample']
                ground_truth = data['ground_truth']
                valid_gt = get_valid_gt(ground_truth, sample)
                for (peak_num, name, confidence) in valid_gt:
                    if confidence == 2:
                        gt = [0, 1]
                    elif confidence == 0:
                        gt = [1, 0]
                    sample_spectrum = sample[peak_num - 1]['spectrum']
                    sample_spectrum = sample_spectrum/np.linalg.norm(sample_spectrum)
                    hit_spectrum = hits[name]['spectrum']
                    hit_spectrum = hit_spectrum/np.linalg.norm(hit_spectrum)
                    hit_spectrum[0:49] = 0
                    X.append(np.concatenate((sample_spectrum, hit_spectrum)))
                    Y.append(gt)
            dnn = ml.DNNModel(args['-s'])
            dnn.cross(np.array(X), np.array(Y), int(args['--e']))
            dnn.train(np.array(X), np.array(Y), int(args['--e']))
            print(len(X))
            #dnn.grid(np.array(X), np.array(Y))
            dnn.save()
        elif args['test']:
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
                        gt = [0, 1]
                    elif confidence == 0:
                        gt = [1, 0]
                    sample_spectrum = sample[peak_num - 1]['spectrum']
                    sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
                    hit_spectrum = hits[name]['spectrum']
                    hit_spectrum = hit_spectrum / np.linalg.norm(hit_spectrum)
                    hit_spectrum[0:49] = 0
                    X.append(np.concatenate((sample_spectrum, hit_spectrum)))
                    Y.append(gt)
            dnn = ml.DNNModel(args['-s'])
            dnn.load()
            print(len(X))
            dnn.evaluate(np.array(X), np.array(Y))
            prediction = dnn.predict(np.array(X))
            pred = translate_pred(prediction)
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv(all_valid_gt, pred)
            rp.report_pdf(all_hits, all_samples, all_valid_gt, pred, all_names)
        elif args['predict']:
            parser = dp.DataParser(args['-d'], dp.Mode.PREDICT)
            data = parser.parseData()
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            peak_num = 0
            pred = []
            X = []
            for compound in sample:
                peak_num = peak_num + 1
                name = compound['name']
                if name in hits:
                    sample_spectrum = compound['spectrum']
                    sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
                    hit_spectrum = hits[name]['spectrum']
                    hit_spectrum = hit_spectrum / np.linalg.norm(hit_spectrum)
                    hit_spectrum[0:49] = 0
                    X.append(np.concatenate((sample_spectrum, hit_spectrum)))
                    pred.append((peak_num, name, 0))
            dnn = ml.DNNModel(args['-s'])
            dnn.load()
            prediction = dnn.predict(np.array(X))
            pred_arr = translate_pred(prediction)
            i = 0
            for Y in pred_arr:
                pred[i] = (pred[i][0], pred[i][1], Y)
                i = i + 1
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv1(sample_name, pred)
            rp.report_pdf1(sample_name, hits, sample, pred)

    if args['ml_features']:
        import MLModel_Features as mlf
        if args['train']:
            parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
            all_data = parser.parseData()
            df = pd.DataFrame(columns=['Name', 'Confidence', 'Score', '#Top Hit Ions in Compound', '#Top Compound Ions in Hit',
                         'Molecular Ion in Compound', 'Correlation Percentage'])
            table_creator = ft.gt_table()
            for data in all_data:
                sample_name = data['name']
                hits = data['hits']
                sample = data['sample']
                ground_truth = data['ground_truth']
                valid_gt = get_valid_gt(ground_truth, sample)
                for (peak_num, name, confidence) in valid_gt:
                    s, x, y, z, m = table_creator.table(hits[name], sample[peak_num - 1])
                    df.loc[-1] = [str(name), confidence, s, x, y, z, m]
                    df.index = df.index + 1
                    df = df.sort_index()
            df = df.iloc[::-1]
            df = df.drop(['Name'], axis=1)
            set = df.values
            X_train = set[:, 1:6].astype(float)
            y_train = set[:, 0]
            encoder = LabelEncoder()
            encoder.fit(y_train)
            y_train = encoder.transform(y_train)
            dnn = mlf.MLFModel(args['-s'])
            dnn.cross(X_train, y_train)
            dnn.train(X_train, y_train)
            #dnn.grid(X_train, y_train)
            print(len(X_train))
            dnn.save()
        if args['test']:
            parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
            all_data = parser.parseData()
            all_names = []
            all_hits = []
            all_samples = []
            all_valid_gt = []
            df = pd.DataFrame(
                columns=['Name', 'Confidence', 'Score', '#Top Hit Ions in Compound', '#Top Compound Ions in Hit',
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
            set = df.values
            X_test = set[:, 1:6].astype(float)
            y_test = set[:, 0]
            encoder = LabelEncoder()
            encoder.fit(y_test)
            y_test = encoder.transform(y_test)
            model = mlf.MLFModel(args['-s'])
            model.load()
            model.test(X_test, y_test)
            pred = model.predict(X_test)
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv(all_valid_gt, pred)
            rp.report_pdf(all_hits, all_samples, all_valid_gt, pred, all_names)
            print(len(y_test))
        elif args['predict']:
            parser = dp.DataParser(args['-d'], dp.Mode.PREDICT)
            data = parser.parseData()
            peak_num = 0
            pred = []
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            df = pd.DataFrame(
                columns=['Name', 'Score', '#Top Hit Ions in Compound', '#Top Compound Ions in Hit',
                         'Molecular Ion in Compound', 'Correlation Percentage'])
            table_creator = ft.gt_table()
            model = mlf.MLFModel(args['-s'])
            model.load()
            for compound in sample:
                peak_num = peak_num + 1
                name = compound['name']
                if name in hits:
                    s, x, y, z, m = table_creator.table(hits[name], compound)
                    df.loc[-1] = [str(name), s, x, y, z, m]
                    df.index = df.index + 1
                    df = df.sort_index()
                    df = df.iloc[::-1]
                    df = df.drop(['Name'], axis=1)
                    set = df.values
                    X = set[0:5].astype(float)
                    prediction = model.predict(X)
                    pred.append((peak_num, name, prediction))
                df = pd.DataFrame(columns=['Name', 'Score', '#Top Hit Ions in Compound', '#Top Compound Ions in Hit',
                                        'Molecular Ion in Compound', 'Correlation Percentage'])
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv1(sample_name, pred)
            rp.report_pdf1(sample_name, hits, sample, pred)

    elif args['rf']:
        import Intensities_Tree as rfi
        if args['train']:
            parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
            all_data = parser.parseData()
            X_train = []
            y_train = []
            for data in all_data:
                sample_name = data['name']
                hits = data['hits']
                sample = data['sample']
                ground_truth = data['ground_truth']
                valid_gt = get_valid_gt(ground_truth, sample)
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
                    X_train.append(np.concatenate((sample_spectrum, hit_spectrum)))
                    y_train.append(gt)
            print(len(X_train))
            model = rfi.model_train(X_train, y_train, args['-s'])
        elif args['test']:
            parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
            all_data = parser.parseData()
            X_test = []
            y_test = []
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
                    X_test.append(np.concatenate((sample_spectrum, hit_spectrum)))
                    y_test.append(gt)
            pred = rfi.model_test(X_test, y_test, args['-s'])
            print(len(X_test))
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv(all_valid_gt, pred)
            rp.report_pdf(all_hits, all_samples, all_valid_gt, pred, all_names)
        elif args['predict']:
            parser = dp.DataParser(args['-d'], dp.Mode.PREDICT)
            data = parser.parseData()
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            peak_num = 0
            pred = []
            for compound in sample:
                peak_num = peak_num + 1
                name = compound['name']
                if name in hits:
                    sample_spectrum = compound['spectrum']
                    sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
                    hit_spectrum = hits[name]['spectrum']
                    hit_spectrum = hit_spectrum / np.linalg.norm(hit_spectrum)
                    hit_spectrum[0:49] = 0
                    X = np.concatenate((sample_spectrum, hit_spectrum))
                    X = X.reshape(1, -1)
                    prediction = rfi.model_predict(X, args['-s'])
                    pred.append((peak_num, name, prediction))
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv1(sample_name, pred)
            rp.report_pdf1(sample_name, hits, sample, pred)

    elif args['rf_features']:
        import DT_Model as dt
        if args['train']:
            parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
            all_data = parser.parseData()
            df = pd.DataFrame(columns=['Name', 'Confidence', 'Score', '#Top Hit Ions in Compound', '#Top Compound Ions in Hit',
                         'Molecular Ion in Compound', 'Correlation Percentage'])
            table_creator = ft.gt_table()
            for data in all_data:
                sample_name = data['name']
                hits = data['hits']
                sample = data['sample']
                ground_truth = data['ground_truth']
                valid_gt = get_valid_gt(ground_truth, sample)
                for (peak_num, name, confidence) in valid_gt:
                    s, x, y, z, m = table_creator.table(hits[name], sample[peak_num - 1])
                    df.loc[-1] = [str(name), confidence, s, x, y, z, m]
                    df.index = df.index + 1
                    df = df.sort_index()
            df = df.iloc[::-1]
            df = df.drop(['Name'], axis=1)
            X_train = df.drop(['Confidence'], axis=1)
            y_train = df['Confidence']
            y_train = y_train.astype('int')
            model = dt.model_train(X_train, y_train, args['-s'])
            print(len(X_train))
        elif args['test']:
            parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
            all_data = parser.parseData()
            all_names = []
            all_hits = []
            all_samples = []
            all_valid_gt = []
            df = pd.DataFrame(
                columns=['Name', 'Confidence', 'Score', '#Top Hit Ions in Compound', '#Top Compound Ions in Hit',
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
            X_test = df.drop(['Confidence'], axis=1)
            y_test = df['Confidence']
            y_test = y_test.astype('int')
            pred = dt.model_test(X_test, y_test, args['-s'])
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv(all_valid_gt, pred)
            rp.report_pdf(all_hits, all_samples, all_valid_gt, pred, all_names)
            print(len(y_test))
        elif args['predict']:
            parser = dp.DataParser(args['-d'], dp.Mode.PREDICT)
            data = parser.parseData()
            peak_num = 0
            pred = []
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            df = pd.DataFrame(
                columns=['Name', 'Score', '#Top Hit Ions in Compound', '#Top Compound Ions in Hit',
                         'Molecular Ion in Compound', 'Correlation Percentage'])
            table_creator = ft.gt_table()
            for compound in sample:
                peak_num = peak_num + 1
                name = compound['name']
                if name in hits:
                    s, x, y, z, m = table_creator.table(hits[name], compound)
                    df.loc[-1] = [str(name), s, x, y, z, m]
                    df.index = df.index + 1
                    df = df.sort_index()
                    df = df.iloc[::-1]
                    df = df.drop(['Name'], axis=1)
                    prediction = dt.model_predict(df, args['-s'])
                    pred.append((peak_num, name, prediction))
                df = pd.DataFrame(columns=['Name', 'Score', '#Top Hit Ions in Compound', '#Top Compound Ions in Hit','Molecular Ion in Compound', 'Correlation Percentage'])
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv1(sample_name, pred)
            rp.report_pdf1(sample_name, hits, sample, pred)

    elif args['algo']:
        import AlgoModel as am
        model = am.AlgoModel(int(args['--st']), int(args['--pt']))
        if args['test']:
            parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
            all_data = parser.parseData()
            final_tn, final_fp, final_fn, final_tp = 0, 0, 0, 0
            all_pred = []
            all_valid_gt = []
            all_names = []
            all_hits = []
            all_samples = []
            for data in all_data:
                pred = []
                gt_conf = []
                sample_name = data['name']
                hits = data['hits']
                sample = data['sample']
                ground_truth = data['ground_truth']
                valid_gt = get_valid_gt(ground_truth, sample)
                all_names.append(sample_name)
                all_valid_gt.append(valid_gt)
                all_hits.append(hits)
                all_samples.append(sample)
                for (peak_num, name, confidence) in valid_gt:
                    conf, percentage_lib_hit_present, molecular_ion_hit, top_three_ion_list_sample = model.predict(
                        hits[name], sample[peak_num - 1])
                    pred.append(conf)
                    all_pred.append(conf)
                    gt_conf.append(confidence)
                pred = np.array(pred)
                gt_conf = np.array(gt_conf)
                matrix = get_matrix(pred, gt_conf)
                final_tn, final_fp, final_fn, final_tp = final_tn + matrix[0, 0], final_fp + matrix[
                    0, 1], final_fn + matrix[1, 0], final_tp + matrix[1, 1]
                # rp = rg.ReportGenerator(ROOT_PATH + "/{}-report".format(sample_name), 'algo')
                # rp.report_csv2(sample_name, valid_gt, pred)
                # rp.report_matrix(sample_name, matrix)
                # rp.report_pdf2(sample_name, hits, sample, valid_gt, pred)
            final_accuracy = (final_tp + final_tn) / (final_tp + final_tn + final_fp + final_fn) * 100
            print("acc: {:0.2f} %".format(final_accuracy))
            rp = rg.ReportGenerator(ROOT_PATH + '/report', 'algo')
            final_matrix = np.matrix([[final_tn, final_fp], [final_fn, final_tp]])
            rp.final_report_matrix(final_matrix)
            rp.report_csv(all_valid_gt, all_pred)
            rp.report_pdf(all_hits, all_samples, all_valid_gt, all_pred, all_names)
        elif args['predict']:
            parser = dp.DataParser(args['-d'], dp.Mode.PREDICT)
            data = parser.parseData()
            peak_num = 0
            pred = []
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            for compound in sample:
                peak_num = peak_num + 1
                name = compound['name']
                if name in hits:
                    conf, percentage_lib_hit_present, molecular_ion_hit, top_three_ion_list_sample = model.predict(hits[name], compound)
                    pred.append((peak_num, name, conf))
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'algo')
            rp.report_csv1(sample_name, pred)
            rp.report_pdf1(sample_name, hits, sample, pred)

    elif args['scatter']:
        parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
        all_data = parser.parseData()
        all_peak_nums = []
        for data in all_data:
            peak_num_list = []
            ground_truth = data['ground_truth']
            sample = data['sample']
            hits = data['hits']
            valid_gt = get_valid_gt(ground_truth, sample)
            for (peak_num, name, confidence) in valid_gt:
                peak_num_list.append(peak_num)
            all_peak_nums.append(peak_num_list)
        scatter = wr.scatter(ROOT_PATH + '/CINeMA.py', args['-w'], all_peak_nums)
