"""
Usage:
  multivariate_spectra.py ml (test|predict) -d <data_dir> -s <model_save_file_path>
  multivariate_spectra.py multi -d <data_dir> -s <model_save_file_path> [--e <epochs>]
  multivariate_spectra.py -h

Options:
  -h --help                     Show this screen.
  -d <data_dir>                 Data directory path.
  --st <similarity_threshold>   Similarity threshold [default: 600].
  --pt <percent threshold>      Percent threshold [default: 80].
  --e <epochs>                  Number of epochs [default: 30]
  -s <model_save_file_path>     File path where model is saved
  --version                     Show version
"""

value = 7
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
np.random.seed(value)
import random
random.seed(value)
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

def get_samples(pair, sum_pair):
    simulated_hits = []
    simulated_samples = []

    hit_spectrum = pair[0]
    sample_spectrum = pair[1]

    hit_spectrum_probabilities = [i / sum(hit_spectrum) for i in hit_spectrum]
    sample_spectrum_probabilities = [i / sum(sample_spectrum) for i in sample_spectrum]

    rv_hit = np.random.multinomial(10000, hit_spectrum_probabilities, size = 500)
    rv_sample = np.random.multinomial(10000, sample_spectrum_probabilities, size = 500)

    for i in rv_hit:
        s1_norm = np.divide(i, sum(i))

        sum_of_intensities = sum_pair[0]

        simulated_hit = s1_norm * sum_of_intensities
        simulated_hits.append(simulated_hit)

    for i in rv_sample:
        s1_norm = np.divide(i, sum(i))

        sum_of_intensities = sum_pair[1]

        simulated_sample = s1_norm * sum_of_intensities
        simulated_samples.append(simulated_sample)

    return simulated_hits, simulated_samples


if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    import MLModel as ml
    #print(args)

    if args['multi']:
        parser = dp.DataParser(args['-d'], dp.Mode.TEST)
        data = parser.parseData()
        sample_name = data['name']
        hits = data['hits']
        sample = data['sample']
        ground_truth = data['ground_truth']
        valid_gt = get_valid_gt(ground_truth, sample)

        all_high_pairs = []
        all_high_sum_pairs = []

        all_low_pairs = []
        all_low_sum_pairs = []

        test_high = 0
        test_low = 0

        for (peak_num, name, confidence) in valid_gt:
            if confidence == 2:
                high_pair = []
                high_sum_pair = []
                sample_spectrum = sample[peak_num - 1]['spectrum']
                hit_spectrum = hits[name]['spectrum']
                hit_sum_of_intensities = sum(hit_spectrum)
                sample_sum_of_intensities = sum(sample_spectrum)
                high_pair.append(hit_spectrum)
                high_pair.append(sample_spectrum)
                high_sum_pair.append(hit_sum_of_intensities)
                high_sum_pair.append(sample_sum_of_intensities)
                all_high_pairs.append(high_pair)
                all_high_sum_pairs.append(high_sum_pair)
                if test_high == 165:
                    print(peak_num)
                test_high = test_high + 1

            if confidence == 0:
                low_pair = []
                low_sum_pair = []
                sample_spectrum = sample[peak_num - 1]['spectrum']
                hit_spectrum = hits[name]['spectrum']
                hit_sum_of_intensities = sum(hit_spectrum)
                sample_sum_of_intensities = sum(sample_spectrum)
                low_pair.append(hit_spectrum)
                low_pair.append(sample_spectrum)
                low_sum_pair.append(hit_sum_of_intensities)
                low_sum_pair.append(sample_sum_of_intensities)
                all_low_pairs.append(low_pair)
                all_low_sum_pairs.append(low_sum_pair)
                if test_low == 154:
                    print(peak_num)
                test_low = test_low + 1


        random_high_pair = random.randint(0, len(all_high_pairs))
        random_low_pair = random.randint(0, len(all_low_pairs))

        new_high_hits, new_high_samples = get_samples(all_high_pairs[random_high_pair], all_high_sum_pairs[random_high_pair])
        new_low_hits, new_low_samples = get_samples(all_low_pairs[random_low_pair], all_low_sum_pairs[random_low_pair])

        print(random_high_pair)
        print(random_low_pair)

        rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
        rp.new_spectra(new_high_hits, new_high_samples, 'new_highs')
        rp.new_spectra(new_low_hits, new_low_samples, 'new_lows')

        x = []
        y = []

        for hh, sh in zip(new_high_hits, new_high_samples):
            sample_spectrum = sh / np.linalg.norm(sh)
            hit_spectrum = hh / np.linalg.norm(hh)
            hh[0:49] = 0
            x.append(np.concatenate((sh, hh)))
            y.append([0, 1])

        for hl, sl in zip(new_low_hits, new_low_samples):
            sample_spectrum = sl / np.linalg.norm(sl)
            hit_spectrum = hl / np.linalg.norm(hl)
            hl[0:49] = 0
            x.append(np.concatenate((sl, hl)))
            y.append(([1, 0]))

        dnn = ml.DNNModel(args['-s'])
        dnn.cross(np.array(x), np.array(y), int(args['--e']))
        dnn.train(np.array(x), np.array(y), int(args['--e']))
        dnn.save()

    if args['ml']:
        if args['test']:
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
            #rp.report_csv(all_valid_gt, pred)
            #rp.report_pdf(all_hits, all_samples, all_valid_gt, pred, all_names)
            rp.original_spectra(all_hits, all_samples, all_valid_gt, pred, all_names)

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
