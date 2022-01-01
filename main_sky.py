import copy
import sys
import time

import tensorflow as tf
from sklearn import metrics

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
from skyNet_ori import SkyNet
from datetime import datetime
from tqdm import trange
from parallel_compute import build_multi_gpu_model
from utils import get_parser, evaluate, enclosed, CHarea
import json
import numpy as np
import os

tf.compat.v1.disable_eager_execution()


def preprocess(x):
    src_trgt = tf.compat.v1.string_split([x], delimiter="output").values  # seperate the src data and output set
    src, trgt = src_trgt[0], src_trgt[1]
    src = tf.compat.v1.string_to_number(tf.compat.v1.string_split([src]).values, tf.float32)
    trgt = tf.compat.v1.string_to_number(tf.compat.v1.string_split([trgt]).values, tf.int32)
    src = tf.reshape(src, shape=(-1, params.input_dim))  # Adjust to input dimensions, by default its 2
    return src, tf.shape(src)[0], trgt, tf.shape(trgt)[0]


def get_iterator(filename, params, is_infer=False):
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.map(preprocess, num_parallel_calls=8)  # preprocess the dataset and put it in the iterator
    if not is_infer:
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(1000))
    dataset = dataset.padded_batch(params.batch_size, padded_shapes=(
        [None, params.input_dim], [], [None], []))
    return tf.compat.v1.data.make_one_shot_iterator(dataset)


def logits_to_index(logits, balance=False, dim=2):
    m = np.shape(logits)[0]

    logits = tf.nn.sigmoid(logits)  # S activation function
    logits = logits.eval(session=tf.compat.v1.Session())

    indexs = []
    for i in range(m):
        temp_logits = logits[i]
        temp_idx = np.where(temp_logits > 0.4)[0].tolist()
        temp_idx = [i + 1 for i in temp_idx]
        indexs.append(temp_idx)
    # print("++++++++++++ index looks like this +++++++++++", indexs)
    return logits, indexs


if __name__ == '__main__':
    parser = get_parser()
    params, unparsed = parser.parse_known_args()

    if len(sys.argv) > 2:
        params.mode = sys.argv[2]
        params.num_epoch = int(sys.argv[3])
    if len(sys.argv) > 4:
        params.use_previous_ckpt = bool(sys.argv[4])
        params.model_dir = sys.argv[5]

    if params.mode == 'train':
        train_graph = tf.compat.v1.Graph()
        eval_graph = tf.compat.v1.Graph()

        if len(sys.argv) >= 2:
            train_file = infer_file = sys.argv[1]
        else:
            train_file = "data/skyline10k-1k-2d-inde.txt"
            infer_file = "data/skyline10k-1k-2d-inde.txt"

        tf.compat.v1.logging.info('train_file: {}'.format(train_file))
        tf.compat.v1.logging.info('infer_file: {}'.format(infer_file))

        with train_graph.as_default(), tf.compat.v1.container("train"):
            train_iter = get_iterator(train_file, params)
            train_model, train_op, loss = build_multi_gpu_model(params, SkyNet, train_iter)
            # train_op, loss = build_no_gpu_model(params, SkyNet, train_iter)
            train_saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=params.max_to_keep)
            init = tf.compat.v1.global_variables_initializer()

        with eval_graph.as_default(), tf.compat.v1.container("eval"):
            eval_iter = get_iterator(infer_file, params)
            eval_params = copy.deepcopy(params)
            eval_params.mode = "eval"
            eval_model, op, eval_loss = build_multi_gpu_model(params, SkyNet, eval_iter)
            eval_saver = tf.compat.v1.train.Saver()

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        train_sess = tf.compat.v1.Session(config=config, graph=train_graph)
        eval_sess = tf.compat.v1.Session(config=config, graph=eval_graph)

        if params.use_previous_ckpt:
            latest_ckpt = tf.train.latest_checkpoint(params.model_dir)
            train_saver.restore(train_sess, latest_ckpt)
        else:
            train_sess.run(init)
            params.model_dir = 'model'
            params.model_dir = os.path.join(params.model_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S/')).replace(
                '\\', '/')
            os.mkdir(params.model_dir)
            tf.compat.v1.logging.info('model_dir: {}'.format(params.model_dir))
            param_path = os.path.join(params.model_dir, "params.json")
            with open(param_path, 'x') as f:
                json.dump(params.__dict__, f, indent=4, sort_keys=True)

        logpath = os.path.join(params.model_dir, "log")
        logfile = open(logpath, 'a')
        logfile.write('SkyNet\n')
        logfile.write(train_file + '\n')
        logfile.write(infer_file + '\n')

        for i in trange(params.num_epoch):
            result = train_sess.run({"update": train_op, "loss": loss})
            if (i + 1) % params.log_step == 0:
                count = 0
                max_recall = 0
                min_recall = float('inf')
                sum_recall = 0
                max_precision = 0
                min_precision = float('inf')
                sum_precision = 0
                max_f1 = 0
                min_f1 = float('inf')
                sum_f1 = 0
                max_enclose = 0
                min_enclose = float('inf')
                sum_enclose = 0
                print(params.model_dir)

                train_saver.save(sess=train_sess, save_path=params.model_dir, global_step=i + 1)
                eval_saver.restore(sess=eval_sess, save_path=tf.train.latest_checkpoint(params.model_dir))
                test = eval_sess.run({"loss": eval_loss, "true": eval_model.trgt, "src_length": eval_model.src_length,
                                      "max_time": eval_model.max_time, "logits": eval_model.logits,
                                      "points": eval_model.src, "src": eval_model.src})
                logits, pred = logits_to_index(test['logits'])
                test.update({'predicted': pred})

                for j in range(len(test['true'])):
                    true_set = set(test['true'][j]) - {0}
                    pred_set = set(test['predicted'][j]) - {0}

                    numCorrect, recall, precision, f1, layer = evaluate(true_set, pred_set, test['points'][j])
                    true_enclose = enclosed(true_set, test['points'][j])
                    pred_enclose = enclosed(pred_set, test['points'][j])
                    enclose_ratio = pred_enclose / (true_enclose + 1e-10)

                    count += 1
                    sum_recall += recall
                    max_recall = max(max_recall, recall)
                    min_recall = min(min_recall, recall)
                    sum_precision += precision
                    max_precision = max(max_precision, precision)
                    min_precision = min(min_precision, precision)
                    sum_f1 += f1
                    max_f1 = max(max_f1, f1)
                    min_f1 = min(min_f1, f1)
                    sum_enclose += enclose_ratio
                    max_enclose = max(max_enclose, enclose_ratio)
                    min_enclose = min(min_enclose, enclose_ratio)

                avg_recall = sum_recall / count
                avg_precision = sum_precision / count
                avg_f1 = sum_f1 / count
                avg_enclose = sum_enclose / count
                s1 = "At step %d:\navg recall = %.5f, avg precision = %.5f, avg f1 = %.5f, avg enclose ratio = %.5f" % (
                    i, avg_recall, avg_precision,
                    avg_f1,
                    avg_enclose)
                s2 = "max recall = %.2f, max precision = %.2f, max f1 = %.2f, max enclose ratio = %.2f" % (
                    max_recall, max_precision, max_f1, max_enclose)
                s3 = "min recall = %.2f, min precision = %.2f, min f1 = %.2f, min enclose ratio = %.2f" % (
                    min_recall, min_precision, min_f1, min_enclose)
                tf.compat.v1.logging.info(s1 + '\n' + s2 + '\n' + s3)
                tf.compat.v1.logging.info("loss: {}".format(test['loss']))
                logfile.write(s1 + '\n' + s2 + '\n' + s3 + '\n')
                logfile.write("loss: {}\n".format(test['loss']))
                print("count:", count)

        logfile.close()


    elif params.mode == 'infer':
        infer_graph = tf.compat.v1.Graph()
        if len(sys.argv) >= 2:
            infer_file = sys.argv[1]
        else:
            infer_file = "data/skyline10k-1k-2d-inde.txt"

        with infer_graph.as_default(), tf.compat.v1.container('infer'):
            infer_iter = get_iterator(infer_file, params, is_infer=True)
            infer_model = SkyNet(params, params.mode, infer_iter.get_next())
            infer_saver = tf.compat.v1.train.Saver()

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        infer_sess = tf.compat.v1.Session(config=config, graph=infer_graph)
        print("save_path:", params.model_dir, tf.train.latest_checkpoint(params.model_dir))
        infer_saver.restore(sess=infer_sess, save_path=tf.train.latest_checkpoint(params.model_dir))

        total_true_logits = []
        total_false_logits = []

        flag = True
        total_avg_recall = 0
        total_avg_precision = 0
        total_avg_f1 = 0
        total_avg_enclose = 0
        total_count = 0
        total_time = None
        total_layer_count = 0
        total_avg_layer = np.array([float(0) for _ in range(11)])
        total_avg_auc = 0
        while True:
            try:
                count = 0
                layer_count = 0
                sum_recall = 0
                max_recall = 0
                min_recall = float('inf')
                max_precision = 0
                min_precision = float('inf')
                sum_precision = 0
                max_f1 = 0
                min_f1 = float('inf')
                sum_f1 = 0
                max_enclose = 0
                min_enclose = float('inf')
                sum_enclose = 0
                max_layer = [0 for _ in range(11)]
                min_layer = [float('inf') for _ in range(11)]
                sum_layer = np.array([float(0) for _ in range(11)])
                auc = 0

                start_time = time.time()
                result = infer_sess.run(
                    {"true": infer_model.trgt, "src_length": infer_model.src_length,
                     "logits": infer_model.logits, "points": infer_model.src, "src": infer_model.src})
                logits, pred = logits_to_index(result['logits'])
                result.update({'predicted': pred})
                time_elapsed = time.time() - start_time

                for i in range(len(result['true'])):
                    tf.compat.v1.logging.info("true:      {}".format(result['true'][i]))
                    tf.compat.v1.logging.info("predicted: {}".format(result['predicted'][i]))

                    true_set = set(result['true'][i]) - {0}
                    pred_set = set(result['predicted'][i])

                    true_logits = []
                    false_logits = []

                    for j in range(len(logits[i])):
                        if j + 1 in true_set:
                            true_logits.append(logits[i][j])
                        else:
                            false_logits.append(logits[i][j])
                    total_true_logits.append(true_logits)
                    total_false_logits.append(false_logits)

                    testy = [1 for _ in range(len(true_logits))] + [0 for _ in range(len(false_logits))]
                    yhat = true_logits + false_logits
                    auc += metrics.roc_auc_score(testy, yhat)

                    numCorrect, recall, precision, f1, layer = evaluate(true_set, pred_set, result['points'][i])
                    true_enclose = enclosed(true_set, result['points'][i])
                    pred_enclose = enclosed(pred_set, result['points'][i])
                    enclose_ratio = pred_enclose / (true_enclose + 1e-10)

                    count += 1
                    sum_recall += recall
                    max_recall = max(max_recall, recall)
                    min_recall = min(min_recall, recall)
                    sum_precision += precision
                    max_precision = max(max_precision, precision)
                    min_precision = min(min_precision, precision)
                    sum_f1 += f1
                    max_f1 = max(max_f1, f1)
                    min_f1 = min(min_f1, f1)
                    sum_enclose += enclose_ratio
                    max_enclose = max(max_enclose, enclose_ratio)
                    min_enclose = min(min_enclose, enclose_ratio)
                    if len(layer) != 0:
                        sum_layer += layer
                        max_layer = [max(max_layer[i], layer[i]) for i in range(11)]
                        min_layer = [min(min_layer[i], layer[i]) for i in range(11)]
                        layer_count += 1

                if flag:
                    flag = False
                    total_time = time_elapsed
                else:
                    total_time += time_elapsed

                total_avg_auc += auc
                total_count += count
                total_layer_count += layer_count
                total_avg_recall += sum_recall
                total_avg_precision += sum_precision
                total_avg_f1 += sum_f1
                total_avg_enclose += sum_enclose
                total_avg_layer += sum_layer
                avg_recall = sum_recall / count
                avg_precision = sum_precision / count
                avg_f1 = sum_f1 / count
                avg_enclose = sum_enclose / count
                avg_layer = sum_layer / layer_count
                avg_auc = auc / count
                s1 = "\navg recall = %.5f, avg precision = %.5f, avg f1 = %.5f, avg enclose ratio = %.5f" % (
                    avg_recall, avg_precision, avg_f1, avg_enclose)
                s2 = "max recall = %.5f, max precision = %.5f, max f1 = %.5f, max enclose ratio = %.5f" % (
                    max_recall, max_precision, max_f1, max_enclose)
                s3 = "min recall = %.5f, min precision = %.5f, min f1 = %.5f, min enclose ratio = %.5f" % (
                    min_recall, min_precision, min_f1, min_enclose)
                s4 = "avg layer = " + str(avg_layer) + ", max layer = " + str(max_layer) + ", min layer = " + str(
                    min_layer)
                s5 = "avg auc = " + str(avg_auc)

                tf.compat.v1.logging.info(s1 + '\n' + s2 + '\n' + s3 + '\n' + s4 + '\n' + s5)
                tf.compat.v1.logging.info("time elapsed: {}".format(time_elapsed / count))

            except tf.errors.OutOfRangeError:
                print(total_count)
                print("total time elapsed: {}".format(total_time / total_count))
                print(
                    "total average recall=%.5f, total average precision=%.5f, total average f1=%.5f, total avg enclose=%.5f " % (
                        total_avg_recall / total_count, total_avg_precision / total_count, total_avg_f1 / total_count,
                        total_avg_enclose / total_count))
                print("total avg layer=" + str(total_avg_layer / total_layer_count))
                print("total auc=" + str(total_avg_auc / total_count))
                print()

                logpath = os.path.join(params.model_dir, "total_true_logits.npy")
                np.save(logpath, total_true_logits)
                logpath = os.path.join(params.model_dir, "total_false_logits.npy")
                np.save(logpath, total_false_logits)

                exit()

        print("total time elapsed: {}".format(total_time / total_count))
        print(
            "total average recall=%.5f, total average precision=%.5f, total average f1=%.5f, total avg enclose=%.5f " % (
                total_avg_recall / total_count, total_avg_precision / total_count, total_avg_f1 / total_count,
                total_avg_enclose / total_count))

        exit()
