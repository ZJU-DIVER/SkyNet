import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
from skyNet import SkyNet
from datetime import datetime
from tqdm import trange
from parallel_compute import build_multi_gpu_model
from utils import get_parser, evaluate, enclosed, CHarea
import json
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def preprocess(x):
    src_trgt = tf.string_split([x], delimiter="output").values  # seperate the src data and output set
    src, trgt = src_trgt[0], src_trgt[1]
    src = tf.string_to_number(tf.string_split([src]).values, tf.float32)
    trgt = tf.string_to_number(tf.string_split([trgt]).values, tf.int32)  # array
    src = tf.reshape(src, shape=(-1, params.input_dim))  # Adjust to input dimensions, by default its 2
    src_real = src
    trgt_real = trgt
    src = tf.concat([tf.reshape(tf.convert_to_tensor(params.end_of_sequence), shape=(1, params.input_dim)), src],
                    axis=0)
    trgt = tf.concat([trgt, tf.fill([1], 0)], axis=0)
    return src, tf.shape(src)[0], trgt, src_real, trgt_real, tf.shape(src_real)[0]


def get_iterator(filename, params, is_infer=False):
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.map(preprocess, num_parallel_calls=8)  # preprocess the dataset and put it in the iterator
    if not is_infer:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000))
    dataset = dataset.padded_batch(params.batch_size, padded_shapes=(
        [None, params.input_dim], [], [None], [None, params.input_dim], [None], []))
    return dataset.make_one_shot_iterator()


def logits_to_index(logits):
    m = np.shape(logits)[0]
    indexs = []
    for i in range(m):
        temp_logits = logits[i]
        temp_idx = list(np.where(temp_logits > temp_logits[0]))
        indexs += temp_idx
    return indexs


if __name__ == '__main__':
    parser = get_parser()
    params, unparsed = parser.parse_known_args()
    if params.mode == 'train':
        train_graph = tf.Graph()
        eval_graph = tf.Graph()

        train_file = "data/skyline10k-1k-2d.txt"
        infer_file = "data/skyline10k-1k-2d.txt"

        with train_graph.as_default(), tf.container("train"):
            train_iter = get_iterator(train_file, params)
            train_op, loss = build_multi_gpu_model(params, SkyNet,
                                                   train_iter)  # return a train operation to minimize the loss
            train_saver = tf.train.Saver(tf.global_variables(), max_to_keep=params.max_to_keep)
            init = tf.global_variables_initializer()

        with eval_graph.as_default(), tf.container("eval"):
            eval_iter = get_iterator(infer_file, params)
            eval_model = SkyNet(params, 'eval', eval_iter.get_next())
            eval_loss = eval_model.compute_loss()
            eval_saver = tf.train.Saver()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        train_sess = tf.Session(config=config, graph=train_graph)
        eval_sess = tf.Session(config=config, graph=eval_graph)

        if params.use_previous_ckpt:
            latest_ckpt = tf.train.latest_checkpoint(params.model_dir)
            train_saver.restore(train_sess, latest_ckpt)
        else:
            train_sess.run(init)
            params.model_dir = 'model'
            params.model_dir = os.path.join(params.model_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S/')).replace(
                '\\', '/')
            os.mkdir(params.model_dir)
            tf.logging.info('model_dir: {}'.format(params.model_dir))
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
                train_saver.save(sess=train_sess, save_path=params.model_dir,
                                 global_step=tf.train.get_global_step(graph=train_graph))

                eval_saver.restore(sess=eval_sess, save_path=tf.train.latest_checkpoint(params.model_dir))
                test = eval_sess.run({"loss": eval_loss, "true": eval_model.trgt_real, "trgt": eval_model.trgt,
                                      "points": eval_model.src_real, "src_length": eval_model.src_length,
                                      "max_time": eval_model.max_time, "logits": eval_model.logits,
                                      "src": eval_model.src})
                # "special_points_logits": eval_model.special_points_logits,'predicted': eval_model.predicted
                #  "condition": eval_model.condition}),"for_fun":eval_model.for_fun
                pred = logits_to_index(test['logits'])
                print("xxxxxxxx", np.shape(pred), np.shape(test['logits']))
                test.update({'predicted': pred})

                for j in range(len(test['true'])):  # len(test['true']) = 128
                    # print('***************************************************','logits is:',test['logits'].shape,type(test['logits']),test['logits'])
                    # print('***************************************************','trgt is:',test['trgt'].shape,type(test['trgt']),test['trgt'])
                    # print('***************************************************','multilabel is:',test['mul_label'].shape,type(test['mul_label']),test['mul_label'])
                    # np.savetxt("multilabel.csv", test['mul_label'],fmt='%5s', delimiter=",")

                    # print('self.special_points_logits is:',test['special_points_logits'].shape,test['special_points_logits'])
                    # print('self.condition is',test['condition'].shape,type(test['condition']), test['condition'])
                    # print('**************************',len(test['predicted']),type(test['predicted']),test['predicted'])

                    # tf.logging.info("true:      {}".format(test['true'][j]))
                    # tf.logging.info("predicted: {}".format(test['predicted'][j]))

                    true_set = set(test['true'][j]) - {0}
                    pred_set = set(test['predicted'][j]) - {0}
                    # print("#####$$$$$",test["src_length"].shape,test["src_length"])
                    # print("##########",test["max_time"].shape,test["max_time"])
                    # print("@@@@@@@@@@",test["trgt_weight"].shape,test["trgt_weight"])        
                    print(" true", true_set)
                    print(" pred", pred_set)
                    print("================================")

                    numCorrect, recall, precision, f1 = evaluate(true_set, pred_set, test['points'][j])
                    true_enclose = enclosed(true_set, test['points'][j])
                    pred_enclose = enclosed(pred_set, test['points'][j])
                    enclose_ratio = pred_enclose / (true_enclose + 1e-10)
                    tf.logging.info(
                        "correct prediction: %d, recall: %.5f, precision: %.5f, f1: %.5f, enclose ratio: %.5f" % (
                            numCorrect, recall, precision, f1, enclose_ratio))

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
                    tf.train.get_global_step(graph=train_graph).eval(session=train_sess), avg_recall, avg_precision,
                    avg_f1,
                    avg_enclose)
                s2 = "max recall = %.2f, max precision = %.2f, max f1 = %.2f, max enclose ratio = %.2f" % (
                    max_recall, max_precision, max_f1, max_enclose)
                s3 = "min recall = %.2f, min precision = %.2f, min f1 = %.2f, min enclose ratio = %.2f" % (
                    min_recall, min_precision, min_f1, min_enclose)
                tf.logging.info(s1 + '\n' + s2 + '\n' + s3)
                tf.logging.info("loss: {}".format(test['loss']))
                logfile.write(s1 + '\n' + s2 + '\n' + s3 + '\n')
                logfile.write("loss: {}\n".format(test['loss']))
        logfile.close()


    elif params.mode == 'infer':
        infer_graph = tf.Graph()

        with infer_graph.as_default(), tf.container('infer'):
            infer_iter = get_iterator("data/skylineAll10k-1k-2d.txt", params, is_infer=True)
            true = []
            logits = []
            points = []
            prob = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(params.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_%d' % (i)) as scope:
                            model = SkyNet(params, 'infer', infer_iter.get_next())
                            tf.get_variable_scope().reuse_variables()

                            logits.append(model.logits)

                            # pred = logits_to_index(model.logits)
                            # predicted.append(pred)

                            true.append(model.trgt)
                            points.append(model.src)
                            # print('$$$$$$$$$$$$$$',model.p)
                            prob.append(model.p)
                            # model.load_model(infer_sess)
            infer_saver = tf.train.Saver()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        infer_sess = tf.Session(config=config, graph=infer_graph)
        print("save_path:", params.model_dir, tf.train.latest_checkpoint(params.model_dir))
        infer_saver.restore(sess=infer_sess, save_path=tf.train.latest_checkpoint(params.model_dir))

        flag = True
        total_avg_recall = 0
        total_avg_precision = 0
        total_avg_f1 = 0
        total_avg_enclose = 0
        total_count = 0
        total_time = 0
        while True:
            try:
                start_time = datetime.now()
                result = infer_sess.run({"true": true, "logits": logits, "points": points, 'p': prob})
                pred = logits_to_index(result['logits'][0])
                pred = [arr.tolist() for arr in pred]

                result.update({'predicted': pred})
                # print('Shape of pred is:',np.shape(result['predicted']), pred)
                # print('======')
                # print('Shape of the true is',np.shape(result['true']),result['true'])

                time_elapsed = datetime.now() - start_time

                if flag:
                    flag = False
                    total_time = time_elapsed
                else:
                    total_time += time_elapsed
                count = 0
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

                for j in range(len(result['true'])):
                    for i in range(len(result['true'][j])):
                        tf.logging.info("true:      {}".format(result['true'][j][i]))
                        tf.logging.info("predicted: {}".format(result['predicted'][i]))
                        true_set = set(result['true'][j][i]) - {0}

                        # print(np.shape(result['true'][j]))
                        # print(np.shape(result['predicted'][j]), type(result['predicted'][j]),result['predicted'][j])
                        # result['predicted'][j] = list(result['predicted'][j])
                        # print(type(result['predicted'][j]), result['predicted'][j],result['predicted'][j][0])
                        pred_set = set(result['predicted'][i]) - {0}
                        numCorrect, recall, precision, f1 = evaluate(true_set, pred_set, result['points'][j][i])
                        true_enclose = enclosed(true_set, result['points'][j][i])
                        pred_enclose = enclosed(pred_set, result['points'][j][i])
                        enclose_ratio = pred_enclose / (true_enclose + 1e-10)
                        tf.logging.info(
                            "correct prediction: %d, recall: %.5f, precision: %.5f, f1: %.5f, enclose ratio: %.5f" % (
                                numCorrect, recall, precision, f1, enclose_ratio))

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

                total_count += count
                total_avg_recall += sum_recall
                total_avg_precision += sum_precision
                total_avg_f1 += sum_f1
                total_avg_enclose += sum_enclose
                avg_recall = sum_recall / count
                avg_precision = sum_precision / count
                avg_f1 = sum_f1 / count
                avg_enclose = sum_enclose / count
                s1 = "\navg recall = %.5f, avg precision = %.5f, avg f1 = %.5f, avg enclose ratio = %.5f" % (
                    avg_recall, avg_precision, avg_f1, avg_enclose)
                s2 = "max recall = %.5f, max precision = %.5f, max f1 = %.5f, max enclose ratio = %.5f" % (
                    max_recall, max_precision, max_f1, max_enclose)
                s3 = "min recall = %.5f, min precision = %.5f, min f1 = %.5f, min enclose ratio = %.5f" % (
                    min_recall, min_precision, min_f1, min_enclose)

                tf.logging.info(s1 + '\n' + s2 + '\n' + s3)
                tf.logging.info("time elapsed: {}".format(time_elapsed))
            except tf.errors.OutOfRangeError:
                print("total time elapsed: {}".format(total_time))
                print(
                    "total average recall=%.5f, total average precision=%.5f, total average f1=%.5f, total avg enclose=%.5f " % (
                        total_avg_recall / total_count, total_avg_precision / total_count, total_avg_f1 / total_count,
                        total_avg_enclose / total_count))
                exit()
        print("total time elapsed: {}".format(total_time))
        print(
            "total average recall=%.5f, total average precision=%.5f, total average f1=%.5f, total avg enclose=%.5f " % (
                total_avg_recall / total_count, total_avg_precision / total_count, total_avg_f1 / total_count,
                total_avg_enclose / total_count))
        exit()
