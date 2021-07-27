import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
from ptrNetPar import PointerNet
import os
from datetime import datetime
from tqdm import trange
from utils import get_parser, enclosed, evaluate, CHarea
from parallel_compute import build_multi_gpu_model
import json


def preprocess(x):
    src_trgt = tf.string_split([x], delimiter="output").values
    src, trgt = src_trgt[0], src_trgt[1]
    src = tf.string_to_number(tf.string_split([src]).values, tf.float32)
    trgt = tf.string_to_number(tf.string_split([trgt]).values, tf.int32)
    src = tf.reshape(src, shape=(-1, params.input_dim))

    src = tf.concat([tf.fill([1, params.input_dim], 2.0), src], axis=0)  # src = special point + src
    trgt_in = tf.gather_nd(src, tf.reshape(trgt, shape=[-1, 1]))  # trgt_in  = [the value of skyline points]
    trgt_out = tf.concat([trgt, tf.fill([1], 0)], axis=0)  # trgt_out = [skyline points indeces] + special point index
    trgt_in = tf.concat([tf.fill([1, params.input_dim], 0.0), trgt_in],
                        axis=0)  # trgt_in =  [0,0] + [the value of skyline points]

    return src, tf.shape(src)[0], trgt_in, trgt_out, tf.shape(trgt_in)[0]


def get_iterator(filename, params, is_infer=False):
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.map(preprocess, num_parallel_calls=8)
    if not is_infer:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000))
    dataset = dataset.padded_batch(params.batch_size, padded_shapes=(
        [None, params.input_dim], [], [None, params.input_dim], [None], []))  # Have doubt of padded_shapes
    return dataset.make_one_shot_iterator()


if __name__ == '__main__':
    parser = get_parser()
    params, unparsed = parser.parse_known_args()

    if params.mode == 'train':

        train_graph = tf.Graph()
        eval_graph = tf.Graph()

        train_file = "data/skyline2k-1k-2d.txt"
        infer_file = "data/skyline2k-1k-2d.txt"

        with train_graph.as_default(), tf.container("train"), tf.device('/cpu:0'):
            train_iter = get_iterator(train_file, params)
            train_op, loss = build_multi_gpu_model(params, PointerNet, train_iter)
            train_saver = tf.train.Saver(tf.global_variables(), max_to_keep=params.max_to_keep)
            init = tf.global_variables_initializer()

        with eval_graph.as_default(), tf.container("eval"), tf.device('/cpu:0'):
            eval_iter = get_iterator(infer_file, params)
            eval_model = PointerNet(params, 'eval', eval_iter.get_next())
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
        logfile.write('Ptr-Net\n')
        logfile.write(train_file + '\n')
        logfile.write(infer_file + '\n')

        for i in trange(params.num_epoch):
            result = train_sess.run({"update": train_op, "loss": loss})
            if (i + 1) % 100 == 0:
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

                train_saver.save(sess=train_sess, save_path=params.model_dir,
                                 global_step=tf.train.get_global_step(graph=train_graph))

                eval_saver.restore(sess=eval_sess, save_path=tf.train.latest_checkpoint(params.model_dir))
                test = eval_sess.run({"loss": eval_loss, "true": eval_model.trgt, 'predicted': eval_model.predicted,
                                      "points": eval_model.src, "logits": eval_model.logits,
                                      "crossent": eval_model.crossent, "max_time": eval_model.max_time})

                for j in range(len(test['true'])):
                    tf.logging.info("true:      {}".format(set(test['true'][j])))
                    tf.logging.info("predicted: {}".format(set(test['predicted'][j])))
                    true_set = set(test['true'][j]) - {0}  # index of trgt
                    # print("===========",type(test['logits']),test['logits'].shape,test['logits'])
                    # print("===========",type(test['predicted']),test['predicted'].shape,test['predicted'])
                    # print("===========", type(test['true']), test['true'].shape, test['true'])  # 128,7
                    # print("===========",type(test['max_time']),test['max_time'].shape,test['max_time']) #7
                    # print("===========",type(test['crossent']),test['crossent'].shape,test['crossent']) #128,7

                    pred_set = set(test['predicted'][j, :]) - {0}  # index of trgt
                    numCorrect, recall, precision, f1 = evaluate(true_set, pred_set, test['points'][j])
                    true_enclose = CHarea(true_set, test['points'][j], shift=False)
                    pred_enclose = CHarea(pred_set, test['points'][j], shift=False)
                    enclose_ratio = (pred_enclose + 0.0001) / (true_enclose + 0.0001)
                    tf.logging.info(
                        "correct prediction: %d, recall: %.2f, precision: %.2f, f1: %.2f, enclose ratio: %.2f" % (
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
                s1 = "At step %d:\navg recall = %.2f, avg precision = %.2f, avg f1 = %.2f, avg enclose ratio = %.2f" % (
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

        with infer_graph.as_default(), tf.container('infer'), tf.device('/cpu:0'):
            infer_iter = get_iterator("data/skyline3k-1k-2d.txt", params, is_infer=True)
            true = []
            predicted = []
            points = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(params.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_%d' % (i)) as scope:
                            model = PointerNet(params, 'infer', infer_iter.get_next())
                            tf.get_variable_scope().reuse_variables()
                            predicted.append(model.predicted)
                            true.append(model.trgt)
                            points.append(model.src)
                            # model.load_model(infer_sess)
            infer_saver = tf.train.Saver()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        infer_sess = tf.Session(config=config, graph=infer_graph)
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
                result = infer_sess.run({"true": true, "predicted": predicted, "points": points})
                print(result)
                time_elapsed = datetime.now() - start_time
                # tf.logging.info("time elapsed: {}".format(time_elapsed))

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
                        tf.logging.info("predicted: {}".format(result['predicted'][j][i]))
                        true_set = set(result['true'][j][i]) - {0}
                        pred_set = set(result['predicted'][j][i]) - {0}
                        numCorrect, recall, precision, f1 = evaluate(true_set, pred_set, result['points'][j][i])
                        true_enclose = enclosed(true_set, result['points'][j][i], shift=False)
                        pred_enclose = enclosed(pred_set, result['points'][j][i], shift=False)
                        enclose_ratio = pred_enclose / true_enclose
                        tf.logging.info(
                            "correct prediction: %d, recall: %.2f, precision: %.2f, f1: %.2f, enclose ratio: %.2f" % (
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
                    "total average recall=%.5f, total average precision=%.5f, total average f1=%.5f, total average enclose ratio=%.5f " % (
                        total_avg_recall / total_count, total_avg_precision / total_count, total_avg_f1 / total_count,
                        total_avg_enclose / total_count))
                exit()

        print("total time elapsed: {}".format(total_time))
        print(
            "total average recall=%.5f, total average precision=%.5f, total average f1=%.5f, total average enclose ratio=%.5f " % (
                total_avg_recall / total_count, total_avg_precision / total_count, total_avg_f1 / total_count,
                total_avg_enclose / total_count))
