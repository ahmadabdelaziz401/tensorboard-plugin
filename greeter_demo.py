import os.path
import time
import tensorflow as tf
import numpy as np
from multiprocessing import Process
from pynput import keyboard
import os
import sys
import socketserver
import json

# from greeter_plugin_dir import ThreadingTCPServerWithProcess, TCPHandler
from threading_tcp_server_with_process import ThreadingTCPServerWithProcess
from server_handler import TCPHandler
import concurrent.futures as cf
# import greeter_plugin_dir.threading_tcp_server_with_process
# path = "/vagrant_data/mnist_tf"

class MNIST():
    batch = 200
    steps = 2000
    print_rate = 100
    files = []
    path = "/home/ahmad/tensor/mnist_tf"

    def __init__(self, log_path="/tmp/initial"):
        self.log_path = log_path
        self.progress = []

        # self.process = Process(target=self.main, args=(log,))

    def _parse_function(example_proto):
        features = {'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)
                    }
        parsed_features = tf.parse_single_example(example_proto, features)

        return parsed_features["image"], parsed_features["label"]

    def read_data(path):
        for i in range(300):
            file = "{}/record{}.tfrecords".format(path, i)

            if not os.path.isfile(file):
                raise Exception("Files Does Not Exist", file)
            else:
                MNIST.files.append(file)

        dataset = tf.data.TFRecordDataset(MNIST.files)
        dataset = dataset.map(MNIST._parse_function)
        dataset = dataset.repeat()
        dataset = dataset.batch(MNIST.batch)
        iterator = dataset.make_one_shot_iterator()

        return iterator

    tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
    tf.app.flags.DEFINE_string("pss", "", "['address']")
    tf.app.flags.DEFINE_string("workers", "", "['address']")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task")

    def main(self, save_path="/tmp/procSoFar"):
        iterator = MNIST.read_data(MNIST.path)

        binary_image, label = iterator.get_next()
        label.set_shape((MNIST.batch))
        one_hot_labels = tf.one_hot(label, 10)
        decoded_image = tf.decode_raw(binary_image, tf.int64)

        global_step = tf.train.get_or_create_global_step()

        # print(label)

        x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

        recasted_image = tf.cast(decoded_image, tf.float32)
        reshaped_labels = tf.reshape(label, (MNIST.batch, -1))

        x_image = tf.reshape(recasted_image, [-1, 28, 28, 1], name="reshape")
        net = tf.layers.conv2d(x_image, 32, [5, 5], name='conv1')
        net = tf.layers.max_pooling2d(net, [2, 2], 1, name='pool1')
        net = tf.layers.conv2d(net, 64, [5, 5], name='conv2')
        net = tf.layers.max_pooling2d(net, [2, 2], 1, name='pool2')
        net = tf.layers.flatten(net, name='flatten')
        net = tf.layers.dense(net, 500, name='fully_connected')
        y = tf.layers.dense(net, 10, name='pred')

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=one_hot_labels))

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

        # Compute the gradients for a list of variables.
        # grads_and_vars = optimizer.compute_gradients(cross_entropy)

        # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        # capped_grads_and_vars = [gv[0] for gv in grads_and_vars]

        # Ask the optimizer to apply the capped gradients.
        # train_step = optimizer.apply_gradients(capped_grads_and_vars)

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(one_hot_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.global_variables_initializer()

        start_time = time.time()
        recovery_directory = []
        try:
            recovery_directory = os.listdir(save_path)
        except FileNotFoundError:
            try:
                os.mkdir(save_path)
            except OSError:
                print("CANNOT USE DIR", file=sys.stderr)

        if len(recovery_directory) == 0:
            recovery_path = ""
            print("EMPTY DIR", file=sys.stderr)
        else:
            recovery_path = save_path
            print("CHECKPOINTS FOUND", file=sys.stderr)

        with tf.train.MonitoredTrainingSession(summary_dir=save_path, checkpoint_dir=recovery_path) as mon_sess:
            mon_sess.run(init_op)
            step = 0
            while step < MNIST.steps:
                # gv = mon_sess.run(grads_and_vars)
                # for g in gv:
                # capped = [g[0] for g in gv]
                # np.save("grads",mon_sess.run(tf.quantize(capped,0,1, tf.float64)))

                _, acc, step, ent = mon_sess.run([train_step, accuracy, global_step, cross_entropy])
                print(ent)
                print("attempting to serialize ", acc, step)
                self.progress += [{'step': str(step), 'acc': str(acc), 'loss': str(ent)}]
                f = open('./progress', "w+")
                f.write(json.dumps(self.progress))
                f.close()
                # step +=1
                if step % MNIST.print_rate == 0:
                    print("Worker : {}, Step: {}, Accuracy (batch): {}".format(0, step, acc))

        end_time = time.time()
        print("Elapsed time: ", end_time - start_time)


if __name__ == '__main__':
    host, port = "localhost", 9000
    print("Starting server on host: `", host, "` with port: `", port, "`", sep="")

    mnist = MNIST()
    process = Process(target=mnist.main, args=("/tmp/mnistRecovery",))
    server =  ThreadingTCPServerWithProcess(process, mnist, (host,port), TCPHandler)
    future = None
    # with cf.ProcessPoolExecutor(1) as executor:
    #     try:
            # None
            # future = executor.submit(server.serve_forever())

        # except Exception as e:
        #     print(e)
        #     future.cancel()

        # finally:
        #     server.shutdown()
        #     future.cancel()
    try:
        server.serve_forever()
    except  Exception as e:
        print(e)
    finally:
        server.shutdown()

