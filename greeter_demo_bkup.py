# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple demo which greets several people."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os.path
import threading
import tensorflow as tf
import time
# TODO(chihuahua): Figure out why pylint invalidates this import.
import greeter_summary  # pylint: disable=import-error
from pynput import keyboard
# Directory into which to write tensorboard data.
LOGDIR = '/home/ahmad/tplug/tensorboard-plugin-example-master'


def run(sess, logdir, run_name, characters, extra_character):
  # print("OMAKx")
  """Greet several characters from a given cartoon."""

  tf.reset_default_graph()

  input_ = tf.placeholder(tf.string)

  summary_op = greeter_summary.op("greetings", input_)

  writer = tf.summary.FileWriter(os.path.join(logdir, run_name))

  sess = tf.Session()

  for character in characters:
    summary = sess.run(summary_op, feed_dict={input_: character})
    writer.add_summary(summary)

  # Demonstrate that we can also add summaries without using the
  # TensorFlow session or graph.
  summary = greeter_summary.pb("greetings", extra_character)
  writer.add_summary(summary)

  writer.close()
  s = 0
  for i in range(100000000):
      s +=i

  print(s)
  # time.sleep(10)

def stop (sess):
    print('Thread Started')
    sess.close()
    os.kill(sess,9)
    # while(True):
        # input = sys.stdin.readline()
        # if(input == 'p'):
            # sess.close()
            # os.kill(sess,9)
            # print('Training paused')
            # break

def run_all(sess, logdir, unused_verbose=False):
  """Run the simulation for every logdir.
  """
  run(sess,logdir, "steven_universe", ["Garnet", "Amethyst", "Pearl"], "Steven")
  run(sess,logdir, "futurama", ["Fry", "Bender", "Leela"],
      "Lrrr, ruler of the planet Omicron Persei 8")

def on_press(key, sess=None):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        if key.char == 'p':
            print("PAUSED")
            stop(sess)

    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
        key))
    if key.char == "p":
        # Stop listener
        return False

# Collect events until released

def main(unused_argv):
  sess = tf.Session()
  # thread1 = threading.Thread(target = stop, args=(sess,))
  # thread1.daemon = True

  # with keyboard.Listener(
    #    on_press=on_press,
    #    on_release=on_release) as listener:
    # thread1 = threading.Thread(target=listener.join(), args=(sess, LOGDIR, True,))
    # thread1.start()
    #listener.join()


  #thread2 = threading.Thread(target=run_all, args=(sess, LOGDIR, True,))
  # thread2.daemon = True

  print('Saving output to %s.' % LOGDIR)
  # run_all(sess, LOGDIR, unused_verbose=True)
  #thread2.start()
  # thread1.start()
  print('Done. Output saved to %s.' % LOGDIR)


if __name__ == '__main__':
  tf.app.run()
