from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
import scipy.misc
import pickle

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = None

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', 'tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")


# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

'''
def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    #softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
#    for i in sess.graph.get_operations():
#      print( i.name)
    img = tf.placeholder(tf.uint8, (299,299,3))
    softmax_tensor = tf.import_graph_def(
            sess.graph.as_graph_def(),
            input_map={'DecodeJpeg:0': tf.reshape(img,((299,299,3)))},
            return_elements=['softmax/logits:0'])

    print(scipy.misc.imread(image).shape)
    dat = scipy.misc.imresize(scipy.misc.imread(image),(299,299))
    print(dat.shape)
    predictions = sess.run(softmax_tensor,
                           {img: dat})
    predictions = np.squeeze(predictions)
    print(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()
#    print(predictions[169])
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    print(top_k)
#    for i in range(1008):
#      print(i, node_lookup.id_to_string(i))
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
'''

def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()

  imgn = ImageNet()
  with tf.Session() as sess:

    model = InceptionModel(sess)
#    for i in sess.graph.get_operations():
#      print( i.name)
    model.pred(imgn.origin_data)

#  image = ('./tmp/imagenet/kitfox.jpg')
#  dat = scipy.misc.imresize(scipy.misc.imread(image),(299,299))
#  with tf.Session() as sess:
#    model = InceptionModel(sess)
#    model.predict(dat)

#  image = (FLAGS.image_file if FLAGS.image_file else
#           os.path.join(FLAGS.model_dir, 'cropped_pand.jpg'))
#  run_inference_on_image(image)

CREATED_GRAPH = False

class InceptionModel:
  image_size = 299
  num_labels = 1008
  num_channels = 3
  def __init__(self, sess):
    global CREATED_GRAPH
    self.sess = sess
    if not CREATED_GRAPH:
      create_graph()
      CREATED_GRAPH = True
      self.graph = self.sess.graph.as_graph_def()

    self.img = tf.placeholder(tf.uint8, (299,299,3))

    self.softmax_tensor = tf.import_graph_def(
            self.graph,
            input_map={'DecodeJpeg:0': tf.reshape(self.img,((299,299,3)))},
            return_elements=['softmax/logits:0'])

  def predict(self, data):

#    print(data.shape)
    scaled = (0.5+data)*255
    scaled = tf.cast(scaled, tf.float32)
    output = []

    for i in range(scaled.shape[0]):
      softmax_tensor = tf.import_graph_def(
            self.graph,
            input_map={'Cast:0': scaled[i]},
            return_elements=['softmax/logits:0'])
      predictions = tf.squeeze(softmax_tensor[0])
      output.append(predictions)
    
    predictions = tf.convert_to_tensor(output)

    return predictions

  def pred(self, data, norm = False):

    if norm:
      data = (data+0.5)*255
      data = data.astype(np.uint8)

    output = []
    for i in range(data.shape[0]):

      predictions = self.sess.run(self.softmax_tensor,
                           {self.img: data[i]})
      predictions = np.squeeze(predictions)
      top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]

      print(top_k, predictions[top_k[0]],predictions[top_k[1]],predictions[top_k[2]],predictions[top_k[3]],predictions[top_k[4]])
      output.append(predictions)

    return output

class Imageset():
  """docstring for Imageset"""
  def __init__(self,l):
    self.image = []
    self.l = l
    self.num = 0

class ImageNet:
  def __init__(self):
#    for filename in os.listdir('./imgs/'):
#      print(filename)
    f = open('./tmp/50/95.pkl','rb')
    imgset = pickle.load(f)
    f.close

    self.origin_data = imgset.image
    self.train_data = imgset.image/255 - 0.5
    label = np.zeros(1008)
    label[imgset.l] = 1
    self.train_labels = np.tile(label, (imgset.num,1))
    print(self.train_labels.shape)

if __name__ == '__main__':
  tf.app.run()
