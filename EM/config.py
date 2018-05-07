import tensorflow as tf

flags = tf.app.flags

############################
#    hyper parameters      #
############################
flags.DEFINE_float('ac_lambda0', 0.01, '\lambda in the activation function a_c, iteration 0')
flags.DEFINE_float('ac_lambda_step', 0.01,
                   'It is described that \lambda increases at each iteration with a fixed schedule, however specific super parameters is absent.')

flags.DEFINE_integer('batch_size', 50, 'batch size')
# flags.DEFINE_integer('epoch', 75, 'epoch')
flags.DEFINE_integer('epoch', 100, 'epoch')
flags.DEFINE_integer('iter_routing', 1, 'number of iterations')
flags.DEFINE_float('m_schedule', 0.2, 'the m will get to 0.9 at current epoch')
flags.DEFINE_float('epsilon', 1e-9, 'epsilon')
flags.DEFINE_integer('epoch_save', 25, 'save after number of epochs')
# flags.DEFINE_integer('epoch_save', 1, 'save after number of epochs')

################################
#    structure parameters      #
################################
flags.DEFINE_integer('A', 32, 'number of channels in output from ReLU Conv1')
flags.DEFINE_integer('B', 8, 'number of capsules in output from PrimaryCaps')
flags.DEFINE_integer('C', 16, 'number of channels in output from ConvCaps1')
flags.DEFINE_integer('D', 16, 'number of channels in output from ConvCaps2')

############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'data/mnist', 'the path for dataset')
# flags.DEFINE_string('dataset', 'data/landmark_chunks', 'the path for dataset')
flags.DEFINE_boolean('is_train', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_string('test_logdir', 'test_logdir', 'test logs directory')

cfg = tf.app.flags.FLAGS


def get_coord_add(dataset_name):
    import numpy as np
    options = {'mnist': ([[[8., 8.], [12., 8.], [16., 8.]],
                          [[8., 12.], [12., 12.], [16., 12.]],
                          [[8., 16.], [12., 16.], [16., 16.]]], 28.),
               'landmark': ([[[8., 8.], [12., 8.], [16., 8.]],
                          [[8., 12.], [12., 12.], [16., 12.]],
                          [[8., 16.], [12., 16.], [16., 16.]]], 28.)
               }

    coord_add, scale = options[dataset_name]
    coord_add = np.array(coord_add, dtype=np.float32) / scale

    return coord_add


def get_dataset_size_train(dataset_name):
    options = {'mnist': 10000}
    # options = {'mnist': 60000}
    return options[dataset_name]


def get_dataset_size_test(dataset_name):
    options = {'mnist': 10000}
    return options[dataset_name]


def get_num_classes(dataset_name):
    options = {'mnist': 10,
               'landmark': 50}
    return options[dataset_name]


from utils import create_inputs_mnist, create_inputs_landmark


def get_create_inputs(dataset_name, is_train, epochs, chunk_id = 0):
    options = {'mnist': lambda: create_inputs_mnist(is_train),
               'landmark': lambda: create_inputs_landmark(is_train, chunk_id=chunk_id)}
    return options[dataset_name]
