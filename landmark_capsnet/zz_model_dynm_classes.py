import tensorflow as tf
from zz_layer_dynm_classes import CapsLayer
from zz_config_face import args


class CapsNet():
    def __init__(self, inpu, label, name='capsnet'):
        #self.is_training=is_training
        self.inpu = inpu
        self.label = label
        self.name = name
        self.batch_size = inpu.get_shape().as_list()[0]

    
    def build(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if args.height == args.width and args.height == 168:
                conv0 = tf.contrib.layers.conv2d(self.inpu, num_outputs=128,
                                            kernel_size=9, stride=7,
                                             padding="SAME")
            else:
                conv0 = self.inpu
            conv1 = tf.contrib.layers.conv2d(conv0, num_outputs=256,
                                         kernel_size=9, stride=1,
                                         padding="VALID")
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False,
                                layer_type="CONV")
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)

            digitCaps = CapsLayer(num_outputs=args.num_class, vec_len=16, with_routing=True,
                              layer_type="FC")
            self.caps2 = digitCaps(caps1)
            print(self.caps2.get_shape())

            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                           axis=2, keep_dims=True) + args.epsilon)
            self.softmax_v = tf.nn.softmax(self.v_length, dim=1)
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(self.batch_size, ))

            #masked_v =[]
            #for batch_size in range(self.batch_size):
            #    v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
            #    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))
            #self.masked_v = tf.concat(masked_v, axis=0)
            
            #vector_j = tf.reshape(self.masked_v, shape=(self.batch_size, -1))
            #fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=1)
            #fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1)
            #self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=1, activation_fn=tf.sigmoid)



    def calc_loss(self):
        print(self.v_length.get_shape(), self.label.get_shape())
        max_l = tf.square(tf.maximum(0., 0.9 - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - 0.1))

        max_l = tf.reshape(max_l, shape=(self.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(self.batch_size, -1))


        T_c = self.label
        L_c = T_c * max_l + 0.5 * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        #orgin = tf.reshape(self.inpu, shape=(self.batch_size, -1))
        #squared = tf.square(self.decoded - orgin)
        #self.reconstruction_err = tf.reduce_mean(squared)

        self.total_loss = self.margin_loss #+ 0 * self.reconstruction_err
        
    


        
