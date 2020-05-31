import tensorflow as tf
import collections
import random
import pathlib
import time
import os

NUM_PRE_LAYERS = 2
MIN_PRE_SIZE = 50
NUM_POST_LAYERS = 2
MIN_POST_SIZE = 100
REPLAY_SIZE = 50000


class Model:
    def __init__(self, propnet):
        self.roles = propnet.roles
        self.num_actions = {role: len(actions)
                            for role, actions in propnet.legal_for.items()}
        self.num_inputs = len(propnet.propositions)
        self.replay_buffer = collections.deque(maxlen=REPLAY_SIZE)
        self.create_model()
        self.create_trainer()
        self.sess = tf.Session(
            config=tf.ConfigProto(inter_op_parallelism_threads=1)
        )
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.saver = tf.train.Saver(max_to_keep=None)
        self.eval_time = self.train_time = 0
        self.losses = []

    def make_residule_layer(self, inp, size, N=3):
        inp = tf.layers.dense(inp, size, activation=tf.nn.relu)
        x = inp
        for i in range(N):
            x = tf.layers.dense(x, size, activation=tf.nn.relu)
        x = tf.add(inp, x)
        return x

    def create_model(self):
        dense = tf.layers.dense

        self.input = tf.placeholder(shape=[None, self.num_inputs], dtype=tf.float32)
        cur = self.input
        size = self.num_inputs
        while size > MIN_PRE_SIZE:
            size = max(MIN_PRE_SIZE, size // 1.5)
            cur = self.make_residule_layer(cur, size)
            # cur = dense(cur, size, activation=tf.nn.relu)
            print('residule block of size', size)
        cur = self.make_residule_layer(cur, size)
        # cur = dense(cur, size, activation=tf.nn.relu)
        print('layer of size', size)
        # for i in range(NUM_PRE_LAYERS):
            # size = max(MIN_PRE_SIZE, size // 2)
            # cur = dense(cur, size, activation=tf.nn.relu)

        self.state_features = cur
        state_size = size

        self.outputs = {}
        for role in self.roles:
            print('role head for', role)
            # sizes = [self.num_actions[role]]
            # for i in range(NUM_POST_LAYERS):
                # sizes.append(max(MIN_POST_SIZE, sizes[-1] // 2))
            # sizes.reverse()

            cur = self.state_features
            # for size in sizes[:-1]:
                # cur = dense(cur, size, activation=tf.nn.relu)
            # final = sizes[-1]  # one more for expected reward
            size = state_size
            while size * 1.1 < self.num_actions[role]:
                size *= 1.1
                cur = dense(cur, size, activation=tf.nn.relu)
                print('layer of size', size)
            final = self.num_actions[role]
            logits = dense(cur, final, activation=None)
            probs = tf.nn.softmax(logits)
            q = dense(cur, 1, activation=tf.nn.sigmoid)
            print('final of sizes', final, 'and 1')
            self.outputs[role] = (q, logits, probs)

    def create_trainer(self):
        self.target = {role: (tf.placeholder(tf.float32, shape=(None, 1)),
                              tf.placeholder(tf.float32, shape=(None, self.num_actions[role])))
                       for role in self.roles}
        self.loss = 0
        for role, (q, logits, probs) in self.outputs.items():
            tq, tprobs = self.target[role]
            # self.loss += tf.reduce_mean(tf.losses.mean_squared_error(probs, tprobs))
            # TODO: add weights based on distance from terminal state?
            self.loss += tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=tprobs, logits=logits))
            self.loss += tf.reduce_mean(tf.losses.mean_squared_error(q, tq))

        # Add weight regularisation?
        vars = tf.trainable_variables()
        self.C = 0.0001
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.C
        self.loss += lossL2

        self.optimiser = tf.train.AdamOptimizer(0.01)
        self.trainer = self.optimiser.minimize(self.loss)

    def add_sample(self, state, probs, scores):
        self.replay_buffer.append((state, probs, scores))

    def eval(self, propnet):
        state = [[x.eval(propnet.data) for x in propnet.propositions]]
        feed_dict = {self.input: state}
        all_qs = {}
        all_probs = {}
        for role, outp in self.outputs.items():
            start = time.time()
            q, _, probs = self.sess.run(outp, feed_dict=feed_dict)
            self.eval_time += time.time() - start
            all_qs[role] = q[0][0]
            all_probs[role] = {}
            for prob, inp in zip(probs[0], propnet.legal_for[role]):
                all_probs[role][inp.id] = prob
        return all_probs, all_qs

    def print_eval(self, propnet):
        probs, qs = self.eval(propnet)
        for role in self.roles:
            print('Role', role, 'expected return:', qs[role])
            for i, pr in probs[role].items():
                print(propnet.id_to_move[i].move_gdl, '%.3f' % pr)

    def train(self, epochs=5, batchsize=128):
        # Sample from replay buffer and train
        if batchsize > len(self.replay_buffer):
            print('Skipping as replay buffer too small')
            return
        sum_loss = 0
        for i in range(epochs):
            sample = random.sample(self.replay_buffer, batchsize)
            feed_dict = {
                self.input: [x[0][0] for x in sample],
            }
            for role in self.roles:
                tq, tprobs = self.target[role]
                feed_dict[tq] = [[x[2][role]] for x in sample]
                feed_dict[tprobs] = [x[1][role] for x in sample]
            start = time.time()
            _, loss = self.sess.run((self.trainer, self.loss), feed_dict=feed_dict)
            self.train_time += time.time() - start
            print('Loss is', loss)
            sum_loss += loss
        self.losses.append(sum_loss/epochs)

    def save(self, game, i):
        path = os.path.join('models', game)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        save_path = self.saver.save(self.sess, path + '/step-%06d.ckpt' % i)
        print('Saved model to', save_path)

    def load(self, path):
        self.saver.restore(self.sess, path)
        print('Loaded model from', path)

    def load_most_recent(self, game):
        models = os.path.join(pathlib.Path(__file__).parent, 'models')
        path = os.path.join(models, game)
        newest = max(os.listdir(path))[:-5]
        self.load(os.path.join(path, newest))
