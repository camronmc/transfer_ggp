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
REPLAY_SIZE = 20000


class Model:
    def __init__(self, propnet):
        self.roles = propnet.roles
        self.legal_for = propnet.legal_for
        self.id_to_move = propnet.id_to_move
        self.num_actions = {role: len(actions)
                            for role, actions in propnet.legal_for.items()}
        self.num_inputs = len(propnet.propositions)
        self.replay_buffer = collections.deque(maxlen=REPLAY_SIZE)
        self.create_model()
        # self.sess = tf.compat.v1.Session(
        #     config=tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)
        # )
        # init_op = tf.compat.v1.global_variables_initializer()
        # self.sess.run(init_op)
        # self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

        self.eval_time = self.train_time = 0
        self.losses = []

    def custom_loss_q(self, q_target, q_pred):
        loss = tf.reduce_mean(input_tensor=tf.keras.losses.MSE(q_pred, q_target))

        # Add weight regularisation?
        vars = self.model.trainable_variables
        self.C = 0.0001
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.C
        loss += lossL2

        return loss
    
    def custom_loss_prob(self, p_target, p_pred):
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss = tf.reduce_mean(input_tensor=cce(p_target, p_pred))

        return loss

    def custom_loss_prob_nologit(self, p_target, p_pred):
        return tf.zeros((1,1))

    def create_model(self):
        dense = tf.keras.layers.Dense
        self.input = tf.keras.Input(shape=(None,self.num_inputs), dtype=tf.float32)
        
        cur = self.input
        size = self.num_inputs
        while size > MIN_PRE_SIZE:
            size = max(MIN_PRE_SIZE, size // 2)
            cur = tf.keras.layers.Dense(size, activation=tf.nn.relu)(cur)

        self.state_features = cur
        state_size = size

        self.outputs = {}
        for role in self.roles:

            cur = self.state_features

            size = state_size
            while size * 2 < self.num_actions[role]:
                size *= 2
                cur = tf.keras.layers.Dense(size, activation=tf.nn.relu)(cur)
            final = self.num_actions[role]
            logits = tf.keras.layers.Dense(final, activation=None, name="logits_"+role)(cur)
            probs = tf.keras.layers.Softmax(name="probs_"+role)(logits)
            q = dense(1, activation=tf.nn.sigmoid, name = "q_"+role)(cur)
            self.outputs[role] = (q, logits, probs)

        self.model = tf.keras.Model(
            inputs=[self.input],
            outputs=[self.outputs[roles] for roles in self.roles]
        )

        self.model.summary()

        # update if more than two player
        if(len(self.roles)> 2):
            print("CHANGE THIS!!")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.01), 
            loss = {
                "q_"+self.roles[0] : self.custom_loss_q,
                "q_"+self.roles[1] : self.custom_loss_q,
                "logits_"+self.roles[0] : self.custom_loss_prob,
                "logits_"+self.roles[1] : self.custom_loss_prob,
                "probs_"+self.roles[0] : self.custom_loss_prob_nologit,
                "probs_"+self.roles[1] : self.custom_loss_prob_nologit
            },
            loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

        ##set up callback/saver
        path = os.path.join('models', game)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        self.callback = tf.keras.callbacks.ModelCheckpoint(
                            filepath=path, 
                            verbose=1, 
                            save_weights_only=True,
                            period=5
                            )

        # self.target = {role: (tf.keras.Input(dtype=tf.float32, shape=(None, 1)),
        #                       tf.keras.Input(dtype=tf.float32, shape=(None, self.num_actions[role])))
        #                for role in self.roles}
        # self.loss = 0
        # for role, (q, logits, probs) in self.outputs.items():
        #     tq, tprobs = self.target[role]
        #     # TODO: add weights based on distance from terminal state?
        #     # self.loss += tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=tprobs, logits=logits)) reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        #     cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        #     self.loss += tf.reduce_mean(input_tensor=cce(tprobs, logits))
        #     # self.loss += tf.reduce_mean(input_tensor=tf.compat.v1.losses.mean_squared_error(q, tq))
        #     self.loss += tf.reduce_mean(input_tensor=tf.keras.losses.MSE(q, tq))

        # # Add weight regularisation?
        # vars = self.model.trainable_variables
        # self.C = 0.0001
        # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.C
        # self.loss += lossL2

    def train(self, epochs=5, batchsize=128):
        # Sample from replay buffer and train
        # if len(replay_buffer) < batchsize:
            # print('Skipping as replay buffer only has', len(replay_buffer), 'items')
        # batchsize = min(batchsize, len(self.replay_buffer))
        if batchsize > len(self.replay_buffer):
            print('Skipping as replay buffer too small')
            return
        sum_loss = 0
        for i in range(epochs):
            sample = random.sample(self.replay_buffer, batchsize)
            feed_dict = {
                self.input: [x[0] for x in sample],
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

    def add_sample(self, state, probs, scores):
        self.replay_buffer.append((state, probs, scores))

    def eval(self, state):
        feed_dict = {self.input: [state]}
        all_qs = {}
        all_probs = {}
        for role, outp in self.outputs.items():
            start = time.time()
            q, _, probs = self.sess.run(outp, feed_dict=feed_dict)
            self.eval_time += time.time() - start
            all_qs[role] = q[0][0]
            all_probs[role] = {}
            for prob, inp in zip(probs[0], self.legal_for[role]):
                all_probs[role][inp.id] = prob
        return all_probs, all_qs

    def print_eval(self, state):
        probs, qs = self.eval(state)
        # for role in self.roles:
        #     print('Role', role, 'expected return:', qs[role])
        #     for i, pr in probs[role].items():
        #         print(self.id_to_move[i].move_gdl, '%.3f' % pr)

    def train(self, epochs=5, batchsize=128):
        # Sample from replay buffer and train
        # if len(replay_buffer) < batchsize:
            # print('Skipping as replay buffer only has', len(replay_buffer), 'items')
        # batchsize = min(batchsize, len(self.replay_buffer))
        if batchsize > len(self.replay_buffer):
            print('Skipping as replay buffer too small')
            return
        sum_loss = 0
        for i in range(epochs):
            sample = random.sample(self.replay_buffer, batchsize)
            feed_dict = {
                self.input: [x[0] for x in sample],
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
