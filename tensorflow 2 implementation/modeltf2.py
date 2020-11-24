import tensorflow as tf
import numpy as np
import collections
import random
import pathlib
import time
import os
import sys

NUM_PRE_LAYERS = 2
MIN_PRE_SIZE = 50
NUM_POST_LAYERS = 2
MIN_POST_SIZE = 100
REPLAY_SIZE = 20000


class Model:
    def __init__(self, propnet, game, load_only=False):
        self.game = game
        self.roles = propnet.roles
        self.legal_for = propnet.legal_for
        self.id_to_move = propnet.id_to_move
        self.num_actions = {role: len(actions)
                            for role, actions in propnet.legal_for.items()}

        actions = {role: actions
                            for role, actions in propnet.legal_for.items()}

        # for action in actions['black']:
        #     print(propnet.id_to_move[action].move_gdl)
        # print(propnet.actions)
        # print("NUM ACTION IS", self.num_actions)
        # exit(0)
        self.num_inputs = len(propnet.propositions)
        self.replay_buffer = collections.deque(maxlen=REPLAY_SIZE)
        if not load_only:
            self.create_model()
        self.eval_time = self.train_time = 0
        self.losses = []

    def custom_loss_q(self, q_target, q_pred):
        loss = tf.reduce_mean(input_tensor=tf.keras.losses.MSE(q_pred, q_target))

        # Add weight regularisation?
        vars = self.model.trainable_variables
        self.C = 0.0001
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.C
        # this was being added twice
        print("lossL2:", lossL2, lossL2/2)
        loss += lossL2/2

        return loss
    
    def custom_loss_prob(self, p_target, p_pred):
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss = tf.reduce_mean(input_tensor=cce(p_target, p_pred))

        return loss

    def custom_loss_prob_nologit(self, p_target, p_pred):
        return tf.zeros((1,1))

    def create_model(self):
        dense = tf.keras.layers.Dense
        self.input = tf.keras.Input(shape=(self.num_inputs,), dtype=tf.float32, name="input")
        
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
            outputs=[
                self.outputs[self.roles[0]][0],
                self.outputs[self.roles[0]][1], 
                self.outputs[self.roles[0]][2],
                self.outputs[self.roles[1]][0],
                self.outputs[self.roles[1]][1],
                self.outputs[self.roles[1]][2]
            ]
        )

        self.model.summary()

        # update if more than two player
        if(len(self.roles)> 2):
            print("CHANGE THIS!!")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.01), 
            loss = {
                "q_"+self.roles[0]      : self.custom_loss_q,
                "q_"+self.roles[1]      : self.custom_loss_q,
                "logits_"+self.roles[0] : self.custom_loss_prob,
                "logits_"+self.roles[1] : self.custom_loss_prob,
                "probs_"+self.roles[0]  : self.custom_loss_prob_nologit,
                "probs_"+self.roles[1]  : self.custom_loss_prob_nologit
            },
            loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

        ##set up callback/saver
        self.path = os.path.join('models', self.game)
        pathlib.Path(self.path).mkdir(parents=True, exist_ok=True)
        # print(path)

        # self.callback = tf.keras.callbacks.ModelCheckpoint(
        #                     filepath=path, 
        #                     verbose=1, 
        #                     save_weights_only=True,
        #                     period=50,
        #                     save_best_only=False
        #                     )

    def train(self, epochs=5, batchsize=128):
        if batchsize > len(self.replay_buffer):
            print('Skipping as replay buffer too small')
            return
        sum_loss = 0

        for e in range(epochs):
            sample = random.sample(self.replay_buffer,batchsize)

            inputs = np.array([x[0] for x in sample])

            tqs = dict()
            logits = dict()
            for role in self.roles:
                tqs[role] = np.empty((len(sample), 1))
                logits[role] = np.empty((len(sample), self.num_actions[role]))
                for i, x in enumerate(sample):
                    tqs[role][i] = x[2][role]
                    logits[role][i] = x[1][role]

            sum_loss += self.model.train_on_batch(
                inputs,
                y = {
                    "q_"+self.roles[0]      : tqs[self.roles[0]],
                    "q_"+self.roles[1]      : tqs[self.roles[1]],
                    "logits_"+self.roles[0] : logits[self.roles[0]],
                    "logits_"+self.roles[1] : logits[self.roles[1]],
                    "probs_"+self.roles[0]  : logits[self.roles[0]],
                    "probs_"+self.roles[1]  : logits[self.roles[1]]
                },
            )[0]

        print(sum_loss/epochs)

    def add_sample(self, state, probs, scores):
        self.replay_buffer.append((state, probs, scores))

    def eval(self, state):
        state = np.array(state)
        state = state.reshape(1,self.num_inputs)
        predictions = self.model.predict_on_batch(state)
    
        # outputs=[self.outputs[roles] for roles in self.roles]
        # (q, logits, probs) for each role
        all_qs= dict()
        all_probs=dict()
        for i, role in enumerate(self.roles):
            all_qs[role] = predictions[i*3][0]
            all_probs[role] = dict()
            probs = predictions[i*3+1][0]
            for prob, inp in zip(probs, self.legal_for[role]):
                all_probs[role][inp.id] = prob
        return all_probs, all_qs

    def print_eval(self, state):
        probs, qs = self.eval(state)
        # for role in self.roles:
        #     print('Role', role, 'expected return:', qs[role])
        #     for i, pr in probs[role].items():
        #         print(self.id_to_move[i].move_gdl, '%.3f' % pr)

    def save(self, game, i):
        modelName = self.path+'/'+game+"-"+str(i)+'.h5'
        print("saving model: " + modelName)
        self.model.save(modelName, overwrite=False)

    def load(self, path, with_training=True):
        self.model = tf.keras.models.load_model(path, compile=with_training)

    def load_most_recent(self, game):
        models = os.path.join(pathlib.Path(__file__).parent, 'models')
        path = os.path.join(models, game)
        newest = max(os.listdir(path))[:-5]
        self.load(os.path.join(path, newest))
