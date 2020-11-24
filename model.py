import tensorflow as tf
import collections
import random
import pathlib
import time
import os
import numpy as np

NUM_PRE_LAYERS = 2
MIN_PRE_SIZE = 50
NUM_POST_LAYERS = 2
MIN_POST_SIZE = 100
REPLAY_SIZE = 20000


class Model:
    def __init__(self, propnet, create=True, transfer=False, base_dims=None, roles_dim=None, multiNet=False, games=[], replay_buffer=dict()):
        self.multiNet = multiNet
        self.games = games
        self.roles = propnet.roles
        self.legal_for = propnet.legal_for
        self.id_to_move = propnet.id_to_move
        self.num_actions = {role: len(actions)
                            for role, actions in propnet.legal_for.items()}
        for role in self.roles:
            self.genNumActions = self.num_actions[role]
        self.num_inputs = len(propnet.propositions)
        if self.multiNet:
            self.replay_buffer = replay_buffer
        else: 
            self.replay_buffer = collections.deque(maxlen=REPLAY_SIZE)
        if transfer:
            self.create_model_transfer(base_dims, roles_dim)
        else:
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

    def create_model(self):
        dense = tf.layers.dense

        self.input = tf.placeholder(shape=[None, self.num_inputs], dtype=tf.float32)
        cur = self.input
        size = self.num_inputs
        while size > MIN_PRE_SIZE:
            size = max(MIN_PRE_SIZE, size // 2)
            cur = dense(cur, size, activation=tf.nn.relu)

        self.state_features = cur
        state_size = size

        self.outputs = {}
        for role in self.roles:
            cur = self.state_features
            size = state_size
            while size * 2 < self.num_actions[role]:
                size *= 2
                cur = dense(cur, size, activation=tf.nn.relu)
            final = self.num_actions[role]
            logits = dense(cur, final, activation=None)
            probs = tf.nn.softmax(logits)
            q = dense(cur, 1, activation=tf.nn.sigmoid)
            self.outputs[role] = (q, logits, probs)

    def create_model_transfer(self, base_dims, roles_dim):
        dense = tf.layers.dense

        self.input = tf.placeholder(shape=[None, self.num_inputs], dtype=tf.float32, name="new_input")
        cur = self.input
        size = self.num_inputs
        while size / 2 > base_dims[0]:
            size = max(MIN_PRE_SIZE, size // 2)
            cur = dense(cur, size, activation=tf.nn.relu, name="new_"+str(size))

        tracker = 0
        for i, s in enumerate(base_dims):
            if i == 0:
                cur = dense(cur, s, activation=tf.nn.relu, name="new_"+str(s))
            else:
                cur = dense(cur, s, activation=tf.nn.relu)
            tracker=i

        self.base_features = cur

        self.outputs = {}
        for role in self.roles:
            cur = self.base_features

            for s in roles_dim:
                tracker+=1
                cur = dense(cur, s, activation=tf.nn.relu)
                size = s

            #add extra layers if necessary
            while size * 2 < self.num_actions[role]:
                size *= 2
                cur = dense(cur, size, activation=tf.nn.relu, name="new_act_"+str(size))

            final = self.num_actions[role]
            logits = dense(cur, final, activation=None)
            probs = tf.nn.softmax(logits)
            q = dense(cur, 1, activation=tf.nn.sigmoid)
            self.outputs[role] = (q, logits, probs)

    def complete_transfer(self, path):
        return
            
    def create_trainer(self):
        self.target = {role: (tf.placeholder(tf.float32, shape=(None, 1)),
                              tf.placeholder(tf.float32, shape=(None, self.num_actions[role])))
                       for role in self.roles}
        self.loss = 0
        for role, (q, logits, probs) in self.outputs.items():
            tq, tprobs = self.target[role]
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

    def add_sample(self, state, probs, scores, game=""):
        if self.multiNet:
            if game not in self.games:
                print("tried to add sample from game not in multi net")
                exit(0)
            self.replay_buffer[game].append((state, probs, scores))
        else:
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
            # print("self.legal_for is:", self.legal_for[role])
            # print("prob preds is:", probs[0])
            for prob, inp in zip(probs[0], self.legal_for[role]):
                # print(prob, inp)
                all_probs[role][inp.id] = prob

        # print(all_probs)
        return all_probs, all_qs

    def print_eval(self, state):
        probs, qs = self.eval(state)
        for role in self.roles:
            print('Role', role, 'expected return:', qs[role])
            for i, pr in probs[role].items():
                print(self.id_to_move[i].move_gdl, '%.3f' % pr)

    def getBuffer(self):
        return self.replay_buffer

    def train(self, epochs=5, batchsize=128, game=""):
        # Sample from replay buffer and train
        # if len(replay_buffer) < batchsize:
            # print('Skipping as replay buffer only has', len(replay_buffer), 'items')
        # batchsize = min(batchsize, len(self.replay_buffer))
        bufferLen = 0
        if self.multiNet:
            if game not in self.games:
                print("game was not in games for multinet")
                exit(0)
            bufferLen = len(self.replay_buffer[game])
        else:
            bufferLen = len(self.replay_buffer)

        if batchsize > bufferLen:
            print('Skipping as replay buffer too small')
            return
        sum_loss = 0
        for i in range(epochs):
            if self.multiNet:
                sample = random.sample(self.replay_buffer[game], batchsize)
            else:
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

    def save(self, model_name, i, transfer=False, multiNet=False):
        if transfer:
            path = os.path.join('models_transfer', model_name)
        elif multiNet:
            path = os.path.join('multiNet', model_name)
        else:
            path = os.path.join('models', model_name)
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

    def print_var(self, var):
        with tf.variable_scope(var, reuse=True):
            w = tf.get_variable("kernel")
            print(w.eval(self.sess))

    def clear_output_layer(self):
        print('clearing output layer')
        biasZeros = np.zeros((self.genNumActions,))
        kernelZeros = np.zeros((MIN_PRE_SIZE,self.genNumActions))

        bZeros = tf.Variable(initial_value=biasZeros, name="bzeros", dtype=tf.float32)
        kZeros = tf.Variable(initial_value=kernelZeros, name="kzeros", dtype=tf.float32)

        newVarsInit = tf.variables_initializer([bZeros,kZeros])
        self.sess.run(newVarsInit)

        for v in tf.global_variables():
            if v.shape == bZeros.shape:
                self.sess.run(v.assign(bZeros))
            elif v.shape == kZeros.shape:
                self.sess.run(v.assign(kZeros))

    def perform_transfer(self, path, reuse_output=False, multiplayer=False, pad_0=False, breakthroughMap = False):
        reader = tf.train.NewCheckpointReader(path)
        saved_shapes = reader.get_variable_to_shape_map()

        var_names = sorted([(var.name, var.name.split(':')[0]) for
                            var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        rearrange_vars = []
        name2var = dict(zip(map(lambda x: x.name.split(':')[0],
                                tf.global_variables()),
                            tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
                else:
                    rearrange_vars.append(var_name)

        saver = tf.train.Saver(restore_vars)
        saver.restore(self.sess, path)

        if multiplayer:
            threepmodel = dict()
            for shape in saved_shapes:
                if shape[0:7] == "dense_4":
                    equiv = "dense_8"
                    var = equiv+shape[7:]
                    p1LogitVar = reader.get_tensor(shape)
                    threepmodel[var] = tf.Variable(initial_value=p1LogitVar, name="output-"+var, dtype=tf.float32)
                if shape[0:7] == "dense_5":
                    equiv = "dense_9"
                    var = equiv+shape[7:]
                    p1Qvar = reader.get_tensor(shape)
                    threepmodel[var] = tf.Variable(initial_value=p1Qvar, name="output-"+var, dtype=tf.float32)

            newVarsInit = tf.variables_initializer(threepmodel.values())
            self.sess.run(newVarsInit)

            for v in tf.global_variables():
                var = v.name.split(':')[0]
                if var in threepmodel:
                    self.sess.run(v.assign(threepmodel[var]))

        if reuse_output:
            if pad_0:

                ## UPDATE THIS METHOD BASED ON TRANSFER CURRENTLY PAD 0s
                checkpoint_vars = dict()
                for var in rearrange_vars:
                    var = var.split(':')[0]
                    npVar = reader.get_tensor(var)
                    if 'bias' in var:
                        #pad 2 after
                        finVar = np.pad(npVar, (0,2))
                    else:
                        # (50,8) need to transform to (50,10)
                        finVar = np.zeros((50,10))
                        finVar[:npVar.shape[0], :npVar.shape[1]] = npVar

                    checkpoint_vars[var] = tf.Variable(initial_value=finVar, name="output-"+var, dtype=tf.float32)
            elif breakthroughMap: 
                checkpoint_vars = dict()
                for var in rearrange_vars:
                    var = var.split(':')[0]
                    npVar = reader.get_tensor(var)
                    if 'bias' in var:
                        finVar = np.random.rand(66)
                        finVar = np.concatenate((npVar[:57], npVar[58:60], npVar[61:63], npVar[64:66], npVar[67:69], npVar[80:81]))

                        
                    else:
                        finVar = np.random.rand(50,66)
                        for i in range(50):
                            temp = npVar[i]
                            finVar[i] = np.concatenate((temp[:57], temp[58:60], temp[61:63], temp[64:66], temp[67:69], temp[80:81]))
                            
                    checkpoint_vars[var] = tf.Variable(initial_value=finVar, name="output-"+var, dtype=tf.float32)

                
            else:
                ## pad mean
                checkpoint_vars = dict()
                for var in rearrange_vars:
                    var = var.split(':')[0]
                    npVar = reader.get_tensor(var)
                    if 'bias' in var:
                        #pad 2 after
                        finVar = np.pad(npVar, (0,2), mode='mean')
                    else:
                        # (50,8) need to transform to (50,10)
                        finVar = np.zeros((50,10))

                        for i in range(50):
                            finVar[i] = np.pad(npVar[i], (0,2), mode='mean')

                    checkpoint_vars[var] = tf.Variable(initial_value=finVar, name="output-"+var, dtype=tf.float32)

            newVarsInit = tf.variables_initializer(checkpoint_vars.values())
            self.sess.run(newVarsInit)

            for v in tf.global_variables():
                print(v)
                var = v.name.split(':')[0]
                if var in checkpoint_vars:
                    print('here')
                    self.sess.run(v.assign(checkpoint_vars[var]))


        

        

            
    