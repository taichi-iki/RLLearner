# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from learners.base import BaseLearner

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
from chainer import optimizers

class QFunction(chainer.Chain):
    def __init__(
            self,
            input_dim=256,
            hidden_dim=256,
            output_dim=256
        ):
        super(QFunction, self).__init__(
                embed=L.EmbedID(input_dim, hidden_dim),
                fc1=L.Linear(hidden_dim, hidden_dim),
                fc2=L.Linear(hidden_dim, output_dim),
            )

    def __call__(self, x):
        h = self.embed(x)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h

class RLLearner(BaseLearner):
    def __init__(
            self,
            action_count=256,
            replay_memory_size=1e4,
            initial_eps=0.50,
            alpha=0.001,
            gamma=0.95,
            weight_decay=0.0001,
            minibatch_size=256
        ):
        self.q_function = QFunction()
        self.xp = self.q_function.xp
        self.optimizer = optimizers.SGD(alpha)
        self.optimizer.setup(self.q_function)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
        self.replay_memory = []
        self.replay_memory_size = replay_memory_size
        self.minibatch_size = minibatch_size
        self.action_count = action_count
        self.initial_eps = initial_eps
        self.latest_phi = None
        self.latest_action = None
        self.latest_reward = None
        self.gamma = gamma
        self.mode_count = 0
        self.exploration = True
    
    def push_memory(self, x):
        if len(self.replay_memory) >= self.replay_memory_size:
            self.replay_memory.pop(0)
        self.replay_memory.append(x)
    
    def draw_memory(self, max_count):
        l = len(self.replay_memory)
        if l <= max_count:
            return self.replay_memory
        else:
            return [self.replay_memory[i] for i in self.xp.random.choice(l, max_count)]
    
    def update_with_minibatch(self):
        minibatch = self.draw_memory(self.minibatch_size)
        if len(minibatch) > 0:
            minibatch_size = len(minibatch)
            minibatch = self.xp.asarray(minibatch, dtype='float32')
            phi_j = self.xp.asarray(minibatch[:, 0], dtype='int32')
            a_j = self.xp.asarray(minibatch[:, 1], dtype='int32')
            r_j = minibatch[:, 2]
            phi_jj = self.xp.asarray(minibatch[:, 3], dtype='int32')
            y = r_j + self.gamma*F.max(self.q_function(phi_jj), axis=1)
            q_j = self.q_function(phi_j)
            t = (self.xp.arange(0, q_j.data.shape[1])[None, :] == a_j[:, None])
            loss = F.sum((y - F.sum(q_j*t, axis=1))**2)/ minibatch_size

                def select_random_action(self):
        return self.xp.random.randint(0, self.action_count)
    
    def select_optimal_action(self, phi):
        x = self.xp.asarray([phi], dtype='int32')
        return self.xp.argmax(self.q_function(x).data)

    def reward(self, reward):
        self.latest_reward = reward

    def next(self, s):
        phi = ord(s)
        if not self.latest_reward is None:
            self.push_memory([self.latest_phi, self.latest_action, self.latest_reward, phi])
        self.update_with_minibatch()
        if self.mode_count >= 5000:
            self.mode_count = 0
            self.exploration = not self.exploration
        self.mode_count += 1
        if self.exploration and (self.xp.random.uniform(0, 1.0) <= self.initial_eps) :
            a = self.select_random_action()
        else:
            a = self.select_optimal_action(phi)
        self.latest_phi = phi
        self.latest_action = a
        print(a, self.exploration)
        return chr(a)
