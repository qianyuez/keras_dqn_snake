import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras import initializers
import matplotlib.pyplot as plt

class DeepQNetwork():
    def __init__(self,
                 n_actions,
                 n_states,
                 dense_units = 20,
                 learning_rate=0.0005,
                 reward_decay=0.9,
                 e_greedy=0.95,
                 update_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 double_dqn=False,
                 e_greedy_increment=None):
        self.n_actions = n_actions
        self.n_states = n_states
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.epsilon_max = e_greedy
        self.update_target_iter = update_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = []
        self.costs = []

        self.q_eval_model = self._build_model()
        self.q_target_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.n_states))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(300, activation='relu'))
        model.add(Dense(self.n_actions))
        model.compile(optimizer=RMSprop(lr=self.learning_rate), loss='mse')
        return model

    def update_model(self):
        self.q_target_model.set_weights(self.q_eval_model.get_weights())

    def save_model(self, path):
        self.q_eval_model.save_weights(path)

    def load_model(self, path):
        self.q_eval_model.load_weights(path)

    def store_transition(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))
        self.memory_counter += 1
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_value = self.q_eval_model.predict(state)
            action = action_value.argmax()
        else:
            action = self.random_action()
        return action

    def random_action(self):
        action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.update_target_iter == 0:
            self.update_model()
        sample_indices = np.random.choice(self.memory_size, size=self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        for index in sample_indices:
            state, action, reward, next_state = self.memory[index]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
        states = np.array(states)
        next_states = np.array(next_states)
        q_eval = self.q_eval_model.predict(states)
        q_next = self.q_target_model.predict(next_states)
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size)
        if self.double_dqn:
            q_eval_next = self.q_eval_model.predict(next_states)
            selected_q_next = q_next[batch_index, np.argmax(q_eval_next, axis=-1)]
        else:
            selected_q_next = np.max(q_next, axis=-1)
        q_target[np.arange(self.batch_size), np.array(actions, dtype=np.int32)] = np.array(rewards) + self.reward_decay * selected_q_next
        loss = self.q_eval_model.train_on_batch(states, q_target)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return loss
