import tensorflow as tf
import gym
import numpy as np
import cv2
from collections import deque
import random

class hjk1024(gym.Env):
    def __init__(self):
        self.state = None
        self.action_space = gym.spaces.Discrete(4)
        self.step_count = 0

    def add_1_or_2(self):
        empty_cell = []
        for i in range(4):
            for j in range(4):
                if (self.state[i][j] == 0):
                    empty_cell.append([i, j])
        if (len(empty_cell) > 0):
            selected_i, selected_j = empty_cell[np.random.choice(len(empty_cell), 1)[0]]
            if (np.random.rand() < 0.9):
                self.state[selected_i][selected_j] = 1
            else:
                self.state[selected_i][selected_j] = 2

    def left_compress(self, map):
        new_state = np.zeros((4, 4), dtype=np.int64)
        changed = False
        for i in range(4):
            pos = 0
            for j in range(4):
                if (map[i][j] != 0):
                    new_state[i][pos] = map[i][j]
                    if (j != pos):
                        changed = True
                    pos = pos + 1
        return new_state, changed

    def left_combine(self, map):
        changed = False
        for i in range(4):
            for j in range(3):
                if (map[i][j] == map[i][j + 1] and map[i][j] != 0):
                    map[i][j] = map[i][j] + 1
                    map[i][j + 1] = 0
                    changed = True

        return map, changed

    def move_left(self, map):
        n_state, changed_1 = self.left_compress(map)
        n_state, changed_2 = self.left_combine(n_state)
        n_state, _ = self.left_compress(n_state)
        changed = changed_1 or changed_2
        return n_state, changed

    def move_down(self, map):
        n_state = self.turn_90(map)
        n_state, changed = self.move_left(n_state)
        n_state = self.turn_90(n_state)
        n_state = self.turn_90(n_state)
        n_state = self.turn_90(n_state)
        return n_state, changed

    def move_right(self, map):
        n_state = self.turn_90(map)
        n_state = self.turn_90(n_state)
        n_state, changed = self.move_left(n_state)
        n_state = self.turn_90(n_state)
        n_state = self.turn_90(n_state)
        return n_state, changed

    def move_up(self, map):
        n_state = self.turn_90(map)
        n_state = self.turn_90(n_state)
        n_state = self.turn_90(n_state)
        n_state, changed = self.move_left(n_state)
        n_state = self.turn_90(n_state)
        return n_state, changed

    def turn_90(self, map):
        new_state = np.zeros((4, 4), dtype=np.int64)
        for i in range(4):
            for j in range(4):
                new_state[j][3 - i] = map[i][j]
        return new_state

    def reset(self):
        self.state = np.zeros((4, 4), dtype=np.int64)
        self.add_1_or_2()
        self.step_count = 0
        return self.state

    def render(self):
        img = np.zeros((450, 450, 3))
        for i in range(4):
            for j in range(4):
                if (self.state[i][j] != 0):
                    color = (int(self.state[i][j] * 25.5), 100, 100)
                    cv2.rectangle(img, (j * 110 + 10, i * 110 + 10), (j * 110 + 10 + 100, i * 110 + 10 + 100), color,
                                  -1)
                    cv2.putText(img, str(2 ** self.state[i][j]), (j * 110 + 45, i * 110 + 70), cv2.FONT_HERSHEY_COMPLEX,
                                1.5, (255, 255, 255))
        return img

    def eval_map(self, map):
        empty_cell_num_score = 0
        bit_score = np.exp2(np.max(map))

        for i in range(4):
            for j in range(4):
                if (map[i][j] == 0):
                    empty_cell_num_score = empty_cell_num_score + 1
        score = empty_cell_num_score + bit_score
        return score

    def step(self, action):

        if (action == 0):
            n_state, changed = self.move_left(self.state)
        if (action == 1):
            n_state, changed = self.move_down(self.state)
        if (action == 2):
            n_state, changed = self.move_right(self.state)
        if (action == 3):
            n_state, changed = self.move_up(self.state)

        reward = 0
        done = False

        if (not changed):
            reward = -10
            self.step_count = self.step_count + 1

            if (self.step_count == 2):
                done = True


        if (changed):
            reward = np.max(n_state)
            self.state = n_state
            self.add_1_or_2()
            self.step_count = 0

            if reward == 10:
                print("성공!!!!!!")
                done = True

        _, changed_1 = self.move_left(self.state)
        _, changed_2 = self.move_down(self.state)
        _, changed_3 = self.move_right(self.state)
        _, changed_4 = self.move_up(self.state)

        d_changed = changed_1 or changed_2 or changed_3 or changed_4
        done = (not d_changed) or done

        if reward != 10 and done:
            reward = -10

        return self.state, reward, done, changed

    def get_mask(self):
        _, changed_1 = self.move_left(self.state)
        _, changed_2 = self.move_down(self.state)
        _, changed_3 = self.move_right(self.state)
        _, changed_4 = self.move_up(self.state)
        return (changed_1, changed_2, changed_3, changed_4)

state_num = (4,4,1)
basic_state_num = (1,4,4,1)
action_num = 4
hidden_state = 256
learning_rate = 0.00001

discount_rate = 0.99

i = tf.keras.Input(shape=(4,4,1),dtype=tf.int64)
out = tf.one_hot(i, 10)
out = tf.keras.layers.Reshape((4,4,10))(out)
out_1 = tf.keras.layers.Conv2D(256, (4,1), activation='relu')(out)
out_1 = tf.keras.layers.Reshape((4,256))(out_1)
out_2 = tf.keras.layers.Conv2D(256, (1,4), activation='relu')(out)
out_2 = tf.keras.layers.Reshape((4,256))(out_2)
out = tf.keras.layers.Concatenate(axis=-2)([out_1,out_2])
out = tf.keras.layers.Dense(128,activation='relu')(out)
out = tf.keras.layers.Dense(64,activation='relu')(out)
out = tf.keras.layers.Flatten()(out)
out = tf.keras.layers.Dense(256,activation='relu')(out)
out = tf.keras.layers.Dense(4,activation='relu')(out)

dqn_model = tf.keras.Model(inputs=[i], outputs=[out])
dqn_model.summary()
dqn_model = tf.keras.models.load_model("hjk_1024_dqn_model.h5")
target_model = tf.keras.models.clone_model(dqn_model)
dqn_opt = tf.keras.optimizers.Adam(learning_rate)

num_episode = 30000

# 환경을 만들어 주자
env = hjk1024()

batch_size = 64
memory_size = 100000

epsilon_max = 0.9
epsilon_min = 0.1
epsilon_count = 10000

epsilon = epsilon_max
epsilon_decay = epsilon_min / epsilon_max
epsilon_decay = epsilon_decay ** (1. / float(epsilon_count))

s_list = deque(maxlen=memory_size)
a_list = deque(maxlen=memory_size)
r_list = deque(maxlen=memory_size)
n_s_list = deque(maxlen=memory_size)
d_list = deque(maxlen=memory_size)

reward_list = []

for epi in range(num_episode+1):
    d = False
    total_reward = 0
    s = env.reset()
    _s = np.reshape(s, basic_state_num)
    while not d:
        q_val = dqn_model.predict(_s)
        if (np.random.rand()) < epsilon:
            action = np.random.choice(range(4))
        else:
            mask = env.get_mask()
            q_val = q_val[0] * mask
            action = np.argmax(q_val)

        n_s, r, d, _ = env.step(action)
        _n_s = np.reshape(n_s, basic_state_num)

        if (epi % 100 == 0):
            # print("------------")
            # print("현재상태")
            # print(np.squeeze(s))
            # print("선택(Q값)")
            print(action, np.around(q_val, 3))

        s_list.append(s)
        a_list.append(action)
        r_list.append(r / 1000.)
        n_s_list.append(n_s)
        d_list.append(1 - d)

        s = n_s
        _s = _n_s
        total_reward = total_reward + r

        if len(s_list) >= batch_size * 10:
            sample = random.sample(range(len(s_list)), batch_size)
            _s_list = tf.convert_to_tensor(np.array(s_list)[sample])
            _a_list = tf.convert_to_tensor(np.array(a_list)[sample])
            _r_list = tf.convert_to_tensor(np.array(r_list, dtype='float32')[sample])
            _d_list = tf.convert_to_tensor(np.array(d_list, dtype='float32')[sample])
            _n_s_list = tf.convert_to_tensor(np.array(n_s_list)[sample])

            with tf.GradientTape() as tape:
                q = dqn_model(_s_list)
                n_q = target_model(_n_s_list)
                q = tf.gather_nd(q, tf.reshape(_a_list, (batch_size, 1)), batch_dims=1)
                td = _r_list + discount_rate * _d_list * tf.reduce_max(n_q, axis=-1)
                tde = tf.stop_gradient(td) - q
                loss = tf.math.square(tde)
                loss = tf.math.reduce_mean(loss)
            grad = tape.gradient(loss, dqn_model.trainable_variables)
            dqn_opt.apply_gradients(zip(grad, dqn_model.trainable_variables))

    if epi % 50 == 0:
        target_model.set_weights(dqn_model.get_weights())

    if (epsilon > epsilon_min):
        epsilon = epsilon * epsilon_decay

    if (epi % 1000 == 0):
        dqn_model.save("hjk_1024_dqn_model.h5")
        fcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('hjk_1024_dqn_model_{}.avi'.format(epi), fcc, 1.0, (450, 450))
        d = False
        s = env.reset()
        s = np.reshape(s, basic_state_num)
        out.write(np.uint8(env.render()))
        while not d:
            q_val = dqn_model.predict(s)
            mask = env.get_mask()
            q_val = q_val[0] * mask
            action = np.argmax(q_val)
            n_s, r, d, _ = env.step(action)
            n_s = np.reshape(n_s, basic_state_num)
            s = n_s
            out.write(np.uint8(env.render()))
        out.release()
    reward_list.append(total_reward)
    print('현재 에프소드 : {}, 현재 점수 : {}, 최고점 : {}, 100번 평균 : {}'.format(epi, total_reward, np.max(env.state),np.mean(reward_list[-100:])))
