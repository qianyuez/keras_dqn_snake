from env import Env
from dqn import DeepQNetwork
import argparse


def start():
    for e in range(EPOCHS):
        distance = 0
        score = 0
        time = 0
        state = env.reset()
        while True:
            env.render(frame_time)
            if distance > MAX_STEPS:
                action = model.random_action()
            else:
                action = model.choose_action(state)
            next_state, reward, done = env.step(action)
            score += reward
            if mode == 'train':
                model.store_transition(state, action, reward, next_state)
                if len(model.memory) >= model.memory_size:
                    model.learn()
            state = next_state
            time += 1
            if reward == 0:
                distance += 1
            else:
                distance = 0

            if done:
                print("episode: {}/{}, score: {}, steps: {}, epsilon: {}".format(e, EPOCHS, score + 1, time, model.epsilon))
                break

        if e % 100 == 0:
            model.save_model(model_path)


if __name__ == '__main__':
    EPOCHS = 50000
    MAX_STEPS = 100
    BATCH_SIZE = 64
    STATE_SHAPE = (8, 8, 1)
    ACTION_SIZE = 3
    MEMORY_SIZE = 3000
    MODEL_PATH = './model/dqn_model.h5'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', type=str, default='train',
        help='train or test model, default "train"'
    )
    parser.add_argument(
        '--path', type=str, default=MODEL_PATH,
        help='path to model weight file, default {}'.format(MODEL_PATH)
    )
    ARGS = parser.parse_args()
    mode = ARGS.mode
    model_path = ARGS.path

    env = Env(STATE_SHAPE, ACTION_SIZE)
    if mode == 'train':
        frame_time = 1
        model = DeepQNetwork(n_actions=ACTION_SIZE,
                             n_states=STATE_SHAPE,
                             memory_size=MEMORY_SIZE,
                             batch_size=BATCH_SIZE,
                             e_greedy=1,
                             double_dqn=True,
                             e_greedy_increment=None)

    if mode == 'test':
        frame_time = 50
        model = DeepQNetwork(n_actions=ACTION_SIZE,
                             n_states=STATE_SHAPE,
                             memory_size=MEMORY_SIZE,
                             batch_size=BATCH_SIZE,
                             e_greedy=1.0,
                             double_dqn=True,
                             e_greedy_increment=None)
        model.load_model(model_path)

    start()
