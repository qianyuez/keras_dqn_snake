import numpy as np
import cv2


class Env:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.snake = []
        self.actions = ['forward', 'left', 'right']
        self.box_size = 35
        self.direction = (1, 0)

    # set random position for new food
    def set_food_pos(self):
        while True:
            x = np.random.randint(self.state_shape[0])
            y = np.random.randint(self.state_shape[1])
            food_pos = (x, y)
            if food_pos not in self.snake:
                self.food_pos = food_pos
                break

    def reset(self):
        self.snake = [(5, 3), (4, 3), (3, 3), (2, 3), (1, 3)]
        self.direction = (1, 0)
        self.set_food_pos()
        return self.get_state()

    def move(self):
        self.snake.pop()
        point = self.snake[0]
        new_point = (point[0] + self.direction[0], point[1] + self.direction[1])
        self.snake.insert(0, new_point)

    def turn(self, action):
        turn_action = self.actions[action]
        if turn_action == 'forward':
            pass
        elif turn_action == 'left':
            self.direction = self._left_direction(self.direction)
        elif turn_action == 'right':
            self.direction = self._right_direction(self.direction)

    def _left_direction(self, direction):
        return (-direction[1], direction[0])

    def _right_direction(self, direction):
        return (direction[1], -direction[0])

    def eat(self):
        last_point = self.snake[-1]
        second_last_point = self.snake[-2]
        direction = (last_point[0] - second_last_point[0], last_point[1] - second_last_point[1])
        x = last_point[0] + direction[0]
        y = last_point[1] + direction[1]
        self.snake.append((x, y))

    def _point_in_range(self, point):
        x = point[0]
        new_x = np.clip(x, 0, self.state_shape[0] - 1)
        y = point[1]
        new_y = np.clip(y, 0, self.state_shape[1] - 1)
        in_range = (x == new_x) and (y == new_y)
        new_point = (new_x, new_y)
        return in_range, new_point

    # get current environment state
    def get_state(self):
        state = np.zeros(self.state_shape)
        for point in self.snake[1:]:
            _, point = self._point_in_range(point)
            state[point[0], point[1]] = [1.0]
        state[self.food_pos[0], self.food_pos[1]] = [-1.0]
        _, point = self._point_in_range(self.snake[0])
        state[point[0], point[1]] += 0.5
        return state

    def step(self, action):
        self.turn(action)
        self.move()
        reward = 0
        done = 0
        in_range, _ = self._point_in_range(self.snake[0])
        if self.food_pos == self.snake[0]:
            self.eat()
            self.set_food_pos()
            reward = 1
        if (self.snake[0] in self.snake[1:]) or not in_range:
            reward = -1
            done = 1

        state = self.get_state()
        return state, reward, done

    def render(self, frame_time=1):
        img = np.zeros((self.state_shape[0] * self.box_size, self.state_shape[1] * self.box_size, 3), np.uint8)
        img.fill(255)
        point = self.snake[0]

        cv2.rectangle(img, (point[0] * self.box_size, point[1] * self.box_size),
                      ((point[0] + 1) * self.box_size, (point[1] + 1) * self.box_size), (90, 90, 90), -1)
        for index in range(1, len(self.snake)):
            point = self.snake[index]
            cv2.rectangle(img, (point[0] * self.box_size, point[1] * self.box_size),
                          ((point[0] + 1) * self.box_size, (point[1] + 1) * self.box_size), (0, 0, 0), -1)
        cv2.rectangle(img, (self.food_pos[0] * self.box_size, self.food_pos[1] * self.box_size),
                      ((self.food_pos[0] + 1) * self.box_size, (self.food_pos[1] + 1) * self.box_size), (150, 150, 150),
                      -1)
        cv2.imshow('image', img)
        cv2.waitKey(frame_time)