# libraries
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# constants
MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

# agent class
class Agent:

    # constructor
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft
        # TODO: model, trainer
        self.model = Linear_QNet(15, 256, 256, 3)
        # comment the next line out if you are going to change the model in any way or form
        self.model.load_state_dict(torch.load('./model/model.pth'))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # right snake training
        self.right_move = 0

    def get_state(self, game):
        head = game.snake[0]

        # left, right, up, down of snake head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # angle direction
        point_lu = Point(head.x - 20, head.y - 20)
        point_ld = Point(head.x - 20, head.y + 20)
        point_ru = Point(head.x + 20, head.y - 20)
        point_rd = Point(head.x + 20, head.y + 20)

        # what direction is it going
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # 15 states
        state = [
            # Danger straight state
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right state
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left state
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Danger bottom right
            (dir_r and game.is_collision(point_ld)) or
            (dir_d and game.is_collision(point_lu)) or
            (dir_l and game.is_collision(point_ru)) or
            (dir_u and game.is_collision(point_rd)),

            # Danger bottom left
            (dir_r and game.is_collision(point_lu)) or
            (dir_d and game.is_collision(point_ru)) or
            (dir_l and game.is_collision(point_rd)) or
            (dir_u and game.is_collision(point_ld)),

            # Danger top right
            (dir_r and game.is_collision(point_rd)) or
            (dir_d and game.is_collision(point_ld)) or
            (dir_l and game.is_collision(point_lu)) or
            (dir_u and game.is_collision(point_ru)),

            # Danger top left
            (dir_r and game.is_collision(point_ru)) or
            (dir_d and game.is_collision(point_rd)) or
            (dir_l and game.is_collision(point_ld)) or
            (dir_u and game.is_collision(point_lu)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff between exploration and exploitation
        # self.epsilon = 80 - self.n_games
        self.epsilon = 20 - self.n_games
        final_move = [0,0,0]

        if random.randint(0,250) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1

            # # can only do right turn and straight
            # move = random.randint(0, 1)
            # if move == 1:
            #     self.right_move += 1
            # else:
            #     self.right_move = 0

            # if self.right_move == 2:
            #     move = 0
            #     self.right_move = 0

            # final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    # training loop
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move based on the old state
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory of the agent
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train the long memory (experience replay), plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            f = open("./model/high.txt", "r")
            all_time_high = int(f.read())
            f.close()

            if score > record:
                record = score
            if score > all_time_high:
                agent.model.save()
                f = open("model/high.txt", "w")
                f.write(str(score))
                f.close()

            print("Game", agent.n_games, "Score", score, "Record", record)
            # print(agent.model.state_dict())

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
