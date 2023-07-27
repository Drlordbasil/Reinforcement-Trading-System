import datetime
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import gym
from gym import spaces
from gym.utils import seeding
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class StockTradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, symbol, start_date, end_date):
        super(StockTradingEnvironment, self).__init__()

        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

        self.df = self._get_stock_data(self.symbol, self.start_date, self.end_date)
        self.max_steps = len(self.df) - 1

        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,))

        self.current_step = None
        self.current_cash = None
        self.current_holdings = None
        self.initial_cash = 10000
        self.initial_holdings = 0
        self.transaction_cost_pct = 0.001

        self._seed()
        self.reset()

    def _get_stock_data(self, symbol, start_date, end_date):
        df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        df['Date'] = df['Date'].astype(str)
        return df

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        prev_holdings_value = self.current_holdings * self.df.loc[self.current_step - 1, 'Close']
        current_holdings_value = self.current_holdings * self.df.loc[self.current_step, 'Close']
        net_worth = self.current_cash + current_holdings_value
        portfolio_value = net_worth / self.initial_cash
        
        observation = np.array([
            self.df.loc[self.current_step, 'Open'] / self.df.loc[self.current_step, 'Close'],  # Open/Close ratio
            self.df.loc[self.current_step, 'High'] / self.df.loc[self.current_step, 'Close'],  # High/Close ratio
            self.df.loc[self.current_step, 'Low'] / self.df.loc[self.current_step, 'Close'],  # Low/Close ratio
            self.df.loc[self.current_step, 'Close'] / self.df.loc[self.current_step - 1, 'Close'],  # Close/prev_close ratio
            prev_holdings_value / self.initial_cash,  # previous holdings value
            portfolio_value  # current portfolio value
        ])

        return observation

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, 'Close']

        if action == 0:  # Hold
            pass
        elif action == 1 and self.current_cash > current_price:  # Buy
            shares_to_buy = int(self.current_cash / current_price)
            transaction_cost = shares_to_buy * current_price * self.transaction_cost_pct
            self.current_cash -= (shares_to_buy * current_price) + transaction_cost
            self.current_holdings += shares_to_buy
        elif action == 2 and self.current_holdings > 0:  # Sell
            shares_to_sell = self.current_holdings
            transaction_cost = shares_to_sell * current_price * self.transaction_cost_pct
            self.current_cash += (shares_to_sell * current_price) - transaction_cost
            self.current_holdings -= shares_to_sell

    def _get_reward(self):
        prev_portfolio_value = (self.current_cash + (self.current_holdings * self.df.loc[self.current_step - 1, 'Close'])) / self.initial_cash
        current_portfolio_value = (self.current_cash + (self.current_holdings * self.df.loc[self.current_step, 'Close'])) / self.initial_cash
        reward = current_portfolio_value - prev_portfolio_value
        return reward

    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        if self.current_step >= self.max_steps:
            self.current_step = 0

        observation = self._get_observation()
        
        reward = self._get_reward()

        done = False

        return observation, reward, done, {}

    def reset(self):
        self.current_step = 1
        self.current_cash = self.initial_cash
        self.current_holdings = self.initial_holdings
        return self._get_observation()

    def render(self, mode='human'):
        pass


def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))
    model.compile(optimizer=Adam(), loss='mse')
    return model


# Hyperparameters
symbol = 'AAPL'
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2021, 12, 31)
episodes = 1000
batch_size = 32

env = StockTradingEnvironment(symbol, start_date, end_date)

input_shape = env.observation_space.shape
output_shape = env.action_space.n

model = build_model(input_shape, output_shape)

for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, input_shape[0]])
    
    total_reward = 0
    done = False

    while not done:
        action = np.argmax(model.predict(state)[0])
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, input_shape[0]])
        total_reward += reward

        state = next_state

    print(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward}")

    replay_buffer.append((state, action, reward, next_state, done))

    if len(replay_buffer) >= batch_size:
        batch = random.sample(replay_buffer, batch_size)
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])

        targets = rewards + gamma * np.max(model.predict(next_states), axis=1) * (1 - dones)
        target_vecs = model.predict(states)
        target_vecs[range(batch_size), actions] = targets

        model.fit(states, target_vecs, epochs=1, verbose=0)

env.close()
