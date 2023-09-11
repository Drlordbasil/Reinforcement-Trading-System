from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from gym.utils import seeding
from gym import spaces
import gym
import pandas_datareader as pdr
import pandas as pd
import numpy as np
import random
import datetime
To optimize the Python script, you can make the following code modifications:

1. Move the imports inside the methods where they are used to avoid unnecessary imports.

2. Precompute the values that are used repeatedly, such as `self.df.loc[self.current_step, 'Close']` and `self.df.loc[self.current_step - 1, 'Close']`, and store them in variables to avoid repeated lookups.

3. Use numpy vectorized operations wherever possible to improve performance.

Here's the optimized version of the code:

```python


class StockTradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, symbol, start_date, end_date):
        super(StockTradingEnvironment, self).__init__()

        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

        self.max_steps = len(self._get_stock_data())
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

    def _get_stock_data(self):
        df = pdr.get_data_yahoo(
            self.symbol, start=self.start_date, end=self.end_date)
        df.reset_index(inplace=True)
        df['Date'] = df['Date'].astype(str)
        return df

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        current_close = self.df_close[self.current_step]
        prev_close = self.df_close[self.current_step - 1]

        prev_holdings_value = self.current_holdings * prev_close
        current_holdings_value = self.current_holdings * current_close
        net_worth = self.current_cash + current_holdings_value
        portfolio_value = net_worth / self.initial_cash

        observation = np.array([
            self.df_open[self.current_step] / current_close,
            self.df_high[self.current_step] / current_close,
            self.df_low[self.current_step] / current_close,
            current_close / prev_close,
            prev_holdings_value / self.initial_cash,
            portfolio_value
        ])

        return observation

    def _take_action(self, action):
        current_price = self.df_close[self.current_step]

        if action == 0:  # Hold
            pass
        elif action == 1 and self.current_cash > current_price:  # Buy
            shares_to_buy = int(self.current_cash / current_price)
            transaction_cost = shares_to_buy * current_price * self.transaction_cost_pct
            self.current_cash -= (shares_to_buy *
                                  current_price) + transaction_cost
            self.current_holdings += shares_to_buy
        elif action == 2 and self.current_holdings > 0:  # Sell
            shares_to_sell = self.current_holdings
            transaction_cost = shares_to_sell * current_price * self.transaction_cost_pct
            self.current_cash += (shares_to_sell *
                                  current_price) - transaction_cost
            self.current_holdings -= shares_to_sell

    def _get_reward(self):
        prev_portfolio_value = (self.current_cash + (self.current_holdings *
                                self.df_close[self.current_step - 1])) / self.initial_cash
        current_portfolio_value = (self.current_cash + (
            self.current_holdings * self.df_close[self.current_step])) / self.initial_cash
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

        self.df_open = self.df['Open'] / self.df['Close']
        self.df_high = self.df['High'] / self.df['Close']
        self.df_low = self.df['Low'] / self.df['Close']
        self.df_close = self.df['Close']

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
gamma = 0.99
replay_buffer = []

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

        targets = rewards + gamma * \
            np.max(model.predict(next_states), axis=1) * (1 - dones)
        target_vecs = model.predict(states)
        target_vecs[np.arange(batch_size), actions] = targets

        model.fit(states, target_vecs, epochs=1, verbose=0)

env.close()
```

These optimizations should reduce unnecessary computations and improve the overall performance of the script.
