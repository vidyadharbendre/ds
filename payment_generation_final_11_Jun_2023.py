#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from collections import deque

from keras.models import Sequential
from keras.layers import Dense


# In[2]:


aggregators = ['Razorpay', 'CCAvenue', 'Cashfree', 'Paytm', 'Mobikwik']
payment_types = ['UPI_RuPay_Debit', 'NetBanking', 'Wallet', 'Domestic_Debit_Credit_Cards', 
                 'EMI_Credit', 'International', 'International_Diners_Amex']

matrix = [[0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03],
          [0, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04],
          [0, 0.0175, 0.0190, 0.0175, 0.0150, 0.035, 0.0295],
          [0, 0.0199, 0.0199, 0.0199, 0.0299, 0.0299, 0.0299],
          [0, 0.0190, 0.0190, 0.0190, 0.0225, 0.0290, 0.0290]
         ]


# In[3]:


updated_keys = []

for i in aggregators:
    for j in payment_types:
        key = (i, j)
        updated_keys.append(key)

len(updated_keys)


# In[4]:


data = {}

for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        key = (i, j)
        value = matrix[i][j]
        data[key] = value

print(data)


# In[5]:


values = data.values()
values


# In[6]:


tran_fee_dict = dict(zip(updated_keys, values))
tran_fee_dict


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Define the range of alpha and beta values
alphas = [5, 4, 2, 1, 3]
betas = [2, 3, 4, 5, 1]

# Create a grid of plots
fig, axes = plt.subplots(len(alphas), len(betas), figsize=(12, 8), sharex=True, sharey=True)

# Generate data and plot the beta distributions
for i, alpha in enumerate(alphas):
    for j, beta_val in enumerate(betas):
        ax = axes[i][j]
        x = np.linspace(0, 1, 100)
        y = beta.pdf(x, alpha, beta_val)
        ax.plot(x, y)
        ax.set_title(f"Alpha = {alpha}, Beta = {beta_val}")

# Add labels and adjust layout
fig.suptitle("Beta Distribution for Different Alpha and Beta Values")
plt.tight_layout()
plt.show()



# In[8]:


import random
random.seed(42)

def generate_payment_data(num_samples):
    payment_data = []
    aggregators = ['Cashfree']

    payment_types = ['UPI_RuPay_Debit', 'NetBanking', 'Wallet', 'Domestic_Debit_Credit_Cards', \
                     'EMI_Credit', 'International', 'International_Diners_Amex']
    
    alpha_beta_dict = {
        'UPI_RuPay_Debit': (5, 2),
        'NetBanking': (4, 1),
        'Wallet': (3, 2),
        'Domestic_Debit_Credit_Cards': (4, 1),
        'EMI_Credit': (5, 2),
        'International': (1, 2),
        'International_Diners_Amex': (3, 5)
    }
    
    for i in range(num_samples):
        payment = {}
        payment['amount'] = round(random.lognormvariate(10, 1.1))    
        payment['type'] = random.choice(payment_types)
        payment['aggregator'] = random.choice(aggregators)
        alpha, beta = alpha_beta_dict[payment['type']]
        payment['success_rate'] = round(random.betavariate(alpha, beta), 2)
        payment['processing_time'] = round(random.uniform(0, 10), 2)
        payment['transaction_charge'] = tran_fee_dict[payment['aggregator'], payment['type']]
        payment['merchant_id'] = random.randint(1, 11)
        payment['user_id'] = random.randint(1, num_samples)
        payment['status'] = 1 if payment['success_rate'] > 0.60 else 0
        payment_data.append(payment)

    payment_df = pd.DataFrame(payment_data)
    payment_df = payment_df[['amount', 'type', 'aggregator', 'success_rate', 'processing_time', 'transaction_charge', 'merchant_id', 'user_id', 'status']]
    return payment_df


# In[9]:


payment_data = generate_payment_data(1000)
print(payment_data.status.value_counts(normalize=True))
payment_data.head()


# In[10]:


df = payment_data


# In[11]:


import matplotlib.pyplot as plt
from scipy.stats import beta

payment_types = ['UPI_RuPay_Debit', 'NetBanking', 'Wallet', 'Domestic_Debit_Credit_Cards', \
                 'EMI_Credit', 'International', 'International_Diners_Amex']

alpha_beta_dict = {
    'UPI_RuPay_Debit': (5, 2),
    'NetBanking': (4, 1),
    'Wallet': (3, 2),
    'Domestic_Debit_Credit_Cards': (4, 1),
    'EMI_Credit': (5, 2),
    'International': (1, 2),
    'International_Diners_Amex': (3, 5)
}

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16,8))
for i, payment_type in enumerate(payment_types):
    alpha, betaa = alpha_beta_dict[payment_type]
    row = i // 4
    col = i % 4
    x = np.linspace(0, 1, 100)
    axs[row, col].plot(x, beta.pdf(x, alpha, betaa), 'r-', lw=5, alpha=0.6, label='beta pdf')
    axs[row, col].set_title(payment_type)
    axs[row, col].legend(loc='best', frameon=False)

plt.tight_layout()
plt.show()

This code will generate a 2x4 grid of subplots, where each subplot will show the beta distribution for a particular payment type. The alpha_beta_dict dictionary specifies the alpha and beta values for each payment type. The scipy.stats.beta.pdf() function is used to calculate the probability density function of the beta distribution.
# In[12]:


sns.pairplot(df.drop(['amount', 'merchant_id', 'user_id'], axis = 1), diag_kind='kde')
plt.show()


# In[13]:


df.drop(['amount', 'merchant_id', 'user_id'], axis = 1).describe()


# In[14]:


df


# In[15]:


df['log_amount'] = df.amount.apply(lambda x: np.log(x))


# In[16]:


# Define the available payment methods and their corresponding action indices
actions = ['UPI_RuPay_Debit', 'NetBanking', 'Wallet', 'Domestic_Debit_Credit_Cards', 
                 'EMI_Credit', 'International', 'International_Diners_Amex']
action_indices = {action: i for i, action in enumerate(actions)}
action_indices


# In[17]:


df.head(3)


# In[18]:


df.rename(columns={'success': 'status'}, inplace=True)


# In[20]:


sns.pairplot(df.drop(['amount', 'merchant_id', 'user_id'], axis = 1), diag_kind='kde', hue='status')
plt.show()


# In[21]:


# set plot size
plt.rcParams["figure.figsize"] = [15,10]

# plot density plot using plot()
# kind: set the type of plot
# subplots: indicates whether to plot subplot for each variable or a single line plot
# layout: specify the arrangement of the subplots
# sharex: indicates whether to have the same scale on x-axis of all subplots
df.drop(['amount', 'merchant_id', 'user_id'], axis = 1).plot(kind = 'density', subplots = True, layout = (3,3), sharex = False)

# show plot
plt.show()


# In[22]:


# calculate the correlation between the variables
corr = df.drop(['amount', 'merchant_id', 'user_id'], axis = 1).corr()

# print correaltion matrix
corr


# In[23]:


df.head()


# In[ ]:





# In[24]:


# set the plot size
fig,ax = plt.subplots(figsize=(7, 5))

# plot a heatmap for the correlation matrix
# annot: print values in each cell
# linewidths: specify width of the line specifying the plot
# vmin: minimum value of the variable
# vmax: maximum value of the variable
# cmap: colour code of the plot
# fmt: set the decimal place of annot
sns.heatmap(corr, annot = True, linewidths = 0.05, vmin = -1 , vmax = 1, cmap = "coolwarm" , fmt = '10.4f')

# display the plot
plt.show()


# In[25]:


sns.pairplot(data=df.drop(['amount', 'merchant_id', 'user_id'], axis = 1), kind='reg', diag_kind='kde')
plt.show()


# In[26]:


df.head(3)


# In[27]:


data = df.drop(['amount', 'merchant_id', 'user_id'], axis = 1)
data


# In[28]:


df.head(3)


# In[29]:


df_num = df.select_dtypes(include = [np.number])
df_cat = df.select_dtypes(exclude = [np.number])


# In[30]:


df_cat = df_cat.drop('aggregator', axis = 1)
df_num = df_num.drop(['amount', 'merchant_id', 'user_id'], axis = 1)


# In[31]:


df_num


# In[32]:


# Define the available payment methods and their corresponding action indices
actions = ['UPI_RuPay_Debit', 'NetBanking', 'Wallet', 'Domestic_Debit_Credit_Cards', 
                 'EMI_Credit', 'International', 'International_Diners_Amex']
action_indices = {action: i for i, action in enumerate(actions)}
action_indices


# In[33]:


df_cat_encoded = df_cat.replace({'type' : { 'UPI_RuPay_Debit': 0,
 'NetBanking': 1,
 'Wallet': 2,
 'Domestic_Debit_Credit_Cards': 3,
 'EMI_Credit': 4,
 'International': 5,
 'International_Diners_Amex': 6 }, 'status': {'success': 1, 'failure': 0}})


# In[34]:


df_cat_encoded.head()


# In[35]:


ms = MinMaxScaler()
scaled = ms.fit_transform(df_num)
df_num_scaled = pd.DataFrame(scaled, columns=df_num.columns)


# In[36]:


y = df_cat_encoded.type.values


# In[37]:


X_y = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
X_y.head(3)


# In[38]:


X_y.shape


# In[39]:


# Plot the distribution for each payment type
payment_types = X_y['type'].unique()
for payment_type in payment_types:
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].set_ylabel('Frequency')
    fig.suptitle(f'Payment Type - {payment_type}')
    data = X_y[X_y['type'] == payment_type]
    for i, col in enumerate(X_y.columns[:-1]):
        axs[i].hist(data[col], bins=20)
        axs[i].set_xlabel(col)
    plt.show()


# In[141]:


# concat the dummy variables with numeric features to create a dataframe of all independent variables
# 'axis=1' concats the dataframes along columns 
X = pd.concat([df_num_scaled, df_cat_encoded.drop('type', axis=1)], axis = 1)

# display first five observations
X.head()


# In[40]:


X_y.to_csv('jun_11_1K_X_y_03.csv')


# In[142]:


X_y.to_csv('jun_04_1K_X_y_02.csv')


# In[101]:


#X_y.to_csv('jun_04_10K_X_y_01.csv')


# In[102]:


#X_y.to_csv('jun_01_100K_X_y_01.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


df = pd.read_csv('june_7_final_data.csv')
df = df.drop('Unnamed: 0', axis=1)
df.head(3)


# In[10]:


df.info()


# In[12]:


sns.pairplot(df, diag_kind='kde', hue='status')
plt.show()


# In[19]:


for i, col in enumerate(df.drop('type', axis=1).columns):
    print(i, col)


# In[20]:


# Plot the distribution for each payment type
payment_types = df['type'].unique()
for payment_type in payment_types:
    fig, axs = plt.subplots(1, 6, figsize=(20, 4))
    axs[0].set_ylabel('Frequency')
    fig.suptitle(f'Payment Type - {payment_type}')
    data = df[df['type'] == payment_type]
    for i, col in enumerate(df.drop('type', axis=1).columns):
        axs[i].hist(data[col], bins=20)
        axs[i].set_xlabel(col)
    plt.show()


# In[26]:


# calculate the correlation between the variables
corr = df.drop('type', axis=1).corr()

# set the plot size
fig,ax = plt.subplots(figsize=(7, 5))

# plot a heatmap for the correlation matrix
# annot: print values in each cell
# linewidths: specify width of the line specifying the plot
# vmin: minimum value of the variable
# vmax: maximum value of the variable
# cmap: colour code of the plot
# fmt: set the decimal place of annot
sns.heatmap(corr, annot = True, linewidths = 0.08, vmin = -1 , vmax = 1, cmap = "coolwarm" , fmt = '4.4f')

# display the plot
plt.show()


# In[28]:


# calculate the correlation between the variables
corr = df.drop(['type', 'success_rate', 'status'], axis=1).corr()

# set the plot size
fig,ax = plt.subplots(figsize=(7, 5))

# plot a heatmap for the correlation matrix
# annot: print values in each cell
# linewidths: specify width of the line specifying the plot
# vmin: minimum value of the variable
# vmax: maximum value of the variable
# cmap: colour code of the plot
# fmt: set the decimal place of annot
sns.heatmap(corr, annot = True, linewidths = 0.04, vmin = -1 , vmax = 1, cmap = "coolwarm" , fmt = '4.4f')

# display the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[53]:


X_y.shape


# In[56]:


X_y.info()


# In[57]:


# X_y.to_csv('may_15_1K_X_y_01.csv')


# In[58]:


X_y.head()


# In[59]:


status_counts = X_y.groupby(['type', 'status']).size().unstack(fill_value=0)
print(status_counts)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[100]:


X_y.status = 1.0


# In[44]:


X.to_csv('may_15_1K_01.csv')


# In[47]:


X = pd.read_csv('may_15_1K_01.csv' )


# In[45]:


X = pd.read_csv('may_15_1K_01.csv')


# In[46]:


X.columns


# In[47]:


X = X.drop('Unnamed: 0', axis=1)
X.columns


# In[48]:


# Define the available payment methods and their corresponding action indices
actions = ['UPI_RuPay_Debit', 'NetBanking', 'Wallet', 'Domestic_Debit_Credit_Cards', 
                 'EMI_Credit', 'International', 'International_Diners_Amex']
action_indices = {action: i for i, action in enumerate(actions)}
action_indices

In this modified reward function, we calculate the success rate reward, processing time reward, and transaction charge reward in the same way as before. However, instead of subtracting them, we subtract the sum of these rewards from the success rate reward. This means that we are rewarding the agent for maximizing the success rate and penalizing it for increasing the processing time and transaction charge. You can modify this function further based on your specific requirements.
# In[49]:


import gym
import numpy as np
import matplotlib.pyplot as plt

# Define the payment environment
class PaymentEnv(gym.Env):
    def __init__(self, X):
        super().__init__()  
        self.X = X
        self.num_states = len(X)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(7)
        self.current_index = 0
        self.current_state = self.X.iloc[self.current_index][['success_rate', 'processing_time', 'transaction_charge', 'log_amount', 'status']].values
        self.reward_range = (-float('inf'), float('inf'))
        self.seed(42)
        self.current_hash = 0
        self.state = []
  
        # Initialize parameters for tracking before and after training
        self.success_rate_before = []
        self.processing_time_before = []
        self.transaction_charge_before = []
        self.average_reward_before = []
        self.total_reward_before = []
        
        self.success_rate_after = []
        self.processing_time_after = []
        self.transaction_charge_after = []
        self.average_reward_after = []
        self.total_reward_after = []

        self.reset()
        
    def len(self, X):
        return len(X)
    def get_state_index(self, state):
        self.current_index = self.X[(self.X['success_rate'] == state[0]) & (self.X['processing_time'] == state[1])\
                                    & (self.X['transaction_charge'] == state[2])].index.values
        self.current_index = self.current_index[0]
        return self.current_index

    def reset(self):
        self.current_index = np.random.randint(low=0, high=self.num_states)
        self.current_state = self.X.iloc[self.current_index][['success_rate', 'processing_time', 'transaction_charge', 'log_amount', 'status', ]].values
#        self.state_index = self.get_state_index(self.current_state)
#        print(self.current_state, self.current_index)
        return self.current_state    

    def step(self, action):
        # Get current state
        current_state = self.X.iloc[self.current_index][['success_rate', 'processing_time', 'transaction_charge', 'log_amount', 'status']].values
#        print('current_state', current_state)
        # Apply the action to get the next state
        next_index = (self.current_index + action) % len(self.X)
#        print('next_index', next_index)
        next_state = self.X.iloc[next_index][['success_rate', 'processing_time', 'transaction_charge', 'log_amount', 'status']].values
#        print('next_state', next_state)
        # Compute the new reward based on the current and next states
#        print('success_rate', next_state[0], current_state[0])
        success_rate_reward = next_state[0] - current_state[0]
#        print('success_rate_reward', next_state[0], current_state[0])
        processing_time_reward = current_state[1] - next_state[1]
        transaction_charge_reward = current_state[2] - next_state[2]
        new_reward = success_rate_reward - processing_time_reward - transaction_charge_reward



        # Check if we've reached the end of the payment data
        self.done = (self.current_index >= len(self.X) - 1) or (self.current_state[-1] == 1.0)

        # Update the current index and state
        self.current_index = next_index
        self.current_state = next_state
        
        return self.current_state, new_reward, self.done, {}


    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]


# In[50]:


import gym
import numpy as np
import matplotlib.pyplot as plt

# Define the payment environment
class PaymentEnv(gym.Env):
    def __init__(self, X):
        super().__init__()  
        self.X = X
        self.num_states = len(X)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(7)
        self.current_index = 0
        self.current_state = self.X.iloc[self.current_index][['success_rate', 'processing_time', 'transaction_charge', 'log_amount', 'status']].values
        self.reward_range = (-float('inf'), float('inf'))
        self.seed(42)
        self.current_hash = 0
        self.state = []
  
        # Initialize parameters for tracking before and after training
        self.success_rate_before = [0]
        self.processing_time_before = [0]
        self.transaction_charge_before = [0]
        self.average_reward_before = [0]
        self.total_reward_before = [0]
        
        self.success_rate_after = [0]
        self.processing_time_after = [0]
        self.transaction_charge_after = [0]
        self.average_reward_after = [0]
        self.total_reward_after = [0]

        self.reset()
        
    def len(self, X):
        return len(X)
    def get_state_index(self, state):
        self.current_index = self.X[(self.X['success_rate'] == state[0]) & (self.X['processing_time'] == state[1])\
                                    & (self.X['transaction_charge'] == state[2])].index.values
        self.current_index = self.current_index[0]
        return self.current_index

    def reset(self):
        self.current_index = np.random.randint(low=0, high=self.num_states)
        self.current_state = self.X.iloc[self.current_index][['success_rate', 'processing_time', 'transaction_charge', 'log_amount', 'status', ]].values
        self.state_index = self.get_state_index(self.current_state)
#        print(self.current_state, self.current_index)
        return self.current_state    

    def step(self, action):
        # Get current state
        current_state = self.X.iloc[self.current_index][['success_rate', 'processing_time', 'transaction_charge', 'log_amount', 'status']].values
#        print('1 - current_state', current_state)
        # Apply the action to get the next state
        next_index = (self.current_index + action) % len(self.X)
#        print('2 - next_index', next_index, 'action', action, 'len-self X', len(self.X))
        next_state = self.X.iloc[next_index][['success_rate', 'processing_time', 'transaction_charge', 'log_amount', 'status']].values
#        print('3 next_state', next_state)
#        print('success_rate-processing_time-transaction_charge-log_amount-status')
        
        # Compute the new reward based on the current and next states
#        print('4 next state and current state', next_state[0], current_state[0])
        success_rate_reward = next_state[0] - current_state[0]
#        print('5 success_rate_reward', success_rate_reward)
        processing_time_reward = current_state[1] - next_state[1]
#        print('6 processing_time_reward', processing_time_reward)
        transaction_charge_reward = current_state[2] - next_state[2]
#        print('7 transaction_charge_reward', transaction_charge_reward)
#        print('success_rate_reward - processing_time_reward - transaction_charge_reward')
        new_reward = success_rate_reward - processing_time_reward - transaction_charge_reward
#        print('8 new_reward', new_reward)


        # Update parameters for tracking before training
        self.success_rate_before.append(current_state[0])
        self.processing_time_before.append(current_state[1])
        self.transaction_charge_before.append(current_state[2])
        self.total_reward_before.append(new_reward)           
        self.average_reward_before.append(sum(self.total_reward_before) / (len(self.total_reward_before) - 1))
        
#        print('total_reward_before', self.total_reward_before)
        

        
        # Update the current index and state
#        print('before update', self.current_state[-1])
        self.current_index = next_index
        self.current_state = next_state    
#        print('after update', self.current_state[-1])

        # Check if we've reached the end of the payment data
        self.done = (self.current_index >= len(self.X) - 1) or (self.current_state[-1] == 1.0)
#        print('self.done', self.done)
#         # Update parameters for tracking after training
        if self.done:
            self.success_rate_after.append(next_state[0])
            self.processing_time_after.append(next_state[1])
            self.transaction_charge_after.append(next_state[2])
            self.total_reward_after.append(new_reward)
            self.average_reward_after.append(sum(self.total_reward_after) / (len(self.total_reward_after) - 1))     

#        print('total_reward_after', self.total_reward_after)
 
        return self.current_state, new_reward, self.done, {}


    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]


# In[51]:


env = PaymentEnv(X)


# In[52]:


env.success_rate_after


# In[53]:


env.reset()


# In[54]:


import gym
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import Adam
import random
import matplotlib.pyplot as plt


# In[55]:


import gym
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import Adam
import random
import matplotlib.pyplot as plt


class DuelingDQN:
    def __init__(self, env, state_space, action_space):
        self.env = env
        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.q_network = self.build_q_network()
        self.target_network = self.build_q_network()
        self.update_target_network()
        self.total_reward = 0
        self.replay_buffer = deque(maxlen=2000)  # define replay_buffer here
        # Initialize the target network weights to match the Q-network weights
        self.target_network.set_weights(self.q_network.get_weights())
 
    def build_q_network(self):
        input_layer = Input(shape=(self.state_space,))
        x = Dense(512, activation='relu')(input_layer)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        value_stream = Dense(1)(x)
        advantage_stream = Dense(self.action_space)(x)

        output_layer = Lambda(lambda x: x[0] + x[1] - K.mean(x[1], axis=1, keepdims=True))([value_stream, advantage_stream])
        
#        output_layer = Lambda(lambda x: x[0] + x[1] - K.mean(x[1], axis=1, keepdims=True))([value_stream, advantage_stream])
        model = Model(inputs=[input_layer], outputs=[output_layer])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])    
    
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([s[0] for s in minibatch])
        actions = np.array([s[1] for s in minibatch])
        rewards = np.array([s[2] for s in minibatch])
        next_states = np.array([s[3] for s in minibatch])
        done = np.array([s[4] for s in minibatch])

        # Store the experience in the replay buffer
        for i in range(self.batch_size):
            self.replay_buffer.append((states[i], actions[i], rewards[i], next_states[i], done[i]))

        target_q_values = self.q_network.predict(states)
        target_q_values_next = self.target_network.predict(next_states)
        # Use the Q-values from the target network to calculate the target Q-values
        target_actions = np.argmax(self.q_network.predict(next_states), axis=1)
        target_q_values_next = target_q_values_next[np.arange(self.batch_size), target_actions]
        # Update the Q-values using the Bellman equation
        target_q_values[np.arange(self.batch_size), actions] = rewards + (1 - done) * self.gamma * target_q_values_next
        # Train the Q-network
        self.q_network.fit(states, target_q_values, epochs=1, verbose=0)
        # Decay the epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_epsilon(self, episode):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay ** episode)
            
            
    def episode_done_callback(self, episode, total_reward):
        print(f"Episode {episode} finished with total reward of {total_reward:.2f}")            
            
    def train(self, episodes):
        num_episodes = episodes
        # Initialize episode rewards and success rates
        episode_rewards = [0]
        episode_success_rates = [0]
        
        for episode in range(1, num_episodes+1):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            episode_reward = 0
#            total_reward = 0
            success_count = 0
            t = 0
            done = False 
#             print('self.env.current_index', self.env.current_index)
#             print('self.env.current_index-success_rate', self.env.X.iloc[self.env.current_index]['success_rate'])
            # Initialize parameters for tracking before training
#             self.env.success_rate_before.append(self.env.X.iloc[self.env.current_index]['success_rate'])
#             self.env.processing_time_before.append(self.env.X.iloc[self.env.current_index]['processing_time'])
#             self.env.transaction_charge_before.append(self.env.X.iloc[self.env.current_index]['transaction_charge'])

            
#             success_rate_before = self.env.success_rate_before[-1]
#             processing_time_before = self.env.processing_time_before[-1]
#             transaction_charge_before = self.env.transaction_charge_before[-1]
#             print(self.env.total_reward_before[-1])
#             total_reward_before = self.env.total_reward_before[-1]             
#             self.env.total_reward_before.append(total_reward_before + self.total_reward)
#             self.env.average_reward_before.append(total_reward_before / (num_episodes + 1))            
  
            
#            print('done', done)
            while not done:                
                # Choose an action using epsilon-greedy exploration
                action = self.choose_action(state)
                # Take the chosen action and observe the next state and reward
                next_state, reward, done, _ = self.env.step(action)
#                print('reward', reward, 'done', done)
                next_state = np.reshape(next_state, [1, self.state_space])
                # Store the experience in the replay buffer
                self.replay_buffer.append((state, action, reward, next_state, done))
                # Update the state and episode reward
                state = next_state
                episode_reward += reward
                self.total_reward += reward
                t += 1
#                print('t', t, 'reward', reward, 'success_count', success_count)
                if reward > 0:
                    success_count += 1                
                # Update parameters for tracking after training
#                 success_rate_after = self.env.success_rate_after[-1]
#                 processing_time_after = self.env.processing_time_after[-1]
#                 transaction_charge_after = self.env.transaction_charge_after[-1]
                total_reward_after = self.env.total_reward_after[-1]
#                print('b-3-total_reward_after', total_reward_after)

                # Decay the epsilon value
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay                

                if len(self.memory) > self.batch_size:
                    self.replay()

                episode_rewards.append(self.total_reward)
                episode_success_rates.append(success_count / t)                  

#                 # Train the Dueling DQN on a batch of experiences from the replay buffer
#                 self.replay()

            # Update the parameter tracking lists


#                 self.env.success_rate_after.append(self.env.X.iloc[self.env.current_index]['success_rate'])
#                 self.env.processing_time_after.append(self.env.X.iloc[self.env.current_index]['processing_time'])
#                 self.env.transaction_charge_after.append(self.env.X.iloc[self.env.current_index]['transaction_charge'])
#                print('self.total_reward', self.total_reward)
#                 print('self.env.total_reward_after', self.env.total_reward_after)
#     #                self.env.total_reward_after.append(total_reward_after + self.total_reward)
#                 self.env.total_reward_after.append(total_reward_after + self.total_reward)
#                 print('self.env.total_reward_after', self.env.total_reward_after)
#                 self.env.average_reward_after.append(total_reward_after / (num_episodes + 1))                
#                 print('self.env.total_reward_after', self.env.total_reward_after)

                # Update the target network weights
                self.update_target_network()

                # Update epsilon
                self.update_epsilon(episode)

            self.episode_done_callback(episode=episode, total_reward=self.total_reward)
#            print("Episode {}: {}".format(episode, episode_reward))
#            print('total_reward', self.total_reward, 'avg_reward', self.total_reward/num_episodes, 'episodes', num_episodes)    
            
                # Plot the parameters before and after
#                 print('episode_rewards', episode_rewards)
#                 print('episode_success_rates', episode_success_rates) 
            # Plot the parameters before and after
#            print('episode_reward',episode_reward)
#            print('total_reward', self.total_reward)
#            print('success_count',success_count)
#            print('t', t)
        plt.figure(figsize=(20, 10))

        # Plot success rate
        plt.subplot(2, 5, 1)
        plt.plot(env.success_rate_before)
        plt.plot(env.success_rate_after)
        plt.title("Success Rate")

        # Plot processing time
        plt.subplot(2, 5, 2)
        plt.plot(self.env.processing_time_before)
        plt.plot(self.env.processing_time_after)
        plt.title("Processing Time")

        # Plot transaction charge
        plt.subplot(2, 5, 3)
        plt.plot(self.env.transaction_charge_before)
        plt.plot(self.env.transaction_charge_after)
        plt.title("Transaction Charge")

        # Plot total reward
        plt.subplot(2, 5, 4)
        plt.plot(self.env.total_reward_before)
        plt.plot(self.env.total_reward_after)
        plt.title("Total Reward")

        # Plot average reward
        plt.subplot(2, 5, 5)
        plt.plot(self.env.average_reward_before)
        plt.plot(self.env.average_reward_after)
        plt.title("Average Reward")
     
        
        plt.show()
        print('episode', episode)
        print('total records before', len(self.env.success_rate_before),'success_rate_before 0.9', sum(list(map(lambda x: 1 if x > 0.9 else 0, self.env.success_rate_before))))
        print('total records after', len(self.env.success_rate_after), 'success_rate_after 0.9', sum(list(map(lambda x: 1 if x > 0.9 else 0, self.env.success_rate_after))))
        print('________________________________________________________')
        print('episode', episode)
        print('success_rate_before 0.5', sum(list(map(lambda x: 1 if x > 0.5 else 0, self.env.success_rate_before))))
        print('success_rate_after 0.5', sum(list(map(lambda x: 1 if x > 0.5 else 0, self.env.success_rate_after))))
        print('________________________________________________________')

#         print('env.processing_time_before', env.processing_time_before)
#         print('env.processing_time_after', env.processing_time_after)
        print('________________________________________________________')        
#         print('Individual total records before', self.env.total_reward_before)
#         print('Individual total records after', self.env.total_reward_after)
        print('________________________________________________________')        
        print('self.env.total_reward_before', sum(self.env.total_reward_before))
        print('self.env.total_reward_after', sum(self.env.total_reward_after))
        print('________________________________________________________')  
        print('self.env.average_reward_before', sum(self.env.total_reward_before)/num_episodes)
        print('self.env.average_reward_after', sum(self.env.total_reward_after)/num_episodes)
        
#         episode_rewards.append(self.total_reward)
#         episode_success_rates.append(success_count / t)              
        
        
#         # Append the episode reward to the list of rewards
#         episode_rewards.append(episode_reward)
#         self.total_reward += episode_reward
        # Plot the average reward per episode
        avg_rewards = [sum(episode_rewards[:i+1])/(len(episode_rewards)-1) for i in range(len(episode_rewards))]
#        print('episode_rewards', episode_rewards)
#        print('avg_rewards', avg_rewards)
#        print('average reward', sum(episode_rewards)/(len(episode_rewards)-1))
        plt.plot(avg_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.show()
        
        epi_suc_rate = [sum(episode_success_rates[:i+1])/len(episode_success_rates[:i+1]) for i in range(len(episode_success_rates))]
#        print('tot_rwds', tot_rwds_bfr)
#        print('self.env.total_reward_after', self.env.total_reward_before)
#        print('total reward before', sum(tot_rwds_bfr)/len(tot_rwds_bfr))
        plt.plot(epi_suc_rate)
        plt.xlabel("Episode")
        plt.ylabel("Episode Success Rate")
        plt.show()
        
        tot_rwds = [sum(self.env.total_reward_after[:i+1])/len(self.env.total_reward_after[:i+1]) for i in range(len(self.env.total_reward_after))]
#        print('tot_rwds', tot_rwds)
#        print('self.env.total_reward_after', self.env.total_reward_after)
#        print('total reward after', sum(tot_rwds)/len(tot_rwds))
        plt.plot(tot_rwds)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward After")
        plt.show()

#         total_rewards_plot = [sum(self.total_reward[:i+1])/len(self.total_reward[:i+1]) for i in range(len(self.total_reward))]
#         print(total_rewards_plot)
#         print('total reward', sum(total_rewards_plot)/len(total_rewards_plot))
#         plt.plot(tot_rwds)
#         plt.xlabel("Episode")
#         plt.ylabel("Total Reward per episode")
#         plt.show()


# In[56]:


import tensorflow.keras.backend as K

env = PaymentEnv(X)
agent = DuelingDQN(env, env.observation_space.shape[0], env.action_space.n)
rewards = agent.train(episodes=1)


# In[57]:


import tensorflow.keras.backend as K

env = PaymentEnv(X)
agent = DuelingDQN(env, env.observation_space.shape[0], env.action_space.n)
rewards = agent.train(episodes=50)


# In[ ]:





# In[ ]:





# In[ ]:




To say that a particular approach is better, you can compare its performance with other approaches or benchmarks in terms of relevant metrics. In the case of the modified Thompson Sampling algorithm that you have implemented, you can compare its performance with other Multi-Objective Reinforcement Learning (MORL) algorithms or baselines using appropriate evaluation metrics.

For example, you can evaluate the performance of the modified Thompson Sampling algorithm in terms of its success rate, transaction charges, and processing time and compare it with other MORL algorithms using metrics such as Pareto Front, Hypervolume, and other appropriate multi-objective evaluation metrics. If the modified Thompson Sampling algorithm outperforms other MORL algorithms or benchmarks in terms of these metrics, then you can conclude that it is better.
# In[94]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(35)

# Define the number of trials and the number of arms
num_trials = 5000
num_arms = 7

# Define the true probabilities of success for each arm
true_probs = np.random.beta(1, 10, size=num_arms)

# Define the transaction charges for each arm
transaction_charges = np.random.uniform(0.01, 0.05, size=num_arms)

# Define the processing time for each arm
processing_times = np.random.normal(10, 2, size=num_arms)

# Define the weight vector for multi-objective Thompson Sampling
weights = np.array([0.6, 0.2, 0.2])

# Define the results arrays for both algorithms
ts_results = np.zeros(num_trials)
mots_results = np.zeros(num_trials)

# Define the Thompson Sampling algorithm function
def thompson_sampling(num_trials, num_arms, true_probs):
    # Define the successes and failures arrays
    successes = np.zeros(num_arms)
    failures = np.zeros(num_arms)
    
    # Define the results array
    results = np.zeros(num_trials)
    
    # Loop over the number of trials
    for i in range(num_trials):
        # Sample a random value from each arm's beta distribution
        samples = np.random.beta(successes+1, failures+1)
        
        # Select the arm with the highest sample value
        choice = np.argmax(samples)
        
        # Update the successes and failures arrays based on the result of the chosen arm
        result = np.random.binomial(1, true_probs[choice])
        if result == 1:
            successes[choice] += 1
        else:
            failures[choice] += 1
        
        # Update the results array
        results[i] = result
    
    return results

# Define the Multi-Objective Thompson Sampling algorithm function
def multi_objective_thompson_sampling(num_trials, num_arms, true_probs, transaction_charges, processing_times, weights):
    # Define the successes and failures arrays
    successes = np.zeros(num_arms)
    failures = np.zeros(num_arms)
    
    # Define the results array
    results = np.zeros(num_trials)
    
    # Loop over the number of trials
    for i in range(num_trials):
        # Sample a random value from each arm's beta distribution
        samples = np.random.beta(successes+1, failures+1)
        
        # Select the arm with the highest weighted sample value for each objective
        success_choices = np.argsort(weights[0]*samples - weights[1]*transaction_charges - weights[2]*processing_times)[::-1]
        choice = success_choices[0]
        
        # Update the successes and failures arrays based on the result of the chosen arm
        result = np.random.binomial(1, true_probs[choice])
        if result == 1:
            successes[choice] += 1
        else:
            failures[choice] += 1
        
        # Update the results array
        results[i] = result
    
    return results

# Run both algorithms and store the results in the results arrays
ts_results = thompson_sampling(num_trials, num_arms, true_probs)
mots_results = multi_objective_thompson_sampling(num_trials, num_arms, true_probs, transaction_charges, processing_times, weights)






# Plot the results
plt.plot(np.cumsum(ts_results)/np.arange(1, num_trials+1), label='Thompson Sampling')
plt.plot(np.cumsum(mots_results)/np.arange(1, num_trials+1), label='Epsilon Greedy')
plt.legend(loc='best')
plt.title('Cumulative Success Rate')
plt.xlabel('Number of Trials')
plt.ylabel('Success Rate')
plt.show()



# In[65]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# split data into train and test sets
X = X_y.drop(['success_rate', 'processing_time', 'transaction_charge'], axis=1)
y_success = X_y['success_rate']
y_processing_time = X_y['processing_time']
y_transaction_charge = X_y['transaction_charge']

X_train, X_test, y_success_train, y_success_test, y_processing_time_train, y_processing_time_test, y_transaction_charge_train, y_transaction_charge_test = train_test_split(X, y_success, y_processing_time, y_transaction_charge, test_size=0.2, random_state=42)


# In[66]:


import tensorflow as tf

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(7)
])

# Compile the model
model.compile(optimizer='sgd', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)


# In[67]:


import tensorflow as tf

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(7)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)


# In[68]:


y_predict = model.predict(X_test)


# In[69]:


y_train


# In[70]:


X_y.head(3)


# In[71]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# split data into train and test sets
X = X_y.drop(['success_rate', 'processing_time', 'transaction_charge'], axis=1)
y_success = X_y['success_rate']
y_processing_time = X_y['processing_time']
y_transaction_charge = X_y['transaction_charge']

X_train, X_test, y_success_train, y_success_test, y_processing_time_train, y_processing_time_test, y_transaction_charge_train, y_transaction_charge_test = train_test_split(X, y_success, y_processing_time, y_transaction_charge, test_size=0.2, random_state=42)

# train random forest models
rf_success = RandomForestRegressor()
rf_processing_time = RandomForestRegressor()
rf_transaction_charge = RandomForestRegressor()

rf_success.fit(X_train, y_success_train)
rf_processing_time.fit(X_train, y_processing_time_train)
rf_transaction_charge.fit(X_train, y_transaction_charge_train)

# predict on test data
y_success_pred = rf_success.predict(X_test)
y_processing_time_pred = rf_processing_time.predict(X_test)
y_transaction_charge_pred = rf_transaction_charge.predict(X_test)


# In[72]:


from sklearn.metrics import mean_squared_error, r2_score

# calculate MSE for success rate prediction
success_rate_mse = mean_squared_error(y_success_test, y_success_pred)
# calculate MSE for processing time prediction
processing_time_mse = mean_squared_error(y_processing_time_test, y_processing_time_pred)
# calculate MSE score for transaction charge prediction
transaction_charge_mse = mean_squared_error(y_transaction_charge_test, y_transaction_charge_pred)

print('MSE for success rate prediction:', success_rate_mse)
print('MSE for processing time prediction:', processing_time_mse)
print('MSE score for transaction charge prediction:', transaction_charge_mse)


# calculate RMSE for processing time prediction
success_rate_rmse = mean_squared_error(y_success_test, y_success_pred, squared=False)
processing_time_rmse = mean_squared_error(y_processing_time_test, y_processing_time_pred, squared=False)
transaction_charge_rmse = mean_squared_error(y_transaction_charge_test, y_transaction_charge_pred, squared=False)
print('RMSE for success rate prediction:', success_rate_rmse)
print('RMSE for processing time prediction:', processing_time_rmse)
print('RMSE score for transaction charge prediction:', transaction_charge_rmse)

# calculate R-squared score for transaction charge prediction
rf_success_score = rf_success.score(X_test, y_success_pred)
rf_processing_time_score = rf_processing_time.score(X_test, y_processing_time_pred)
rf_transaction_charge_score = rf_transaction_charge.score(X_test, y_transaction_charge_pred)

print("Random Forest Success Rate R^2 Score:", rf_success_score)
print("Random Forest Processing Time R^2 Score:", rf_processing_time_score)
print("Random Forest Transaction Charge R^2 Score:", rf_transaction_charge_score)


# In[73]:


X_y.head(3)


# In[74]:


import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_y.drop(["success_rate", "processing_time", "transaction_charge"], axis=1), X_y[["success_rate", "processing_time", "transaction_charge"]], test_size=0.2, random_state=42)

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(3,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(3)
])

print(model.summary())
# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model on test data
accuracy = model.evaluate(X_test, y_test)

# Print accuracy
print('Accuracy:', accuracy)


# In[76]:


y_pred


# In[77]:


import numpy as np

#y_pred = y_pred_tensor.numpy()
y_success_pred, y_processing_time_pred, y_transaction_charge_pred = np.split(y_pred, 3, axis=1)


# In[78]:


from sklearn.metrics import mean_squared_error

# calculate MSE
mse_success = mean_squared_error(y_success_test, y_success_pred)
mse_processing_time = mean_squared_error(y_processing_time_test, y_processing_time_pred)
mse_transaction_charge = mean_squared_error(y_transaction_charge_test, y_transaction_charge_pred)

# calculate RMSE
rmse_success = mean_squared_error(y_success_test, y_success_pred, squared=False)
rmse_processing_time = mean_squared_error(y_processing_time_test, y_processing_time_pred, squared=False)
rmse_transaction_charge = mean_squared_error(y_transaction_charge_test, y_transaction_charge_pred, squared=False)

print('MSE Success Rate:', mse_success)
print('RMSE Success Rate:', rmse_success)

print('MSE Processing Time:', mse_processing_time)
print('RMSE Processing Time:', rmse_processing_time)

print('MSE Transaction Charge:', mse_transaction_charge)
print('RMSE Transaction Charge:', rmse_transaction_charge)


# In[81]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Step 1: Import libraries

# Step 2: Load data
#X_y = pd.read_csv('your_dataset.csv')

# Step 3: Prepare data
X = X_y
y = X_y[['success_rate', 'processing_time', 'transaction_charge']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Define model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(3)
])

# Step 5: Compile model
model.compile(optimizer='adam', loss='mse')

# Step 6: Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Step 7: Evaluate model
test_loss = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

# Compute accuracy and other performance metrics


# In[ ]:


X_y.columns


# In[ ]:


X_y.status.value_counts(normalize=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import tensorflow as tf

# Define the neural network architecture
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(7)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Define the number of trials and the number of arms
num_trials = 5000
num_arms = 7

# Define the true probabilities of success for each arm
true_probs = np.random.beta(1, 10, size=num_arms)

# Define the transaction charges for each arm
transaction_charges = np.random.uniform(0.01, 0.05, size=num_arms)

# Define the processing time for each arm
processing_times = np.random.normal(10, 2, size=num_arms)

# Define the weight vector for multi-objective Thompson Sampling
weights = np.array([0.6, 0.2, 0.2])

# Define the results arrays for both algorithms
ts_results = np.zeros(num_trials)
mots_results = np.zeros(num_trials)

# Define the Thompson Sampling algorithm function
def thompson_sampling(num_trials, num_arms, true_probs):
    # Define the successes and failures arrays
    successes = np.zeros(num_arms)
    failures = np.zeros(num_arms)
    
    # Define the results array
    results = np.zeros(num_trials)
    
    # Loop over the number of trials
    for i in range(num_trials):
        # Sample a random value from each arm's beta distribution
        samples = np.random.beta(successes+1, failures+1)
        
        # Select the arm with the highest sample value
        choice = np.argmax(samples)
        
        # Update the successes and failures arrays based on the result of the chosen arm
        result = np.random.binomial(1, true_probs[choice])
        if result == 1:
            successes[choice] += 1
        else:
            failures[choice] += 1
        
        # Update the results array
        results[i] = result
    
    return results

# Define the Multi-Objective Thompson Sampling algorithm function
def multi_objective_thompson_sampling(num_trials, num_arms, true_probs, transaction_charges, processing_times, weights):
    # Define the successes and failures arrays
    successes = np.zeros(num_arms)
    failures = np.zeros(num_arms)
    
    # Define the results array
    results = np.zeros(num_trials)
    
    # Loop over the number of trials
    for i in range(num_trials):
        # Sample a random value from each arm's beta distribution
        samples = np.random.beta(successes+1, failures+1)
        
        # Select the arm with the highest weighted sample value for each objective
        success_choices = np.argsort(weights[0]*samples - weights[1]*transaction_charges - weights[2]*processing_times)[::-1]
        choice = success_choices[0]
        
        # Update the successes and failures arrays based on the result of the chosen arm
        result = np.random.binomial(1, true_probs[choice])
        if result == 1:
            successes[choice] += 1
        else:
            failures[choice] += 1
        
        # Update the results array
        results[i] = result
    
    return results

# Run both algorithms and store the results in the results arrays
ts_results = thompson_sampling(num_trials, num_arms, true_probs)
mots_results = multi_objective_thompson_sampling(num_trials, num_arms, true_probs, transaction_charges, processing_times, weights)






# Plot the results
plt.plot(np.cumsum(ts_results)/np.arange(1, num_trials+1), label='Thompson Sampling')
plt.plot(np.cumsum(mots_results)/np.arange(1, num_trials+1), label='Epsilon Greedy')
plt.legend(loc='best')
plt.title('Cumulative Success Rate')
plt.xlabel('Number of Trials')
plt.ylabel('Success Rate')
plt.show()



# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Define the number of trials and the number of arms
num_trials = 5000
num_arms = 10

# Define the true probabilities of success for each arm
true_probs = np.random.beta(1, 10, size=num_arms)

# Define the transaction charges for each arm
transaction_charges = np.random.uniform(0.01, 0.05, size=num_arms)

# Define the processing time for each arm
processing_times = np.random.normal(10, 2, size=num_arms)

# Define the weight vector for multi-objective Thompson Sampling
weights = np.array([0.6, 0.2, 0.2])

# Define the results arrays for both algorithms
ts_results = np.zeros(num_trials)
mots_results = np.zeros(num_trials)

# Define the Thompson Sampling algorithm function
def thompson_sampling(num_trials, num_arms, true_probs):
    # Define the successes and failures arrays
    successes = np.zeros(num_arms)
    failures = np.zeros(num_arms)
    
    # Define the results array
    results = np.zeros(num_trials)
    
    # Loop over the number of trials
    for i in range(num_trials):
        # Sample a random value from each arm's beta distribution
        samples = np.random.beta(successes+1, failures+1)
        
        # Select the arm with the highest sample value
        choice = np.argmax(samples)
        
        # Update the successes and failures arrays based on the result of the chosen arm
        result = np.random.binomial(1, true_probs[choice])
        if result == 1:
            successes[choice] += 1
        else:
            failures[choice] += 1
        
        # Update the results array
        results[i] = result
    
    return results

# Define the Multi-Objective Thompson Sampling algorithm function
def multi_objective_thompson_sampling(num_trials, num_arms, true_probs, transaction_charges, processing_times, weights):
    # Define the successes and failures arrays
    successes = np.zeros(num_arms)
    failures = np.zeros(num_arms)
    
    # Define the results array
    results = np.zeros(num_trials)
    
    # Loop over the number of trials
    for i in range(num_trials):
        # Sample a random value from each arm's beta distribution
        samples = np.random.beta(successes+1, failures+1)
        
        # Select the arm with the highest weighted sample value for each objective
        success_choices = np.argsort(weights[0]*samples - weights[1]*transaction_charges - weights[2]*processing_times)[::-1]
        choice = success_choices[0]
        
        # Update the successes and failures arrays based on the result of the chosen arm
        result = np.random.binomial(1, true_probs[choice])
        if result == 1:
            successes[choice] += 1
        else:
            failures[choice] += 1
        
        # Update the results array
        results[i] = result
    
    return results

# Run both algorithms and store the results in the results arrays
ts_results = thompson_sampling(num_trials, num_arms, true_probs)
mots_results = multi_objective_thompson_sampling(num_trials, num_arms, true_probs, transaction_charges, processing_times, weights)






# Plot the results
plt.plot(np.cumsum(ts_results)/np.arange(1, num_trials+1), label='Thompson Sampling')
plt.plot(np.cumsum(mots_results)/np.arange(1, num_trials+1), label='Epsilon Greedy')
plt.legend(loc='best')
plt.title('Cumulative Success Rate')
plt.xlabel('Number of Trials')
plt.ylabel('Success Rate')
plt.show()


# Define the number of trials and the number of arms
num_trials = 5000
num_arms = len(action_space)  # use the length of the action space as the number of arms


# Extract the transaction charges and processing times from the dataframe
transaction_charges = df['transaction_charges'].values[:num_arms]
processing_times = df['processing_time'].values[:num_arms]

# In[ ]:


df.columns


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Define the number of trials and the number of arms
num_trials = 5000
num_arms = 10

# Define the true probabilities of success for each arm
true_probs = np.random.beta(1, 10, size=num_arms)

# Define the transaction charges for each arm
transaction_charges = np.random.uniform(0.01, 0.05, size=num_arms)

# Define the processing time for each arm
processing_times = np.random.normal(10, 2, size=num_arms)

# Define the weight vector for multi-objective Thompson Sampling
weights = np.array([0.6, 0.2, 0.2])

# Define the results arrays for both algorithms
ts_results = np.zeros(num_trials)
mots_results = np.zeros(num_trials)

# Define the Thompson Sampling algorithm function
def thompson_sampling(num_trials, num_arms, true_probs):
    # Define the successes and failures arrays
    successes = np.zeros(num_arms)
    failures = np.zeros(num_arms)
    
    # Define the results array
    results = np.zeros(num_trials)
    
    # Loop over the number of trials
    for i in range(num_trials):
        # Sample a random value from each arm's beta distribution
        samples = np.random.beta(successes+1, failures+1)
        
        # Select the arm with the highest sample value
        choice = np.argmax(samples)
        
        # Update the successes and failures arrays based on the result of the chosen arm
        result = np.random.binomial(1, true_probs[choice])
        if result == 1:
            successes[choice] += 1
        else:
            failures[choice] += 1
        
        # Update the results array
        results[i] = result
    
    return results

# Define the Multi-Objective Thompson Sampling algorithm function
def multi_objective_thompson_sampling(num_trials, num_arms, true_probs, transaction_charges, processing_times, weights):
    # Define the successes and failures arrays
    successes = np.zeros(num_arms)
    failures = np.zeros(num_arms)
    
    # Define the results array
    results = np.zeros(num_trials)
    
    # Loop over the number of trials
    for i in range(num_trials):
        # Sample a random value from each arm's beta distribution
        samples = np.random.beta(successes+1, failures+1)
        
        # Select the arm with the highest weighted sample value for each objective
        success_choices = np.argsort(weights[0]*samples - weights[1]*transaction_charges - weights[2]*processing_times)[::-1]
        choice = success_choices[0]
        
        # Update the successes and failures arrays based on the result of the chosen arm
        result = np.random.binomial(1, true_probs[choice])
        if result == 1:
            successes[choice] += 1
        else:
            failures[choice] += 1
        
        # Update the results array
        results[i] = result
    
    return results

# Run both algorithms and store the results in the results arrays
ts_results = thompson_sampling(num_trials, num_arms, true_probs)
mots_results = multi_objective_thompson_sampling(num_trials, num_arms, true_probs, transaction_charges, processing_times, weights)






# Plot the results
plt.plot(np.cumsum(ts_results)/np.arange(1, num_trials+1), label='Thompson Sampling')
plt.plot(np.cumsum(mots_results)/np.arange(1, num_trials+1), label='Epsilon Greedy')
plt.legend(loc='best')
plt.title('Cumulative Success Rate')
plt.xlabel('Number of Trials')
plt.ylabel('Success Rate')
plt.show()



# In[ ]:


def simulate(num_trials, num_arms, success_rates, transaction_charges, processing_times):
    # Initialize arrays to store results
    ts_rewards = np.zeros(num_trials)
    ts_actions = np.zeros(num_trials, dtype=int)
    mo_ts_rewards = np.zeros(num_trials)
    mo_ts_actions = np.zeros(num_trials, dtype=int)

    # Initialize Thompson Sampling agent
    ts_agent = ThompsonSampling(num_arms)

    # Initialize Multi-objective Thompson Sampling agent
    mo_ts_agent = MultiObjectiveThompsonSampling(num_arms)

    # Run simulations
    for i in range(num_trials):
        # Thompson Sampling
        ts_action = ts_agent.get_action()
        ts_reward = np.random.binomial(1, success_rates[ts_action]) - transaction_charges[ts_action]
        ts_agent.update(ts_action, ts_reward)
        ts_rewards[i] = ts_reward
        ts_actions[i] = ts_action

        # Multi-objective Thompson Sampling
        mo_ts_action = mo_ts_agent.get_action()
        mo_ts_reward = np.array([
            np.random.binomial(1, success_rates[a]) - transaction_charges[a] for a in range(num_arms)
        ])
        mo_ts_agent.update(mo_ts_action, mo_ts_reward)
        mo_ts_rewards[i] = np.sum(mo_ts_reward)
        mo_ts_actions[i] = mo_ts_action

    return ts_rewards, ts_actions, mo_ts_rewards, mo_ts_actions


# In[ ]:


def simulate(num_trials, num_arms, success_rates, transaction_charges, processing_times):
    # Thompson Sampling
    ts_rewards = thompson_sampling(num_trials, num_arms, success_rates)

    # Multi-Objective Thompson Sampling
    mo_ts_rewards = multi_objective_thompson_sampling(num_trials, num_arms, success_rates, transaction_charges, processing_times)

    # Plot the results
    plt.plot(np.cumsum(ts_rewards)/np.arange(1, num_trials+1), label='Thompson Sampling')
    plt.plot(np.cumsum(mo_ts_rewards)/np.arange(1, num_trials+1), label='Multi-Objective Thompson Sampling')
    plt.legend()
    plt.title('Cumulative Average Rewards')
    plt.xlabel('Trials')
    plt.ylabel('Cumulative Average Reward')
    plt.show()

    # Print the total rewards and actions for each algorithm
    print('Thompson Sampling Total Rewards: {}'.format(np.sum(ts_rewards)))
    print('Thompson Sampling Total Actions: {}'.format(ts_actions))
    print('Multi-Objective Thompson Sampling Total Rewards: {}'.format(np.sum(mo_ts_rewards)))
    print('Multi-Objective Thompson Sampling Total Actions: {}'.format(mo_ts_actions))


# In[ ]:


# Run both algorithms and store the results in the results arrays
ts_results = thompson_sampling(num_trials, num_arms, true_probs)
mots_results = multi_objective_thompson_sampling(num_trials, num_arms, true_probs, transaction_charges, processing_times, weights)


# In[ ]:


ds_ts_results = pd.Series(ts_results)
ds_ts_results.value_counts()


# In[ ]:


ds_mots_results = pd.Series(mots_results)
ds_mots_results.value_counts()


# In[ ]:


X_train.shape, X_test.shape


# In[ ]:




here's a general outline of the steps involved in implementing the changes:

Evaluating DuellingDQN performance on the dataset:
Load the dataset into memory
Initialize a DuellingDQN model with appropriate hyperparameters and train on the dataset
Use appropriate metrics to evaluate the performance of the model, such as accuracy, precision, recall, and F1 score
Compare the performance of the model with other models that you might want to use, using the same dataset
Incorporating Thompson Sampling into DuellingDQN:
Modify the algorithm to use Thompson Sampling instead of the epsilon-greedy approach for exploration
Re-initialize the modified DuellingDQN model with appropriate hyperparameters and train on the same dataset used for the original model
Use appropriate metrics to evaluate the performance of the modified model and compare with the original version
Comparing the performance of the modified DuellingDQN with the original model:
Use appropriate metrics to evaluate the performance of both the modified and original DuellingDQN models on the same dataset
If the modified model outperforms the original model, then it can be said that the success rate has increased
However, consider other factors such as the complexity of the modified model and the time required to train it.
Keep in mind that these steps are just a general outline and may need to be customized based on the specific dataset and models being used.
# In[63]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate random data using beta distribution
success_rate = np.random.beta(2, 5, 100)   # alpha=2, beta=5
processing_time = np.random.beta(3, 3, 100) # alpha=3, beta=3
transaction_charges = np.random.beta(4, 2, 100) # alpha=4, beta=2
status = np.random.beta(5, 1, 100) # alpha=5, beta=1
amount = np.random.beta(1, 4, 100) # alpha=1, beta=4

# Create a pandas DataFrame
df = pd.DataFrame({'success_rate': success_rate,
                   'processing_time': processing_time,
                   'transaction_charges': transaction_charges,
                   'status': status,
                   'amount': amount})

# Add a payment type column with values 0-6
df['payment_type'] = np.random.randint(0, 7, 100)

# Plot the distribution for each payment type
payment_types = df['payment_type'].unique()
for payment_type in payment_types:
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].set_ylabel('Frequency')
    fig.suptitle(f'Payment Type {payment_type}')
    data = df[df['payment_type'] == payment_type]
    for i, col in enumerate(df.columns[:-1]):
        axs[i].hist(data[col], bins=20)
        axs[i].set_xlabel(col)
    plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

a = np.array([81.13925399369354,
  727.8398809294558,
  3948.1147623095508,
  580.8549698236619,
  512.5526313507845,
  346.7319229036252,
  1137.311194224576,
  66.23181141897115,
  276.6011178949515,
  151.3548891827271])


# In[3]:


a.mean()


# In[6]:


plt.plot(1, len(a), a)
plt.show()


# In[ ]:




