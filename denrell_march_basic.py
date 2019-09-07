#!/usr/bin/env python
# coding: utf-8

# # Tutorial on Denrell and March (2001)
# 
# In this tutorial, you will be introduced to a simple model that replicates the main finding from the paper by Jerker Denrell and Jim March, published in 2001 in Organization Science. 
# 
# This tutorial provides a barebones description of the model. If you want to explore a more flexible version or explore how different agents or bandit distributions would affect Jerker's and Jim's results please follow the denrell_march.ipynb tutorial.
# 
# **Reference:** Denrell, J., & March, J. G. (2001). Adaptation as information restriction: The hot stove effect. Organization Science, 12(5), 523-538.
# 

# # Basic Building Blocks
# 
# In this first sections, I will present the basic building blocks. Namely, the àgent's learning and decision processes and how we generate options in the environment.
# 
# ## Agents
# The agents follow a few simple actions. 
# 
# ### 1. Choose
# The agents choose one option based upon their attraction to this option. In this model, we use softmax as the rule for transforming the different attractions for each option into probabilities of choosing one option. Other rules as greedy and e-greedy are possible. The agent's level of exploration is determined by the parameter tau. A small tau leads to high exploitation, a high tau to infinite exploration. Due to limitations in the floating point operation we cannot use taus lower that 0.002.
# 
# ### 2. Update
# Updating is done via the Bush-Mossteller equation. The parameter phi determins how much the agent updates its beliefs based upon new information. A value of zero leads to agents to not update their beliefs. A value of one to full update of beliefs. A mixture leads to what is known as an Exponentially Recency Weighted Average (Sutton and Barto, 1998). In Denrell and March (2001), we use a constant phi value. Agents in this paper as fully specified by providing them a tau and a phi value.
# 
# ### 3. Learn
# Learn is a subroutine. It receives two parameters, the number of periods and the bandits to learn from. 
# It initalizes two lists and starts a for loop that run for the specified number of periods.
# The period starts by asking the agent to choose an option. The payoff of the option is calculated by measuring the option's value. This is explained in the next section. The period ends with the updating of the agent's attractions and the storage of the choice and payoff. After this a new period starts, the choices are stored and returned to the user.
# 
# ### 4. Reset
# This function resets the attractions of the agent. It takes one value, the number of bandits in the environment.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def softmax(tau, attraction): #softmax action selection with attraction vector as parameters
    denom = np.sum(np.exp((attraction[:])/tau))
    probabilities = np.exp(attraction/tau)/denom
    choice = np.random.choice(range(len(probabilities)), p = probabilities)
    return(choice)

class agent:
    def __init__(self, tau, phi):
        self.tau = tau
        self.phi = phi
    def update(self, choice, payoff): self.attraction[choice] += self.phi*(payoff-self.attraction[choice])
    def choose(self): return(softmax(self.tau, self.attraction))
    def learn(self, num_periods, bandits):
        choices = []
        payoffs = []
        for i in range(num_periods):
            choice = self.choose()
            payoff = bandits.measure(choice)
            self.update(choice, payoff)
            choices.append(choice)
            payoffs.append(payoff)
        return([choices, payoffs])
    def reset(self, num_bandits): self.attraction = np.ones(num_bandits)/2.0


# ## Environment
# The environment is composed of an n-arm bandit. Each arm of the bandit is definied by an object of class bandit. Put together these objects create one object of class bandits. 
# 
# ### Bandit
# The bandit is a simple structure. It has a mean and a standard deviation. The style determines whether the bandit outputs an uniformly distributed value, a normally distributed value, or one without noise.
# 
# #### Measure
# The bandits perform one function, when called upon, they give one output, centered around a mean value and with an added noise. The style of bandit determines where the noise is drawn upon.

# In[2]:


class bandit:
    def __init__(self, mean, noise):
        self.mean = mean
        self.noise = noise
    def measure(self): return(self.mean+self.noise*(np.random.random()-0.5))


# ### Bandits_D_M
# This class creates the environment for the paper. In specific, two bandits. One with stable output and one with variable output. Both bandits have the same mean.
# 
# #### Measure
# This is a wrapper function. The objective is that the agents ask the bandits class and not the specific bandit for the measurement. Then the bandits class is in charge of asking its bandit for the performance value. 

# In[3]:


class bandits_D_M:
    def __init__(self, noise):
        self.arms = [bandit(mean = 0.5, noise = noise), #0.622 equiprobability at 100 periods # 0.6392 0.5 expected value at 1000 periods
                     bandit(mean = 0.5, noise = 0.0)]
    def measure(self, choice): return(self.arms[choice].measure())


# ### Simulation
# 
# With these two building blocks, we can run a simulation to replicate the main finding of Denrell and March (2001).
# 
# #### 1. Initialize values
# We start by initailizing the attributes of the simulation. The agents are given a set of tau and phi. The agents will learn for 50 periods and the results replicated 2500 times. We specify the noise to be 1, that means the bandits will draw from values between 0 and 1. Changes in the tau, phi, noise, and bandit style should change the learning. Changes in the number of repetitions lead to more noisy results.

# In[4]:


## Bandits
noise = 1.0
## Agents
num_bandits = 2
tau = 0.01/num_bandits
phi = 0.1
## Simulation
num_periods = 100
num_reps = 2500


# #### 2. Initialize agent and Bandits
# We create one agent, Alice and initialize the environment for the paper. The bandits are created by specifying first two agents one drawn from an uniform distribution and the second one from a stable value.

# In[5]:


## Initialize agents
Alice = agent(tau = tau, phi = phi)
Alice.reset(num_bandits = num_bandits)
## Initialize bandits
options = bandits_D_M(noise = noise)


# #### 3. Run simulation
# Having the agent and environment we can run a simulation. We initialize two arrays, one for payoff and one for choices. Additionally, we create an empty list to store the last choices of the agents and one value to save the attraction to the option with variable output after every replication of the simulation is finished.
# 
# This takes some time.

# In[6]:


all_payoffs = np.zeros(num_periods)
all_choices = np.zeros(num_periods)
all_attractions = 0.0
last_choices = []
for j in range(num_reps):
    Alice.reset(num_bandits = num_bandits)      
    choice, payoff = Alice.learn(num_periods, options)
    all_payoffs += payoff
    all_choices += choice
    all_attractions += Alice.attraction[0]
    last_choices.append(choice[-1])


# #### 4. Display results
# 
# ##### Choice as function of time
# We present two plots. The first one presents the option chosen on every period. As on every period the agent can choose 0 or 1, what we plot in the y-axis is the number of times the stable option is chosen. As expected, the first period starts at 50% of the time and it increases towards a 100% as time goes by.
# 

# In[7]:


plt.scatter(range(num_periods), all_choices)


# ##### Performance as function of time
# The second graph presents the average payoff. This looks like a funnel, narrowing from left to right. As the stable option is chosen more and more, the variance in the performanc decreases. 

# In[8]:


plt.scatter(range(num_periods), all_payoffs)


# #### Summary variables
# ##### Percentage of time stability is chosen
# Both options have the same performance. Nonetheless, the stable option is chosen 98% of the time after 50 periods. 

# In[9]:


100*float(sum(last_choices))/num_reps


# ##### Expected attraction
# At the end ofthe each replication, we stored the attraction each agent had for the variable option. Below we cane see that agents perceived the average performance of this option to be 0.464, much lower than the 0.5 it really has.  

# In[10]:


all_attractions/num_reps


# # Exercise
# Find the how high the mean of the variable option needs to be in order to be chosen 50% of the time at the end of the simulation. How does it related to the amount of noise in the option? How does it change if normal and not uniform noise is used? Or if we use the 1/(k+1) updating instead of constant updating?

# In[ ]:




