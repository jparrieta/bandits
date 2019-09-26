#!/usr/bin/env python
# coding: utf-8

# # Tutorial on Denrell and March (2001)
# 
# In this tutorial, you will be introduced to a simple model that replicates the main finding from the paper by Jerker Denrell and Jim March, published in 2001 in Organization Science. 
# 
# This tutorial provides a barebones description of the model. If you want to explore a more flexible version or explore how different agents or bandit distributions would affect Jerker's and Jim's results please follow the denrell_march.ipynb tutorial instead.
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
# ### 1. Update
# The updating follows an aspiration level. If the payoff received was higher than the aspiration then the probability of choosing that option is directly increased. If the payoff received is lower than the aspiration then the probability of choosing that options is lowered. The other options get updated accordingly. In the case of Denrell and March there are just two options so the other alternative the complement of the probability of the chosen option. The code I include allows for environments with more bandits. I include a more general version of how to update the probabilities o N bandits. 
# 
# ### 1. Choose
# The agents choose one option based upon the probabilities it has stored for each option. The probabilities are set during the update function. 
# 
# ### 3. Learn
# Learn is a subroutine. It receives two parameters, the number of periods and the bandits to learn from. 
# It initalizes two lists and starts a for loop that run for the specified number of periods.
# The period starts by asking the agent to choose an option. The payoff of the option is calculated by measuring the option's value. This is explained in the next section. The period ends with the updating of the agent's attractions and the storage of the choice and payoff. After this a new period starts, the choices are stored and returned to the user.
# 
# ### 4. Reset
# This function resets the attractions of the agent. It takes two values, the mean of the normal distributions of the bandits and the starting attraction to each bandit. 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

class agent:
    def __init__(self, tau, phi):
        self.tau = tau
        self.phi = phi
    def choose(self): return(np.random.choice(range(len(self.attraction)), p = self.attraction))
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
    def reset(self, means, att):
        if len(att) == num_bandits: self.attraction = np.asarray(att)
        else: self.attraction = np.ones(num_bandits)/2.0
        self.aspiration = np.sum(att[:]*means[:])
    def update(self, choice, payoff):
        # update Choice
        if payoff > self.aspiration: self.attraction[choice] += self.phi*(1.0-self.attraction[choice])
        else: self.attraction[choice] = (1-self.phi)*self.attraction[choice]
        # Update Others
        others = np.arange(len(self.attraction)) != choice
        self.attraction[others] = self.attraction[others]*((1.0-self.attraction[choice])/sum(self.attraction[others]))
        # Update Aspiration
        self.aspiration = self.aspiration*(1.0-self.tau) + payoff*self.tau


# ## Environment
# The environment is composed of an n-arm bandit. Each arm of the bandit is definied by an object of class bandit. Put together these objects create one object of class bandits. 
# 
# ### 1. Bandit
# The bandit is a simple structure. It has a mean and a standard deviation. 
# 
# #### Measure
# The bandits perform one function, when called upon, they give one output, a draw from a normal distribution centered around a mean value and with the given standard deviation.

# In[2]:


class bandit:
    def __init__(self, mean, noise):
        self.mean = mean
        self.noise = noise
    def measure(self): return(np.random.normal(loc = self.mean, scale = self.noise))


# ### 2. Bandits_D_M
# This class creates the environment for the paper. In specific, two bandits. Both bandits have a different mean. The first bandit has a noisy output and the second bnadit has a stable output.
# 
# #### Measure
# This is a wrapper function. The objective is that the agents ask the bandits class and not the specific bandit for the measurement. Then the bandits class is in charge of asking its bandit for the performance value.

# In[3]:


class bandits_D_M:
    def __init__(self, means,  noise): 
        self.means = means
        self.arms = [bandit(mean = means[0], noise = noise),
                     bandit(mean = means[1], noise = 0.0)]
        self.means = np.zeros(len(self.arms))
        for i in range(len(self.arms)): self.means[i] = self.arms[i].mean
    def measure(self, choice): return(self.arms[choice].measure())


# # Simulation
# 
# With these two building blocks, we can run a simulation to replicate the main finding of Denrell and March (2001).
# 
# ## 1. Initialize values
# We start by initailizing the attributes of the simulation. The agents are given a set of tau and phi. The agents will learn for 50 periods and the results replicated 2500 times. We specify the noise to be 1, that means the bandits will draw from values between 0 and 1. Changes in the tau, phi, noise, and bandit style should change the learning. Changes in the number of repetitions lead to more noisy results.

# In[4]:


## Bandits
X = 10.0
Y = 10.0
S = 10.0
num_bandits = 2
## Agents
a = 0.5
b = 0.5
start_p = np.ones(num_bandits)/num_bandits # can use a list of values
## Simulation
num_periods = 50
num_reps = 5000


# ## 2. Initialize agent and Bandits
# We create one agent, Alice and initialize the environment for the paper. The bandits are created by specifying first two agents one drawn from an uniform distribution and the second one from a stable value.

# In[5]:


Alice = agent(tau = a, phi = b)
options = bandits_D_M(means = [X,Y], noise = S)


# ## 3. Run simulation
# Having the agent and environment we can run a simulation. We initialize two arrays, one for payoff and one for choices. Additionally, we create an empty list to store the last choices of the agents and one value to save the attraction to the option with variable output after every replication of the simulation is finished.
# 
# This takes some time.

# In[6]:


all_payoffs = np.zeros(num_periods)
all_choices = np.zeros(num_periods)
all_aspiration = 0.0
last_choices = []
for j in range(num_reps):
    Alice.reset(means = options.means, att = np.ones(num_bandits)/2.0) # second attribute gets updated after a reset somehow
    choice, payoff = Alice.learn(num_periods, options)
    all_payoffs += payoff
    all_choices += choice
    all_aspiration += Alice.aspiration
    last_choices.append(choice[-1])


# ## 4. Display results
# 
# ### Choice as function of time
# We present two plots. The first one presents the option chosen on every period. As on every period the agent can choose 0 or 1, what we plot in the y-axis is the number of times the stable option is chosen. As expected, the first period starts at 50% of the time and it increases towards a 100% as time goes by.
# 

# In[7]:


plt.scatter(range(num_periods), all_choices)


# ### Performance as function of time
# The second graph presents the average payoff. This looks like a funnel, narrowing from left to right. As the stable option is chosen more and more, the variance in the performanc decreases. 

# In[8]:


plt.scatter(range(num_periods), all_payoffs)


# ## Summary Values  
# ### Fraction of individuals who chose the risky alternative at the end of period 50
# Both options have the same performance. Nonetheless, the risky option is chosen less than 1% of the time after 50 periods when b = 0.5, and around 30% of time if b = 0.1. X=Y=S=10.0 

# In[9]:


100*(1-float(sum(last_choices))/num_reps)


# ### Average aspiration level at the end of each simulation   
# The average aspiration at the end of each simulation was:

# In[10]:


all_aspiration/num_reps


# ## 5. Exercise
# Find the how high the mean of the variable option needs to be in order to be chosen 50% of the time at the end of the simulation. How does it related to the amount of noise in the option? How does it change if normal and not uniform noise is used? How is it affected by the values of a and b? 
