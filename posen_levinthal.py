#!/usr/bin/env python
# coding: utf-8

# # Tutorial on Modeling of Learning Under Uncertainty
# 
# In this tutorial, you will be introduced to three modeling papers on reinforcement learning. These papers study the phenomenon of organizationa learning under uncertainty trough the use of agent based models who learn from n-Arm bandits.
# 
# The models we will study are:
# 1. March and Denrell (2001)
# This paper introduced the so called Hotstove effect. The idea being that if agents choose between two options with the same mean, one with variance in its feedback and one without, the agents will learn to choose the variant without variance.
# 
# 2. Posen and Levinthal (2012)
# This paper expands the model from Denrell and March (2001) and studies how agents learn in an environment with ten options, all of them with variance. It studies the effects of different shocks and environmental changes as agents chase a moving target.
# 
# 3. Puranam and Swamy (2016)
# This paper expands the model from Denrell and March (2001) through the process of coupled learning. Two agents, not one as before, learn of the perfoemance of two options. The catch is that the performance is based upon their coupled action and the agetns do not know what the other agent is doing. Through time the agents learn to cooperate. Interestingly, if agents start with the same choice, they reach the optimal choice faster than if they started with mixed choices. This holds even if they start choosing the wrong option.

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
# Updating is done via the Bush-Mossteller equation. The parameter phi determins how much the agent updates its beliefs based upon new information. A value of zero leads to agents to not update their beliefs. A value of one to full update of beliefs. A mixture leads to what is known as an Exponentially Recency Weighted Average (Sutton and Barto, 1998). In Denrell and March (2001) and Puranam and Swamy (2016), we use a constant phi value. Posen and Levinthal use a varying phi for every trial. The phi varies according to 1/(ki+1) where ki is the number of times an option has been tried. 
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
    def __init__(self, tau, phi, style):
        self.tau = tau
        self.phi = phi
        self.style = style
    def update(self, choice, payoff):
        if self.style == "constant": self.attraction[choice] += self.phi*(payoff-self.attraction[choice])
        elif self.style == "over k":
            self.times[choice] += 1 #starts in 1
            self.attraction[choice] += (payoff-self.attraction[choice])/(self.times[choice]+1) # divides by 2
    def choose(self):
        return(softmax(self.tau, self.attraction))
    def learn(self, num_periods, bandits):
        choices = []
        payoffs = []
        knowledge = []
        for i in range(num_periods):
            choice = self.choose()
            payoff = bandits.measure(choice)
            nugget = 1-sum((self.attraction-bandits.means)**2)
            self.update(choice, payoff)
            choices.append(choice)
            payoffs.append(payoff)
            knowledge.append(nugget)
        return([choices, payoffs, knowledge])
    def reset(self, num_bandits):
        self.attraction = np.ones(num_bandits)/2.0
        if self.style == "over k": self.times = np.zeros(num_bandits)


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
    def __init__(self, mu, stdev, style):
        self.style = style
        self.mean = mu
        self.stdev = stdev
        if style == "Beta": self.mean = np.random.beta(a=mu, b=stdev)
    def measure(self):
        if self.style == "Uniform":  value = self.mean+self.stdev*(np.random.random()-0.5)
        elif self.style == "Normal": value = np.random.normal(loc=self.mean, scale=self.stdev)
        elif self.style == "Beta": value = np.random.binomial(n=1, p=self.mean)
        elif self.style == "Stable": value = self.mean
        return(value)


# ### Bandits_P_L
# This class creates the environment for the Posen and Levinthal (2012) paper. In specific, 10 bandits drawn from a Beta(2,2) distribution of probabilities of drawing -1 or 1.
# 
# #### Measure
# This is a wrapper function. The objective is that the agents ask the bandits class and not the specific bandit for the measurement. Then the bandits class is in charge of asking its bandit for the performance value. 

# In[3]:


class bandits_P_L:
    def __init__(self, num_bandits, eta=0.0):
        self.eta = eta
        self.arms = []
        self.means = np.zeros(num_bandits)
        for i in range(num_bandits): 
            mu = np.random.random()
            self.arms.append(bandit(mu = 2.0, stdev = 2.0, style = "Beta")) 
            self.means[i] = self.arms[i].mean
    def measure(self, choice):
        if np.random.binomial(n=1, p = self.eta):
            for i in range(len(self.arms)):
                if np.random.binomial(n=1, p = 0.5):
                    self.arms[i] = bandit(mu = 2.0, stdev = 2.0, style = "Beta")
                    self.means[i] = self.arms[i].mean
        return(self.arms[choice].measure())


# ### Posen and Levinthal (2012)
# 
# We now need to change some aspects before we can replicate the Posent and Levinthal (2012) paper.
# 
# Reference: Posen, H. E., & Levinthal, D. A. (2012). Chasing a moving target: Exploitation and exploration in dynamic environments. Management Science, 58(3), 587-601.
# 
# #### 1. Initialize values
# We start by initailizing the attributes of the simulation. The agents are given a tau but do not require phi because the learning method follows the 1/k+1 weighting. The agents will learn for 500 periods and the results replicated 1000 times. There is no noise value here because in this paper the bandits output either 1 or -1 with probabilities drawn from a beta(2,2) distribution. Changes in the tau, and bandit style should change the learning. Changes in the number of repetitions lead to more noisy results.

# In[4]:


## Bandit
eta = 0.0
num_bandits = 10
## Agents
phi = 0.1
tau = 0.5/num_bandits
agent_style = "over k"
## Simulation
num_periods = 500
num_reps = 1000


# #### 2. Initialize agent and Bandits
# We create one agent, Alice and initialize the environment for the paper. We create an environment with 10 bnadits. These bandits are different from the ones in the other papers as they are created from a Beta distribution of payoff probabilties. 

# In[5]:


## Initialize agents
Alice = agent(tau = tau, phi = phi, style = agent_style)
Alice.reset(num_bandits = num_bandits)


# #### 3. Run simulation
# Having the agent and environment we can run a simulation. We initialize two arrays, one for payoff and one for choices. Additionally, we create an empty list to store the last choices of the agents and one value to save the attraction to the option with variable output after every replication of the simulation is finished.
# 
# This takes some time.

# In[6]:


all_payoffs = np.zeros(num_periods)
all_knowledge = np.zeros(num_periods)
all_RE = np.zeros(num_periods)
for j in range(num_reps):
    Alice.reset(num_bandits = num_bandits)
    options = bandits_P_L(num_bandits = num_bandits, eta = eta)
    choice, payoff, knowledge = Alice.learn(num_periods, options)
    all_payoffs += payoff
    all_knowledge += knowledge
    # Calculate exploration
    all_RE[0] += 1
    for i in range(len(choice)-1):
        if choice[i+1]!=choice[i]: all_RE[i+1] +=1


# #### 4. Display results
# 
# ##### Amount of exploration
# First we present the amount of exploration done by the agents. 

# In[7]:


plt.scatter(range(num_periods), all_RE)


# #### Knowledge over time
# Somthing quite sad happens for the amount of knowledge over time in this paper. Given the way the Bush Mossteller equation is updated, 1/k+1 and not with a constant update percentage, initial values have much more weight that later values. This leads to the system to erode knowledge. ***Fast!

# In[8]:


plt.scatter(range(num_periods), all_knowledge)


# ##### Probability of Getting a Reward
# The second graph presents the average payoff. This looks like a funnel, narrowing from left to right. As the stable option is chosen more and more, the variance in the performanc decreases. 

# In[9]:


plt.scatter(range(num_periods), all_payoffs)


# ##### Summary variables
# ###### Total accumulated payoff
# The bandits in this paper give a payoff of 1 or -1, so one needs to recalculate the total payoff. So that it matches what is shown in the paper. In the simulation I store the probability of getting a 1, here, the actual expected value.

# In[10]:


print(sum(2*all_payoffs-num_reps)/num_reps)


# ###### Fraction of exploration events
# The average percentage of exploration events during the 500 periods.

# In[11]:


print(sum(all_RE)/(num_reps*num_periods))


# ###### Knowledge
# The average SSE knowledge at different stages of the simulation

# In[12]:


print("Knowledge at Period 400: " + str(all_knowledge[-101]/num_reps))
print("Knowledge at Period 500: " + str(all_knowledge[-1]/num_reps))


# #### 5. Exercise
# What would happen if Posen and Levinthal had chosen other learning rules? They studied e-greedy but how about constant update? How woud it affect the erosion of knowledge and the adaptation of the agents?
# 

# In[ ]:




