#!/usr/bin/env python
# coding: utf-8

# # Tutorial on Modeling of Learning Under Uncertainty
# 
# In this tutorial, you will be introduced to three modeling papers on reinforcement learning. These papers study the phenomenon of organizationa learning under uncertainty trough the use of agent based models who learn from n-Arm bandits.
# 
# The models we will study are:  
# **1. March and Denrell (2001)**
# This paper introduced the so called Hotstove effect. The idea being that if agents choose between two options with the same mean, one with variance in its feedback and one without, the agents will learn to choose the variant without variance.
# 
# **2. Posen and Levinthal (2012)**
# This paper expands the model from Denrell and March (2001) and studies how agents learn in an environment with ten options, all of them with variance. It studies the effects of different shocks and environmental changes as agents chase a moving target.
# 
# **3. Puranam and Swamy (2016)**
# This paper expands the model from Denrell and March (2001) through the process of coupled learning. Two agents, not one as before, learn of the perfoemance of two options. The catch is that the performance is based upon their coupled action and the agetns do not know what the other agent is doing. Through time the agents learn to cooperate. Interestingly, if agents start with the same choice, they reach the optimal choice faster than if they started with mixed choices. This holds even if they start choosing the wrong option.
# 
# The document builds a general agent who can update its beliefs and choose options in different ways, the bandits can also draw values from different distributions. These differences come from the different papers. This tutorial introduces the three papers at once in order to allow you to play around. At every point you can change the forms of the distributions or how the agents learn, in most cases the results you will find will be clear. In other cases you will need to find the aprropriate parameters.  
#   
# **Note:** You can find narrower tutorials that cover just one paper at a time in the repository: https://github.com/jparrieta/bandits.  
# 

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# # Basic Building Blocks
# 
# In this first sections, I will present the basic building blocks. Namely, the àgent's learning and decision processes and how we generate options in the environment.  
# 
# ## Agents
# The agents follow a few simple actions.  
# 
# ### 1. Choose
# The agents choose one option based upon their attraction to this option. Here there four versions of choosing options: greedy, e-greedy, softmax, and aspiration-level based. Denrell and March (2001) use aspiration-levels, Posen and Levinthal (2012) the other three.  
# * If the agent selects options via softmax, The agent's level of exploration is determined by the parameter tau. A small tau leads to high exploitation, a high tau to infinite exploration. Due to limitations in the floating point operation we cannot use taus lower that 0.002.  
# * If greedy is chosen, there is no free parameter to specify the level exlorativeness of the agent.  
# * If e-greedy is chosen, then the e parameter specifies the percentage of times the agent chooses an option other than the greedy one.  
# * If aspiration is chosen, then an option is drawn from a Polya urn with probabilities specified by the prior learning. To use aspiration level one needs to choose self_update = self_choose = "aspiration".  
# 
# ### 2. Update
# There are three ways of updating. Two are based on the Bush Mossteller equation one on aspiration-level search.  
# * The Bush Mossteller updating uses an Exponentially Recency Weighted Average (Sutton and Barto, 2018). The updating is based on phi. Phi can be constant, or varying according to 1/(k+1). Posen and Levinthal uses the 1/(k+1) version.  
# * The aspiration-level version is used in Denrell and March (2001). Here if the payoff received was higher than the aspiration then the probability of choosing that option is directly increased. If the payoff received is lower than the aspiration then the probability of choosing that options is lowered. The other options get updated accordingly. To use aspiration level one needs to choose self_update = self_choose = "aspiration".  
# 
# ### 3. Learn
# Learn is a subroutine. It receives two parameters, the number of periods and the bandits to learn from. 
# It initalizes two lists and starts a for loop that run for the specified number of periods.  
# The period starts by asking the agent to choose an option. The payoff of the option is calculated by measuring the option's value. This is explained in the next section. The period ends with the updating of the agent's attractions and the storage of the choice and payoff. After this a new period starts, the choices are stored and returned to the user.  
# 
# ### 4. Reset
# This function resets the attractions of the agent. It takes one value, the number of bandits in the environment.  

# In[147]:


import numpy as np
import matplotlib.pyplot as plt

def softmax(tau, attraction): #softmax action selection with attraction vector as parameters
    denom = np.sum(np.exp((attraction[:])/tau))
    probabilities = np.exp(attraction/tau)/denom
    choice = np.random.choice(range(len(probabilities)), p = probabilities)
    return(choice)

class agent:
    def __init__(self, tau, phi, style_update, style_choose, style_start = "same"):
        self.tau = tau
        self.phi = phi
        self.style_update = style_update
        self.style_choose = style_choose
        self.style_start = style_start
    def choose(self):
        if self.style_choose == "softmax": choice = softmax(self.tau, self.attraction)
        elif self.style_choose == "greedy": choice = np.argmax(self.attraction)
        elif self.style_choose == "aspiration": choice = np.random.choice(range(len(self.attraction)), p = self.attraction)
        elif type(self.style_choose) == float: # for e-greedy you pass the e parameter only
            best_choice = np.argmax(self.attraction)
            other_choice = best_choice
            while other_choice == best_choice: other_choice = np.random.choice(range(len(self.attraction))) #lazy
            choice = np.random.choice([best_choice,other_choice], p = [1-self.style_choose,self.style_choose])
        return(choice)
    def learn(self, num_periods, bandits):
        choices = []
        payoffs = []
        knowledge = []
        RE = [1]
        for i in range(num_periods):
            choice = self.choose()
            payoff = bandits.measure(choice)
            nugget = 1-sum((self.attraction-bandits.means)**2)
            self.update(choice, payoff)
            choices.append(choice)
            payoffs.append(payoff)
            knowledge.append(nugget)
            if len(choices) > 1: RE.append(1*(choices[-1] != choices[-2]))
        return([choices, payoffs, knowledge, RE])
    def reset(self, means, att):
        if self.style_start == "same":
            self.attraction = np.ones(len(means))/2.0
            if self.style_update == "over k": self.times = np.zeros(len(means))
            if self.style_update == "aspiration": 
                self.aspiration = np.sum(att[:]*means[:])
                if len(att) == len(means): self.attraction = np.asarray(att)/np.sum(att)
                else: self.attraction = np.ones(len(means))/len(means)       
        elif self.style_start == "fixed":
            self.attraction = np.array(att)
            if self.style_update == "over k": self.times = np.zeros(len(means))
            if self.style_update == "aspiration": 
                self.aspiration = np.sum(att[:]*means[:])
                self.attraction = att
    def update(self, choice, payoff):
        if self.style_update == "constant": self.attraction[choice] += self.phi*(payoff-self.attraction[choice])
        elif self.style_update == "over k":
            self.times[choice] += 1 #starts in 1
            self.attraction[choice] += (payoff-self.attraction[choice])/(self.times[choice]+1) # divides by 2
        elif self.style_update == "aspiration":
            # update Choice
            if payoff > self.aspiration: self.attraction[choice] += self.phi*(1.0-self.attraction[choice])
            else: self.attraction[choice] = (1-self.phi)*self.attraction[choice]
            # Update Others
            others = np.arange(len(self.attraction)) != choice
            self.attraction[others] = self.attraction[others]*((1.0-self.attraction[choice])/sum(self.attraction[others]))
            # Update Aspiration
            self.aspiration = self.aspiration*(1.0-self.tau) + payoff*self.tau


# ## Bandit  
# The environment is composed of an n-arm bandit. Each arm of the bandit is definied by an object of class bandit. Put together these objects create one object of class bandits.  
# The bandit is a simple structure. It has a mean and a standard deviation. The style determines whether the bandit outputs an uniformly distributed value, a normally distributed value, or one without noise. A bandit is the building block of the n-arm bandits of every paper we replicate in this study.  
# 
# ### Measure
# The bandits perform one function, when called upon, they give one output, centered around a mean value and with an added noise. The style of bandit determines where the noise is drawn upon.

# In[7]:


class bandit:
    def __init__(self, mean, noise, style):
        self.mean = mean
        self.noise = noise
        self.style = style
    def measure(self):
        if self.style == "Uniform":  value = self.mean + self.noise*(np.random.random() - 0.5)
        elif self.style == "Normal": value = np.random.normal(loc = self.mean, scale = self.noise)
        elif self.style == "Beta": value = np.random.binomial(n = 1, p = self.mean)
        elif self.style == "Stable": value = self.mean
        return(value)


# 
# ## Environments for the different papers
# Below you will find three sets of n-arm bandits one for every paper we replicate in this document.  
# 
# ### 1. Bandits_D_M: Denrell and March (2001)
# This class creates the environment for the Denrell and March (2001) paper. In specific, two bandits. One with stable output and one with variable output. Both bandits have the same mean.
# 
# #### Measure
# This is a wrapper function. The objective is that the agents ask the bandits class and not the specific bandit for the measurement. Then the bandits class is in charge of asking its bandit for the performance value. 

# In[3]:


class bandits_D_M:
    def __init__(self, means, noise): 
        self.means = means
        self.arms = [bandit(mean = means[0], noise = noise, style = "Normal"),
                     bandit(mean = means[1], noise = 0.0, style = "Stable")]
        self.means = np.zeros(len(self.arms))
        for i in range(len(self.arms)): self.means[i] = self.arms[i].mean
    def measure(self, choice): return(self.arms[choice].measure())
    def reset(self): pass # needed for compatibility


# ###  2. Bandits_P_L: Posen and Levinthal (2012)
# This class creates the environment for the Posen and Levinthal (2012) paper. In specific, 10 bandits with probabilities of drawing 1s drawn from a Beta(2,2) distribution.
# 
# #### Measure
# This is a wrapper function. The objective is that the agents ask the bandits class and not the specific bandit for the measurement. Then the bandits class is in charge of asking its bandit for the performance value.  
# The second role of this function is to determine if a bandit needs to be changed and if so change it. This process is what Posen and Levinthal call turbulence and is determined by the eta parameter. 
# 
# #### Make Bandit
# This function creates the bandits. As in this model the bandits need to be remade on every iteration, we need a simple way of remaking the bandits. These function makes a new bandit and stores the mean value of its output.  
# 
# #### Reset
# This function makes new bandits for every arm of the bandit. 

# In[ ]:


class bandits_P_L:
    def __init__(self, num_bandits, eta=0.0):
        self.eta = eta
        self.means = np.zeros(num_bandits)
        self.arms = ['']*num_bandits
        for i in range(num_bandits): self.make_bandit(i)
    def make_bandit(self, position):
        self.means[position] = np.random.beta(a=2.0, b=2.0)
        self.arms[position] = bandit(mean = self.means[position], noise = 0.0, style = "Beta")
    def measure(self, choice):
        # Change some arms?
        if np.random.binomial(n=1, p = self.eta):
            for i in range(len(self.arms)):
                if np.random.binomial(n=1, p = 0.5): self.make_bandit(i)
        return(self.arms[choice].measure())
    def reset(self): 
        for i in range(len(self.arms)): self.make_bandit(i)


# ###  3. Bandits_P_S: Puranam and Swamy (2016)
# This class creates the environment for the Puranam and Swamy (2012) paper. In specific, 4 bandits with two entries each and with means specified from game thoerretic values.
# 
# #### Measure
# The bandits in Puranam and Swamy are much simpler than our prior examples. The model does have a complication due to the coupled learning but we will present this later on. 
# #### Make Bandit
# There are technically just two bandits, even though Puranam and Swamy model m=10. One bandit gives a high output when both choices are the same and accurate, and if not it gives a low value. 
# #### Reset
# This function does not operate. 

# In[215]:


class bandits_P_S:
    def __init__(self, num_bandits, bingo, val_max, val_min): 
        self.means = val_min*np.ones(num_bandits)
        self.means[bingo] = val_max
        self.arms = [bandit(mean = val_max, noise = 0.0, style = "Stable"),
                     bandit(mean = val_min, noise = 0.0, style = "Stable")]
        self.bingo = bingo
    def measure(self, choice1, choice2): 
        if choice1 == choice2 and choice1 == self.bingo: return(self.arms[0].measure())
        else: return(self.arms[1].measure())
    def reset(self): pass # needed for compatibility


# ## Run Simulation
# This tutorail runs several simulations. All of them follow the same structure, they include agents, and environments. The agents explore the environment for a number of periods and then this is repeated again. The iterated process is stored and used for our analyses.  
# This method automates this process to avoid repeating every time. 

# In[173]:


def run_simulation(num_reps, num_periods, Alice, options, start_attraction, n=1):
    all_payoffs = np.zeros((num_periods))
    all_knowledge = np.zeros((num_periods,n))
    all_RE = np.zeros((num_periods,n))
    all_choices = np.zeros(num_periods)
    all_aspiration = [0.0]*n
    last_choices = []
    for j in range(num_reps):
        Alice.reset(means = options.means, att = start_attraction)     
        options.reset()
        choice, payoff, knowledge, re = Alice.learn(num_periods, options)
        all_payoffs = np.add(all_payoffs, payoff)
        all_knowledge = np.add(all_knowledge, knowledge)
        all_RE = np.add(all_RE, re)
        # Specific for this paper
        all_choices = np.add(all_choices, choice)
        if Alice.style_choose == "aspiration": all_aspiration += Alice.aspiration
        last_choices.append(choice[-1])
    return([all_payoffs, all_knowledge, all_RE, all_choices, all_aspiration, last_choices])


# 
# # Denrell and March (2001)
# 
# With these two building blocks, we can run a simulation to replicate the main finding of Denrell and March (2001).
# 
# ## 1. Initialize values
# We start by initailizing the attributes of the simulation. The agents are given a set of tau and phi. The agents will learn for 50 periods and the results replicated 2500 times. We specify the noise to be 1, that means the bandits will draw from values between 0 and 1. Changes in the tau, phi, noise, and bandit style should change the learning. Changes in the number of repetitions lead to more noisy results.
# 
# **Note:** The bandits were specificed as outputing 0 and 1. Posen and Denrell mix their outputs as their environment outputs -1 and 1. I go back to that notation when I estimate the accumulated payoff. However for the bulk of the simulation, I estimate the probability of success only.

# In[6]:


## Bandits
X = 10.0
Y = 10.0
S = 10.0
num_bandits = 2
## Agents
style_update = "aspiration" # "constant", "over k", or "aspiration"
style_choose =  "aspiration" # "softmax", "greedy", "aspiration" or e value as float for e-greedy
a = 0.5
b = 0.5
start_attraction = np.ones(num_bandits)/num_bandits # can use a list of values
## Simulation
num_periods = 50
num_reps = 5000


# ## 2. Initialize agent and Bandits
# We create one agent, Alice and initialize the environment for the paper. The bandits are created by specifying first two agents one drawn from an uniform distribution and the second one from a stable value.

# In[7]:


Alice = agent(tau = a, phi = b, style_update = style_update, style_choose = style_choose)
options = bandits_D_M(means = [X,Y], noise = S)


# ## 3. Run simulation
# Having the agent and environment we can run a simulation. We give the environment and the agents and run it.  
# 
# * **This takes some time.**

# In[8]:


payoffs, knowledge, RE, choices, aspiration, last_choices = run_simulation(num_reps, num_periods, Alice, options, start_attraction)


# ## 4. Display results
# 
# ### Choice as function of time
# We present two plots. The first one presents the option chosen on every period. As on every period the agent can choose 0 or 1, what we plot in the y-axis is the number of times the stable option is chosen. As expected, the first period starts at 50% of the time and it increases towards a 100% as time goes by.
# 

# In[9]:


plt.scatter(range(num_periods), choices)


# ### Performance as function of time
# The second graph presents the average payoff. This looks like a funnel, narrowing from left to right. As the stable option is chosen more and more, the variance in the performanc decreases. 

# In[10]:


plt.scatter(range(num_periods), payoffs)


# ### Summary Values  
# #### Fraction of individuals who choose the risky alternative
# Both options have the same performance. Nonetheless, the risky option is chosen less than 1% of the time after 50 periods when b = 0.5, and around 30% of time if b = 0.1. X=Y=S=10.0 

# In[11]:


100*(1-float(sum(last_choices))/num_reps)


# #### Average aspiration level at the end of each simulation   
# The average aspiration at the end of each simulation was:

# In[12]:


aspiration/num_reps


# ## 5. Exercise
# Find the how high the mean of the variable option needs to be in order to be chosen 50% of the time at the end of the simulation. How does it related to the amount of noise in the option? How does it change if normal and not uniform noise is used? Or if we use the 1/(k+1) updating instead of constant updating?

# # Posen and Levinthal (2012)
# With these two building blocks, we can run a simulation to replicate the main findings of Posen and Levinthal (2021).
# 
# ## 1. Initialize values
# We start by initailizing the attributes of the simulation. The agents are given a tau but do not require phi because the learning method follows the 1/k+1 weighting. The agents will learn for 500 periods and the results replicated 1000 times. There is no noise value here because in this paper the bandits either output a positive value or not. Changes in the tau, and bandit style should change the learning. Changes in the number of repetitions lead to more noisy results.
# 
# **Note:** A phi value was added in case you want to explore how agents would differ if the do not update with the 1/k+1 mode.  

# In[13]:


## Bandit
eta = 0.0
num_bandits = 10
## Agents
style_update = "over k" # "constant", "over k" or "aspiration" 
style_choose = "softmax" # "softmax", "greedy", "aspiration", or e value as float for e-greedy
phi = 0.5 # not needed in "over k" updating mode
tau = 0.5/num_bandits
start_attraction = np.ones(num_bandits)/2.0
## Simulation
num_periods = 500
num_reps = 1000


# ## 2. Initialize agent and Bandits
# Having the agent and environment we can run a simulation. We give the environment and the agents and run it. 

# In[14]:


Alice = agent(tau = tau, phi = phi, style_update = style_update, style_choose = style_choose)
options = bandits_P_L(num_bandits = num_bandits, eta = eta)


# ## 3. Run simulation
# Having the agent and environment we can run a simulation. We initialize two arrays, one for payoff and one for choices. Additionally, we create an empty list to store the last choices of the agents and one value to save the attraction to the option with variable output after every replication of the simulation is finished.
# 
# This takes some time.

# In[15]:


payoffs, knowledge, RE, choices, aspiration, last_choices = run_simulation(num_reps, num_periods, Alice, options, start_attraction)


# ## 4. Display results
# 
# ### Amount of exploration
# First we present the amount of exploration done by the agents. 

# In[16]:


plt.scatter(range(num_periods), RE)


# ### Knowledge over time
# Somthing quite sad happens for the amount of knowledge over time in this paper. Given the way the Bush Mossteller equation is updated, 1/k+1 and not with a constant update percentage, initial values have much more weight that later values. This leads to the system to erode knowledge. **Fast!**

# In[17]:


plt.scatter(range(num_periods), knowledge)


# ### Probability of Getting a Reward
# The second graph presents the average payoff. This looks like a funnel, narrowing from left to right. As the stable option is chosen more and more, the variance in the performanc decreases. 

# In[18]:


plt.scatter(range(num_periods), payoffs)


# ### Summary variables
# #### Total accumulated payoff
# The bandits in this paper give a payoff of 1 or -1, so one needs to recalculate the total payoff. So that it matches what is shown in the paper. In the simulation I store the probability of getting a 1, here, the actual expected value.
# 
# The result shown below is the performance of Figure 1 in Posen and Levinthal (2012).

# In[19]:


print(sum(2*payoffs-num_reps)/num_reps)


# #### Fraction of exploration events
# The average percentage of exploration events during the 500 periods, see Figure 1 in Posen and Levinthal (2012) for comparison. 

# In[20]:


print(sum(RE)/(num_reps*num_periods))


# #### Knowledge  
# The average SSE knowledge at different stages of the simulation. This result is shown in Figure 1 of Posen and Levinthal (2012) and in Figure 4 when the turbulence changes.

# In[21]:


print("Period 400: " + str(knowledge[-101]/num_reps))
print("Period 500: " + str(knowledge[-1]/num_reps))


# ## 5. Exercise
# What would happen if Posen and Levinthal had chosen other learning rules? They studied e-greedy but how about constant update? How woud it affect the erosion of knowledge and the adaptation of the agents?  

# # Puranam and Swamy (2016)
# In contrast to the prior papers, Phanisah and Maurati a coupled learning process where the decisions of one agent interact with the decisions of another to achieve high performance.  
# The fact that we now have two agents instead of one leads to create a new class. THis class is represents the organization and it is a wrapper structure so that when we run the simulation, the simulation function believes as if it is interacting with one agent, even though there are two.  
# 
# ## 1. Organization class
# ### 1.1 Initialization
# The organization class is initialized with two agents, Alice and Bob. 
#   
# ### 1.2 Learn function
# The learn function is functionally equivalent to the learning function of the individual agents. However, there are replicated values stored. This is due to the fact that we need each agent to make a choice. The two choices are then sent to the bandit to get one payoff. The oayoff then is used to update the agent's attractions. Finally, we store instead of the choices, only whether the agents had the same choice in the specific period. We also store duplicates of the knowledge SSE per period and the exploration in the period. 
#   
# ### 1.3 Reset
# The reset function is different from the prior cases. Here the values used to initialize the attractions at the start of every simulation are different for each agent. The agents in Puranam and Swamy (2016) have their attractions directly initialized every time the simulation starts. Each bandit gets a different and specific attraction. In contrast to the prior papers where beliefs at the start of the simulation for every bandit were homogenous, here the key aspect Phanish and Murati study is how differences in the initial representation affect coordination and thus require that the agents have very specific starting beliefs. 

# In[448]:


class organization():
    def __init__(self, Alice, Bob):
        self.Alice = Alice
        self.Bob = Bob
        self.style_choose = self.Alice.style_choose
    def learn(self, num_periods, bandits):
        choices = []
        coordination = []
        payoffs = []
        knowledge = []
        RE = [[1,1]]
        for i in range(num_periods):
            choice1 = self.Alice.choose()
            choice2 = self.Bob.choose()
            payoff = bandits.measure(choice1, choice2)
            coordinate = 1*(choice1==choice2 and choice1 == np.argmax(bandits.means))
            self.Alice.update(choice1, payoff)
            self.Bob.update(choice2, payoff)
            nugget1 = 1-sum((self.Alice.attraction-bandits.means)**2)
            nugget2 = 1-sum((self.Bob.attraction-bandits.means)**2)
            choices.append([choice1, choice2])
            payoffs.append(payoff)
            knowledge.append([nugget1, nugget2])
            coordination.append(coordinate)
            if len(choices) > 1:
                re1 = 1*(choices[-1][0] != choices[-2][0])
                re2 = 1*(choices[-1][1] != choices[-2][1])
                RE.append([re1, re2])
        if self.style_choose == "aspiration":
            self.aspiration = [self.Alice.aspiration, self.Bob.aspiration]
        return([coordination, payoffs, knowledge, RE])
    def reset(self, means, att):
        self.Alice.reset(means, att[0])
        self.Bob.reset(means, att[1])


# ## 2. Initialize values
# In this simulation we need more parameters. We start with the bandit.
# 
# ### 2.1 Bandit
# Tehr are ten bandits. The bandits have a maximum value that appears when both agents choose option 2 (bingo!). IF not then a minimum value is given. Here, in contrast to Posen and Levinthal, I input the maximum and minimum values used by Phanish and Murati.

# In[449]:


num_bandits = 10
val_max = 1.0
val_min = -1.0
bingo = 2
flop = 9


# ### 2.2 Agents
# the agents have constant phi values, and use softmax. This paper has the peculiarity that the agent's attractions are initialized at the start of the simulation to precise values. For this reason it uses the style_start = "fixed". That allows the organization to set the beliefs every time the simulation starts again.  
# In addition to tau and phi, in this paper there is a p value that represent how certain the agents are of their initial believes (1 for full certainty, 0 for no certainty).

# In[450]:


style_update = "constant" # "constant", "over k" or "aspiration" 
style_choose = "softmax" # "softmax", "greedy", "aspiration", or e value as float for e-greedy
style_start = "fixed"
phi = 0.25 # not needed in "over k" updating mode
tau = 0.1
p = 0.8


# ### 2.3 Simulation
# In the paper, the simulation is run for 100 periods and 5k replications. Here, I do the same but if you run it online, it might be better to run less. The simulation is quite noisy as it has Bernoulli bandits but given that there are two agents, it takes much longer to run.

# In[459]:


num_periods = 100
num_reps = 5000 # 5000 typical


# #### Coupled learning
# The agents have opinions about the bandits'payoffs before every simulation. How strong or weak the opinion is depends on the parameter p.   
# The belief are negative for every but one arm of the N-arm bandit. The chosen arm has a high belief. There are three different ways to initialize an agent, with good beliefs, bad beliefs or uniform beliefs. The combinations of these three are used to understand the performance implications of the initial beliefs in an organization. 

# In[460]:


max_setting = (p*val_max + (1-p)*val_min)
min_setting = (p*val_min+(1-p)*val_max)
# Good beliefs
good = min_setting*np.ones(num_bandits)
good[bingo] = max_setting
# Bad beliefs
bad = min_setting*np.ones(num_bandits)
bad[flop] = max_setting
# Uniform beliefs
uniform = max_setting*np.ones(num_bandits)/num_bandits
# Organization settings
org_setting1 = [bad, bad]
org_setting2 = [uniform, uniform]
org_setting3 = [good, bad]


# ## 3. Initialize agent and Bandits
# Having the organization and environment we can run a simulation. We give the environment and the organization runs it by asking each agetn for its choices. 

# In[461]:


Alice = agent(tau = tau, phi = phi, style_update = style_update, style_choose = style_choose, style_start = "fixed")
Bob = agent(tau = tau, phi = phi, style_update = style_update, style_choose = style_choose, style_start = "fixed")
Inc = organization(Alice = Alice, Bob = Bob)
options = bandits_P_S(num_bandits = num_bandits, bingo = bingo, val_max = val_max, val_min = val_min)


# ## 4. Run simulation
# W3 run three simulation. One for each organizational setting. This is usefulto get the one plot of Figure 2a and 2b at once. Given that to make the plot we need two simulation, a third does not increase the time too much. It does take long. 

# In[462]:


payoffs, kn, RE, coordination, asp, last = run_simulation(num_reps, num_periods, Inc, options, org_setting1, 2)
payoffs2, kn2, RE2, coordination2, asp2, last2 = run_simulation(num_reps, num_periods, Inc, options, org_setting2, 2)
payoffs3, kn3, RE3, coordination3, asp3, last3 = run_simulation(num_reps, num_periods, Inc, options, org_setting3, 2)


# ### 5. Display results: Relative coordination
# We present just the results shown on Figures 2a and 2b. These results relate to the percentage of times the agents chose the correct answer. However, there is more data available in case you interested. We log the amount of exploration each agent does, the accuracy of their knowledge, the amount of coordinatation (i.e. times the chose the correct bandit), the aspiration levels, and last choices. 

# In[463]:


plt.scatter(range(num_periods), (coordination-coordination2)/num_reps, c = "blue")


# In[464]:


plt.scatter(range(num_periods), (coordination-coordination3)/num_reps, c = "red")


# # References
# 
# * Denrell, J., & March, J. G. (2001). Adaptation as information restriction: The hot stove effect. Organization Science, 12(5), 523-538.   
# * Posen, H. E., & Levinthal, D. A. (2012). Chasing a moving target: Exploitation and exploration in dynamic environments. Management Science, 58(3), 587-601.
# * Puranam, P., & Swamy, M. (2016). How initial representations shape coupled learning processes. Organization Science, 27(2), 323-335.
# 
# **Note:** The code below builds the table of contents

# In[1]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")

