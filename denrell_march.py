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
            self.times[choice] += 1
            self.attraction[choice] += (payoff-self.attraction[choice])/(self.times[choice]+1)
    def choose(self):
        return(softmax(self.tau, self.attraction))
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
    def reset(self, num_bandits):
         self.attraction = np.ones(num_bandits)/num_bandits
         if self.style == "over k": self.times = np.zeros(num_bandits)

class bandit:
    def __init__(self, mu, stdev, style):
        self.style = style
        self.mean = mu
        self.stdev = stdev
    def measure(self):
        if self.style == "Uniform":  value = self.mean+self.stdev*(np.random.random()-0.5)
        elif self.style == "Normal": value = np.random.normal(loc=self.mean, scale=self.stdev)
        elif self.style == "Stable": value = self.mean
        return(value)
        
class bandits_D_M:
    def __init__(self, noise):
        self.arms = []
        self.arms.append(bandit(mu = 0.5, stdev = noise, style = "Uniform")) #0.622 equiprobability at 100 periods # 0.6392 0.5 expected value at 1000 periods
        self.arms.append(bandit(mu = 0.5, stdev = noise, style = "Stable"))
    def measure(self, choice):
        return(self.arms[choice].measure())
            
# Denrell, J., & March, J. G. (2001). Adaptation as information restriction: The hot stove effect. Organization Science, 12(5), 523-538.

## Values
### Bandits
noise = 1.0
### Agents
tau = 0.01/num_bandits
phi = 0.1
agent_style = "constant"
### Simulation
num_periods = 100
num_reps = 2500

## Initialize agents
Alice = agent(tau = tau, phi = phi, style = agent_style)
Alice.reset(num_bandits = 2)
## Initialize bandits
options = bandits_D_M(noise = noise)

## Run simulation
all_choices = np.zeros(num_periods)
all_payoffs = np.zeros(num_periods)
last_choices = []
all_attractions = 0.0
for j in range(num_reps):
     Alice.reset(num_bandits = num_bandits)      
     choice, payoff = Alice.learn(num_periods, options)
     all_choices += choice
     all_payoffs += payoff
     all_attractions += Alice.attraction[0]
     last_choices.append(choice[-1])

## Display results
plt.scatter(range(num_periods), all_choices)
plt.show()
plt.scatter(range(num_periods), all_payoffs)
plt.show()
print(all_attractions/num_reps)
print(100*float(sum(last_choices))/num_reps)
