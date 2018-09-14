#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#This code buids upon the repo @ https://github.com/johnmyleswhite/BanditsBook
#We'd like to thank the author for such clean implementations
#Make Sure you install these dependencies

import math
import numpy as np
import heapq
import random

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#Returns the generator for creating steps given a start and an end 
def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

#categorical_draw function for the EXP3 algorithm
def categorical_draw(probs):
  z = random.random()
  cum_prob = 0.0
  for i in range(len(probs)):
    prob = probs[i]
    cum_prob += prob
    if cum_prob > z:
      return i

  return len(probs) - 1

#Bernoulli KL Divergence for KL-UCB algorithm
def bernoulli_KL_divergence(p,q):
  if p==0 and q==0:
  	return 0
  if p==0 and q!=0:
  	return (p*math.log((p+1e-6)/float(q)) + abs((1-p))*math.log(abs((1-p))/(float(abs(1-q)))))
  if p > 1:
  	return (1*math.log(1/float(q)) + (1e-8)*math.log(1e-8/(float(1))))	
  return (p*math.log(p/float(q)) + (1-p)*math.log((1-p)/(float(abs(1-q)))))

#Bernoulli KL Divergence newton iterator
def bernoulli_KL_grad(p,q):
  return (q-p)/float(q*(1-q))

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#EXP3 algroithm: An algorithm for adversarial bandits, used to demonstrate that the corruption task is not suitably handled by such algorithms
#The class functions are self-explanatory please refer for details on implementing bandit algorithms @ https://github.com/johnmyleswhite/BanditsBook
class Exp3():
  def __init__(self, weights, gamma):
    self.gamma = gamma
    self.weights = weights
    return
  
#INitialize the arms
  def initialize(self, n_arms):
    self.weights = np.array([1.0 for i in range(n_arms)])
    return
  
#Select the appropriate arms according to the algorithm
  def select_arm(self):
    n_arms = len(self.weights)
    total_weight = sum(self.weights)
    probs = [0.0 for i in range(n_arms)]
    for arm in range(n_arms):
      probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
      probs[arm] = probs[arm] + (self.gamma) * (1.0 / float(n_arms))
    return categorical_draw(probs)
  
#Update your means for the arms
  def update(self, chosen_arm, reward):
    n_arms = len(self.weights)
    total_weight = sum(self.weights)
    probs = [0.0 for i in range(n_arms)]
    for arm in range(n_arms):
      probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
      probs[arm] = probs[arm] + (self.gamma) * (1.0 / float(n_arms))
    
    x = reward / probs[chosen_arm]
    
    growth_factor = math.exp((self.gamma / n_arms) * x)
    self.weights[chosen_arm] = self.weights[chosen_arm] * growth_factor


#Thompson Sampling Algorithm: Implemented by the authors over https://github.com/johnmyleswhite/BanditsBook>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class TS():
  def __init__(self, counts, values):
    self.counts = counts
    self.values = values
    return
  
  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return
  
  def select_arm(self):
    n_arms = len(self.counts)
    for arm in range(n_arms):
      if self.counts[arm] == 0:
        return arm
    ts_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.counts)
    for arm in range(n_arms):
      s = np.random.randn()
      bonus = s*(1/float(self.counts[arm]))
      ts_values[arm] = self.values[arm] + bonus
    return np.argmax(ts_values)
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]

    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value
    return


#Vanilla UCB1 algorithm as described in Auer et al with an alterable rho value to be specified in the class constructor call>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class UCB1():
  def __init__(self, counts, values, rho):
    self.counts = counts
    self.values = values
    self.rho = rho
    return
  
  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return
  
  def select_arm(self):
    n_arms = len(self.counts)
    for arm in range(n_arms):
      if self.counts[arm] == 0:
        return arm

    ucb_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.counts)
    for arm in range(n_arms):
      bonus = math.sqrt((self.rho * math.log(total_counts)) / float(self.counts[arm]))
      ucb_values[arm] = self.values[arm] + bonus
    return np.argmax(ucb_values)
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]

    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value
    return

#UCBV algorithm as described in Audivert et al, one of the first analysis of variance estimating bernstein inequality based algorithms>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class UCBV():
  def __init__(self, counts, values, squares, rho):
    self.counts = counts
    self.values = values
    self.squares = squares
    self.rho = rho
    return
  
  def initialize(self, n_arms):
    self.counts = [1 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    self.squares = [0.0 for col in range(n_arms)]
    return
  
#Compare the variance dependent exploration term with vanilla UCB1
  def select_arm(self):
    n_arms = len(self.counts)
    total_counts = sum(self.counts)
    for arm in range(n_arms):
      if self.counts[arm] == 1:
        return arm

    ucb_values = [0.0 for arm in range(n_arms)]
    for arm in range(n_arms):
      V = (self.squares[arm] / float(self.counts[arm])) - self.values[arm]**2
      bonus = math.sqrt((self.rho * math.log(total_counts) * V) / float(self.counts[arm])) + (8*math.log(total_counts))/(3*float(self.counts[arm]))
      ucb_values[arm] = self.values[arm] + bonus
    return np.argmax(ucb_values)
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]

    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value

    square = self.squares[chosen_arm]
    new_square = square + float(reward) * reward
    self.squares[chosen_arm] = new_square
    return


#UCB Normal algorithm as described in Auer et al for normal distributions>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class UCB1_normal():
  def __init__(self, counts, values, squares):
    self.counts = counts
    self.values = values
    self.squares = squares
    return
  
  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    self.squares = [0.0 for col in range(n_arms)]
    return
  
#Compare the variance dependent exploration term with vanilla UCB1
  def select_arm(self):
    n_arms = len(self.counts)
    total_counts = sum(self.counts)
    for arm in range(n_arms):
      if self.counts[arm] == 0:
        return arm
      if self.counts[arm] < math.ceil(8 * math.log(total_counts)/math.log(2)):
        return arm

    ucb_values = [0.0 for arm in range(n_arms)]
    for arm in range(n_arms):
      bonus = 4 * math.sqrt(math.fabs(((self.squares[arm] - self.counts[arm] * self.values[arm] * self.values[arm]) * (math.log(total_counts - 1))) / ( float(self.counts[arm]) * float(self.counts[arm]) ) ))
      ucb_values[arm] = self.values[arm] + bonus
    return np.argmax(ucb_values)
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]

    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value

    square = self.squares[chosen_arm]
    new_square = square + float(reward) * reward
    self.squares[chosen_arm] = new_square
    return


#UCB Tuned algorithm which is self tuning as described in Auer et a; performs better than UCB1 except when an optimal tuning is known for the same>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class UCB1_tuned():
  def __init__(self, counts, values, squares, var_bound):#var_bound corresponds to the bound on variance here.
    self.counts = counts
    self.values = values
    self.var_bound = var_bound
    self.squares = squares
    return
  
  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    self.squares = [0.0 for col in range(n_arms)]
    return
  
#Notice the dfference in the exploration metric
  def select_arm(self):
    n_arms = len(self.counts)
    for arm in range(n_arms):
      if self.counts[arm] == 0:
        return arm

    ucb_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.counts)
    for arm in range(n_arms):
      V = (self.squares[arm] / float(self.counts[arm])) - self.values[arm]**2 + math.sqrt( 2 * math.log(total_counts) / float(self.counts[arm])) 
      bonus = math.sqrt((math.log(total_counts) / float(self.counts[arm])) * min(self.var_bound, V))
      ucb_values[arm] = self.values[arm] + bonus
    return np.argmax(ucb_values)
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]

    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value

    square = self.squares[chosen_arm]
    new_square = square + float(reward) * reward
    self.squares[chosen_arm] = new_square
    return


#Our Algorithm the rUCB or the "Robust UCB", that uses median instead of the mean for exploitation and has a different exploration term; does depend on variance though; not self tuning
class rUCB1():
  def __init__(self, counts, lower_values, upper_values, est, rho):
    self.counts = counts
    self.lower_values = lower_values
    self.upper_values = upper_values
    self.est = est
    self.rho = rho
    return
  
  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.lower_values = [[] for col in range(n_arms)]
    self.upper_values = [[] for col in range(n_arms)]
    self.est = [0 for col in range(n_arms)]
    return

  def select_arm(self, err):
    n_arms = len(self.counts)
    for arm in range(n_arms):
      if self.counts[arm] == 0:
        return arm
    ucb_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.counts)
    for arm in range(n_arms):
      bonus = (math.sqrt(self.rho *(math.log(total_counts)) / float(self.counts[arm])) + (math.sqrt(self.rho))*err)
      if len(self.lower_values[arm]) > len(self.upper_values[arm]):
        ucb_values[arm] = -1*self.lower_values[arm][0]
      else:
        ucb_values[arm] =  self.upper_values[arm][0]
      ucb_values[arm] += bonus
    return np.argmax(ucb_values)
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    if len(self.lower_values[chosen_arm]) == 0 or reward < -1*self.lower_values[chosen_arm][0]:
      heapq.heappush(self.lower_values[chosen_arm], -1*reward)
    else:
      heapq.heappush(self.upper_values[chosen_arm], reward)
    if len(self.lower_values[chosen_arm]) > len(self.upper_values[chosen_arm]) + 1:
      heapq.heappush(self.upper_values[chosen_arm], -1*heapq.heappop(self.lower_values[chosen_arm]))
    elif len(self.upper_values[chosen_arm]) > len(self.lower_values[chosen_arm]) + 1:
      heapq.heappush(self.lower_values[chosen_arm], -1*heapq.heappop(self.upper_values[chosen_arm]))
    if len(self.lower_values[chosen_arm]) > len(self.upper_values[chosen_arm]):
      self.est[chosen_arm] = -1*self.lower_values[chosen_arm][0]
    else:
      self.est[chosen_arm] =  self.upper_values[chosen_arm][0]
    return



#UCB2 algorithm as described in Auer et al. Implementation of the slides @ http://lane.compbio.cmu.edu/courses/slides_ucb.pdf>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    
class UCB2(object):
  def __init__(self, alpha, counts, values):
    
    self.alpha = alpha
    self.counts = counts
    self.values = values
    self.__current_arm = 0
    self.__next_update = 0
    return
  
  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    self.r = [0 for col in range(n_arms)]
    self.__current_arm = 0
    self.__next_update = 0
  
  def __bonus(self, n, r):
    tau = self.__tau(r)
    bonus = math.sqrt((1. + self.alpha) * math.log(math.e * float(n) / tau) / (2 * tau))
    return bonus
  
  def __tau(self, r):
    return int(math.ceil((1 + self.alpha) ** r))
  
  def __set_arm(self, arm):
    self.__current_arm = arm
    self.__next_update += max(1, self.__tau(self.r[arm] + 1) - self.__tau(self.r[arm]))
    self.r[arm] += 1
  
  #notice the difference with UCB1 Algorithm
  def select_arm(self):
    n_arms = len(self.counts)
    
    # play each arm once
    for arm in range(n_arms):
      if self.counts[arm] == 0:
        self.__set_arm(arm)
        return arm
    
    # make sure we aren't still playing the previous arm.
    if self.__next_update > sum(self.counts):
      return self.__current_arm
    
    ucb_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.counts)
    for arm in range(n_arms):
      bonus = self.__bonus(total_counts, self.r[arm])
      ucb_values[arm] = self.values[arm] + bonus
    
    chosen_arm = np.argmax(ucb_values)
    self.__set_arm(chosen_arm)
    return chosen_arm
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]
    
    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value

#Robust UCB tuned algorithm; it uses a variance estimator exploration term which looks similar to UCB tuned but uses median instead of mean>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class rUCBT():
  def __init__(self, counts, lower_values, upper_values, values, lower_deviations, upper_deviations, est, est_deviation, rho):
    self.counts = counts
    self.lower_values = lower_values
    self.upper_values = upper_values
    self.lower_deviations = lower_deviations
    self.upper_deviations = upper_deviations
    self.est = est
    self.est_deviation = est_deviation
    self.rho = rho
    self.values = values
    return
  
  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [[] for col in range(n_arms)]
    self.lower_values = [[] for col in range(n_arms)]
    self.upper_values = [[] for col in range(n_arms)]
    self.lower_deviations = [[] for col in range(n_arms)]
    self.upper_deviations = [[] for col in range(n_arms)]
    self.est = [0 for col in range(n_arms)]
    self.est_deviation = [0 for col in range(n_arms)]
    return

  def select_arm(self, err):
    n_arms = len(self.counts)
    for arm in range(n_arms):
      if self.counts[arm] == 0:
        return arm
    ucb_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.counts)
    
    for arm in range(n_arms):
      if len(self.lower_values[arm]) > len(self.upper_values[arm]):
        ucb_values[arm] = -1*self.lower_values[arm][0]
      else:
        ucb_values[arm] =  self.upper_values[arm][0]

      if len(self.lower_deviations[arm]) > len(self.upper_deviations[arm]):
        deviation = -1*self.lower_deviations[arm][0]
      else:
        deviation =  self.upper_deviations[arm][0]
      bonus = math.sqrt((math.log(total_counts) / float(self.counts[arm]) + err)*self.rho*(deviation*1.4826))
      ucb_values[arm] += bonus
    return np.argmax(ucb_values)
  

#Heap based median finding for optimization
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    self.values[chosen_arm].append(reward)
    n = self.counts[chosen_arm]
    
    if len(self.lower_values[chosen_arm]) == 0 or reward < -1*self.lower_values[chosen_arm][0]:
      heapq.heappush(self.lower_values[chosen_arm], -1*reward)
    else:
      heapq.heappush(self.upper_values[chosen_arm], reward)
    if len(self.lower_values[chosen_arm]) > len(self.upper_values[chosen_arm]) + 1:
      heapq.heappush(self.upper_values[chosen_arm], -1*heapq.heappop(self.lower_values[chosen_arm]))
    elif len(self.upper_values[chosen_arm]) > len(self.lower_values[chosen_arm]) + 1:
      heapq.heappush(self.lower_values[chosen_arm], -1*heapq.heappop(self.upper_values[chosen_arm]))
    if len(self.lower_values[chosen_arm]) > len(self.upper_values[chosen_arm]):
      self.est[chosen_arm] = -1*self.lower_values[chosen_arm][0]
    else:
      self.est[chosen_arm] =  self.upper_values[chosen_arm][0]
    
    if len(self.lower_values[chosen_arm]) > len(self.upper_values[chosen_arm]):
    	median = -1*self.lower_values[chosen_arm][0]
    else:
    	median =  self.upper_values[chosen_arm][0]

    if len(self.lower_deviations[chosen_arm]) == 0 or reward < -1*self.lower_deviations[chosen_arm][0]:
      heapq.heappush(self.lower_deviations[chosen_arm], -1*abs(reward - median))
    else:
      heapq.heappush(self.upper_deviations[chosen_arm], abs(reward - median))
    if len(self.lower_deviations[chosen_arm]) > len(self.upper_deviations[chosen_arm]) + 1:
      heapq.heappush(self.upper_deviations[chosen_arm], -1*heapq.heappop(self.lower_deviations[chosen_arm]))
    elif len(self.upper_deviations[chosen_arm]) > len(self.lower_deviations[chosen_arm]) + 1:
      heapq.heappush(self.lower_deviations[chosen_arm], -1*heapq.heappop(self.upper_deviations[chosen_arm]))
    if len(self.lower_deviations[chosen_arm]) > len(self.upper_deviations[chosen_arm]):
      self.est_deviation[chosen_arm] = -1*self.lower_deviations[chosen_arm][0]
    else:
      self.est_deviation[chosen_arm] =  self.upper_deviations[chosen_arm][0]


    return 

# KL-UCB algorithm as described in Gariveier et al. >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class KLUCB():
  def __init__(self, counts, values, c):
    self.counts = counts
    self.values = values
    self.c = c
    return

  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return

  def select_arm(self):
    n_arms = len(self.counts)
    for arm in range(n_arms):
      if self.counts[arm] == 0:
        return arm

    delta = 1e-8
    eps = 1e-12
    max_iter = 10 #some constants for newton iterations
    kl_ucb_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.counts)

    for arm in range(n_arms):
      p = self.values[arm]
      q = self.values[arm] + delta
      upperbound = (math.log(total_counts) + self.c * math.log(math.log(total_counts)))/self.counts[arm]
      for iteration in range(0, max_iter):
        f = upperbound - bernoulli_KL_divergence(p,q)
        if(f*f < eps):
          break
        df = - bernoulli_KL_grad(p,q)
        q = min(1 - delta, max(q - f/df, p + delta))

      kl_ucb_values[arm] = q

    return np.argmax(kl_ucb_values)

  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]
    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value
    return


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# SAO algorith as described in Bubeck et al. >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#@sayash write the algorithm here. I explained you the code before.

class SAO():
  def __init__(self, beta, tau, n, A, p, q, counts, values, weights, Htilde):
    self.beta = beta
    self.p = p
    self.q = q
    self.tau = tau
    self.n = n
    self.A = A
    self.counts = counts
    self.values = values
    self.weights = weights
    self.Htilde = Htilde
    return
  
#Initialize the arms
  def initialize(self, n_arms, n):
    self.p = np.array([1.0/n_arms for i in range(n_arms)])
    self.tau = np.array([n for i in range(n_arms)])
    self.A = np.array([1 for i in range(n_arms)])
    self.q = np.array([0 for i in range(n_arms)])
    self.counts = [1 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    self.weights = [1.0/n_arms for col in range(n_arms)]
    self.Htilde = [0.0 for i in range(n_arms)]
    # 1 denotes an arm is in A, 0 denotes an arm is not in A
    return
  
#Select the appropriate arm according to the algorithm
  def select_arm(self):
    n_arms = len(self.weights)
    total_weight = sum(self.weights)
    probs = [0.0 for i in range(n_arms)]
    for arm in range(n_arms):
      probs[arm] = self.weights[arm] / total_weight
    return categorical_draw(probs)

#Update your means for the arms
  def update(self, chosen_arm, reward):
    n_arms = len(self.weights)
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]
    total_counts = sum(self.counts)
    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value 
    for i in range(n_arms):
      self.Htilde[i] = ((total_counts - 1) / float(total_counts)) * self.Htilde[i]
    self.Htilde[chosen_arm] += float(reward)/(1e-15+self.p[chosen_arm])
    max_Htilde = np.max(self.Htilde)
    for i in range(n_arms):
      temp_val = n_arms*math.log(self.beta)/total_counts
      if self.A[i] and self.Htilde[i] < max_Htilde - 6*math.sqrt(4*temp_val + 5*(temp_val**2)):
        self.A[i] = 0
        self.q[i] = self.p[i]
        self.tau[i] = total_counts
      ti_star = min(self.tau[i], total_counts)
      if abs(self.Htilde[i] - self.values[i]) > (math.sqrt(2*math.log(self.beta)/(self.counts[i])) + math.sqrt(4*math.log(self.beta)*(n_arms*ti_star/(total_counts**2) + (total_counts - ti_star)/(1e-15+self.q[i]*self.tau[i]*total_counts)) + 5*((n_arms*math.log(self.beta)/(1e-15+ti_star)))**2)):
        return 1
      if self.A[i] == 0 and max_Htilde-self.Htilde[i] > 10*math.sqrt(4*n_arms*math.log(self.beta)/(self.tau[i]-1) + 5*((n_arms*math.log(self.beta)/(self.tau[i]-1))**2)):
        return 1
      if self.A[i] == 0 and max_Htilde-self.Htilde[i] < 2*math.sqrt(4*n_arms*math.log(self.beta)/(self.tau[i]-1) + 5*((n_arms*math.log(self.beta)/(self.tau[i]-1))**2)):
        return 1
    p_not_A_sum = 0
    for i in range(n_arms):
      if self.A[i] == 0:
        self.p[i] = self.q[i]*self.tau[i]/(total_counts+1)
        p_not_A_sum += self.p[i]
    for i in range(n_arms):
      if self.A[i] != 0:
        self.p[i] = (1/sum(self.A))*(1 - p_not_A_sum)
    return 0

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>