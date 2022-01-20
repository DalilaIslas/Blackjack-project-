from collections import defaultdict
import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt


# Agent class can be seen as the blackjack player.
class Agent:
    def __init__(self):
        self.cards = [] # States

    def addCard(self, card):
        self.cards.append(card)

class Game:
    STICK = 0
    HIT = 1

    def __init__(self):
        self.agent = Agent()
        self.returns = defaultdict(list)
        self.valueFunctions = defaultdict(int)
        self.Q = defaultdict(dict)
        self.policies = defaultdict(dict)

        for i in range(1, 32):
            for k in range(0, 2):
                self.Q[i][k] = 0
                self.policies[i][k] = 0.5

    # Choose an action using optimal policy.
    def chooseAction(self, score):
        if score > 21:
            return self.STICK

        # Agent should always hit scores 2-11.   
        if score < 11:
            return self.HIT
        
        # Otherwise, we'll use the optimal policy.
        return np.random.choice(2, p = list(self.policies[score].values()))

    def finiteDeck(): 
        cards = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])

        deck = np.repeat(cards, 4)
        np.random.shuffle(deck)

        return deck.tolist()
    
    # Creates a shuffled deck and returns a random card.
    def getCard(self):
        deck = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
        return np.random.choice(deck)

    # Compute the current points of the agent.
    def sumPoints(self):
        result = sum(self.agent.cards)

        # Ace value.
        if result + 10 <= 21 and 1 in self.agent.cards:
            result += 10

        return result

    def getReward(self, score):  

        # Rewards for MC optimal results.
        #if score < 17:
        #    return 0
        #if score <= 21:
        #    return 2
        #return -1

        # Rewards for Q learning optional results.
         if score < 17:
             return 1
         if score < 21:
             return 2
         if score == 21:
             return 3
         return 0


    def playEpisode(self):
        self.agent = Agent()

        # Deal the first card
        self.agent.addCard(self.getCard())

        states = []
        while True:
            state = self.sumPoints()
            
            if state > 21:
                break

            action = self.chooseAction(state)

            if action == self.HIT:
                self.agent.addCard(self.getCard())
            
            nextState = self.sumPoints()
            reward = self.getReward(nextState)

            # Store state.
            states.append((state, action, reward))

            if action == self.STICK:
                break

        return states

    # On Policy Monte Carlo First Visit Control
    def monteCarloControl(self, episodes, gamma = 1, epsilon = 1.0):
        scores = []
        for i in range(episodes):
            print(f'{i}')
            states = self.playEpisode()

            # Epsilon decays
            epsilon = max(epsilon * 0.99995, 0.0001)

            score = 0
            G = 0
            for k, (state, action, reward) in enumerate(states):
                G = gamma * G + reward

                score += state

                if not (state, action) in states[0:k]:
                    self.returns[(state, action)].append(G)

                    self.Q[state][action] = np.mean(self.returns[(state, action)])

                    # Same as argmax but with dictionaries.
                    bestAction = max(self.Q[state], key=self.Q[state].get) 
                    
                    for action in range(0, 2):
                        if action == bestAction:
                            self.policies[state][action] = 1 - epsilon + (epsilon / 2) # |A(st)| = 2
                        else:
                            self.policies[state][action] = epsilon / 2

            # We'll store quadratic scores per episode.
            scores.append(score ** 2)

        return scores

    # Choose an action using Exploration vs Exploitation Trade-off 
    def chooseActionEGreedy(self, state, epsilon = 1):
        # We'll use the Q values based on epsilon.
        if np.random.uniform(0, 1) > epsilon:
            return max(self.Q[state], key=self.Q[state].get)

        # Otherwise, we'll make a random choice.
        return np.random.choice(2, p=[0.5, 0.5])

    # Q-Learning Off-Policy TD Control
    def QLearningControl(self, episodes, gamma = 0.9, alpha = 0.5, epsilon=1):
        rewards = []
        scores = []

        for i in range(episodes):
            self.agent = Agent()
            episodeRewards = 0

            epsilon = max(epsilon * 0.99995, 0.00001)

            # Add the first card.
            self.agent.addCard(self.getCard())
            while True:
                state = self.sumPoints()

                if state > 21:
                    # Store score.
                    scores.append(state ** 2)
                    break
                
                # Choose A from S using E-greedy
                action = self.chooseActionEGreedy(state, epsilon)

                # Take action A
                if action == self.HIT:
                    self.agent.addCard(self.getCard())

                # Observe R, S'
                nextState = self.sumPoints()
                reward = self.getReward(nextState)

                maxNextState = max(self.Q[nextState].values())

                self.Q[state][action] += alpha * (reward + (gamma * maxNextState) - self.Q[state][action]) 

                # Acumulate reward of episode for results.
                episodeRewards += reward

                if action == self.STICK:
                    # Store quadratic score for results.
                    scores.append(state ** 2)
                    break
                
            rewards.append(episodeRewards)
        
        return scores, rewards

# plots

#n = 100000
#game = Game()
#scores = game.monteCarloControl(n)

#count = 1000
#print('**********Scores*************')
#xs =[]
#ys =[]

#scores = np.split(np.array(scores), n / 1000)

#for r in scores:
#    print(count, ':', str(sum(r/1000)))
#    count += 1000
#    xs.append(count)
#    ys.append(sum(r/1000))

#poly = np.polyfit(xs,ys,5)
#poly_y = np.poly1d(poly)(xs)

#plt.plot(xs,poly_y)

#plt.title('On-Policy First Visit MC Control (for E-soft policies)')
#plt.xlabel('Episodes')
#plt.ylabel('Quadratic Score')
#plt.show()
#sys.exit()

# Number of episodes.
n = 1000000

game = Game()
scores, rewards = game.QLearningControl(n)

rewards = np.split(np.array(rewards), n / 1000)
scores = np.split(np.array(scores), n / 1000)

count = 1000

# For plotting.
xs =[]
ys =[]

print('********** Scores per 1000 episodes *************')
for score in scores:
     print(count, ':', str(sum(score / 1000)))
     count += 1000

     xs.append(count)
     ys.append(sum(score / 1000))

print('********* Q *********')
for i in game.Q:
     print(f'Episode {i}', ':', game.Q[i])

# Plot Results
# Smooth the curve
poly = np.polyfit(xs,ys,5)
poly_y = np.poly1d(poly)(xs)

plt.plot(xs,poly_y)

plt.title('Q Learning')
plt.xlabel('Episodes')
plt.ylabel('Quadratic Score')
plt.show()
sys.exit()