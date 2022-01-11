import numpy as np
import matplotlib.pyplot as plt

def finiteDeck(): 
    cards = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])

    deck = np.repeat(cards, 4)
    np.random.shuffle(deck)

    return deck.tolist()

class Agent:
    def __init__(self):
        self.cards = [] # States

    def addCard(self, card):
        self.cards.append(card)


class Game:
    STICK = 0
    HIT = 1

    def __init__(self, samples):
        self.samples = samples
        self.episodes = []
        self.valueFunctions = []
        self.returns = []

    def policy(self):
        if self.sumPoints() == 20 or self.sumPoints() >= 21:
            return self.STICK 
        return self.HIT;

    def getCard(self):
        cards = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])
        return np.random.choice(cards)

    def sumPoints(self):
        result = sum(self.agent.cards)

        # Ace value.
        if result > 21 and 11 in self.agent.cards:
            result -= 10

        return result

    def getReward(self):
        if self.sumPoints() <= 21:
            return 1
        return 0

    def saveEpisode(self, score, action, reward):
        self.episodes.append((score, action, reward))

    def play(self):
        for i in range(self.samples):
            self.agent = Agent()

            # Deal cards for i in range(2):
            self.agent.addCard(self.getCard())

            while True:
                action = self.policy()

                if action == self.STICK:
                    reward = self.getReward()

                    self.saveEpisode(self.sumPoints(), action, reward)
                    break
                else:
                    self.agent.addCard(self.getCard())
            G=0

            previousStates = [] 
            stateValues = []
            for i in range(len(self.episodes) - 1, 0, -1):
                gamma = 1

                state = self.episodes[i][0] 

                G = gamma * G + self.episodes[i][2] # reward

                if state not in previousStates:
                    self.returns.append(G)
                    stateValues.append(np.mean(self.returns))
                    self.valueFunctions.append(stateValues)
                    previousStates.append(state)


game = Game(10)
game.play()

x = [i[0] for i in game.episodes]
print(x)
print(len(x))

y = game.valueFunctions
print(y)
print(len(y))

# plt.plot(x, y)
# plt.show()
