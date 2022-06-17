scores = [22,29,38,52,71,97,137,148,173,200,210,444,363,400,400,500,500,500,444,400,500,500]



import matplotlib.pyplot as plt

def plotScores(scores):

    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.plot(scores)
    plt.grid()
    plt.show()

plotScores(scores)