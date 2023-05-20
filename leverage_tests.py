
from manifold_plays_chess_3 import *
import random

import matplotlib.pyplot as plt
import numpy as np


# test and plot the leverage stuff


def test_inverse_score_mkt():
    # Test whether score() and mkt() are inverses of each other
    # using random input values

    for i in range(10**5):
        # Generate random input values
        param = [random.uniform(2.1, 8.0), random.uniform(0.0, 1.0)]
        mkt_val = random.uniform(0.0, 1.0)
        score_val = score(mkt_val, param)

        # Compute the inverse of score()
        mkt_val2 = mkt(score_val, param)

        # Check whether the computed values are equal within tolerance
        assert(abs(mkt_val - mkt_val2) < 1e-8)

def plot_score_mkt(center = 0.73):
    # Plot the score() and mkt() functions with param = [4.0, center]

    # Generate mkt values
    mkt_vals = np.linspace(0, 1, 1000)

    param = [5.0, center]

    # also print the table
    print_mkt_score_table(param)

    # Compute corresponding score values
    score_vals = [score(mkt_val, param) for mkt_val in mkt_vals]

    # Compute corresponding mkt values
    mkt_vals2 = [mkt(score_val, param) for score_val in score_vals]


    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(mkt_vals, score_vals, label='score')
    ax.plot(mkt_vals, mkt_vals2, label='mkt')
    ax.set_xlabel('mkt')
    ax.set_ylabel('score')
    ax.legend()
    plt.show()


def init_mkt_file():
    # call this function to setup the conditional markets file
    # make sure file does not exist or has a backup before calling

    # hardcoded data for winning move 28
    key_28 = "vNDqO66vByz9eeqBEZbl"
    data_28 = {}
    data_28["parent"] = None
    data_28["params"] = [1.0, 0.5]
    data_28["moveNumber"] = 28
    data_28["move"] = "28. Kd2"
    data_28["mkt_avg"] = 0.770406553510696
    data = { key_28 : data_28 }
    save_conditional_market_file(data)

