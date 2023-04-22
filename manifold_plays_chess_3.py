#! /usr/bin/python3

import requests
import json
import time
import hashlib
import readline
import sys
import random
import os

import avg_prob


API_ENDPOINT="https://manifold.markets/api/v0"


def get_market_by_url(market_url):
    """ extracts the slug from the url and calls the api """

    # extract the slug: its the part of the url after the username
    slug = market_url.split("/")[-1]

    # call the api
    r = requests.get(API_ENDPOINT + "/slug/" + slug)
    return r.json()


def print_market_info(market):
    """ given a market object (dict), print question, id, creatorUsername """
    print("Question: " + market["question"])
    print("ID: " + market["id"])
    print("Creator: " + market["creatorUsername"])


def ask_yes_no(prompt):
    """ asks the user a yes-no question. User answers y or n."""
    while True:
        answer = input(prompt + " [y/n] ")
        if answer == "y":
            return True
        elif answer == "n":
            return False
        else:
            print("Please answer y or n.")


def interactive_filter(outcomes, market):
    """ filter outcomes by asking user for each outcome if it is accepted. Return list of accepted outcomes """

    accepted_outcomes = []
    remove_ids = set()
    accept_all = False

    is_unique = (len(outcomes) == len(set([x[0] for x in outcomes])))
    if not is_unique:
        # kick out the ones with higher id
        outcomes_id_sorted = sorted(outcomes, key=lambda x: x[2])
        unique_moves = set()
        for x in outcomes_id_sorted:
            if x[0] in unique_moves:
                remove_ids.add(x[2])
            else:
                unique_moves.add(x[0])
    for x in outcomes:
        if x[1] == 0.0:
            remove_ids.add(x[2])

    str_outcomes = ", ".join([x[0] for x in outcomes if x[2] not in remove_ids])
    print("Available moves:", str_outcomes)
    accept_all = ask_yes_no("accept all moves?")
    
    for outcome in outcomes:
        if outcome[2] in remove_ids:
            continue
        if accept_all:
            accepted_outcomes.append(outcome)
            continue
        answer = input("accept {:s} (p={:.3f})? [[Y]/n/a]".format(outcome[0], outcome[1]))
        if answer == "y" or answer == "Y" or answer == "[Y]" or answer == "":
            accepted_outcomes.append(outcome)
        if answer == "a":
            accepted_outcomes.append(outcome)
            accept_all = True

    # make sure accpeted_outcomes has unique texts:
    assert(len(accepted_outcomes) == len(set([x[0] for x in accepted_outcomes])))
    return accepted_outcomes


def print_outcomes_avg_probs(outcomes, accepted_outcomes, text=None):
    """ print outcomes by average probability in last x hours before close """

    # print the outcomes, sorted by probability
    if text == None:
        text = "Moves by average probability:"
    print(text)
    for outcome in sorted(outcomes, key=lambda x: -x[1]):
        invalid = ""
        if outcome not in accepted_outcomes:
            invalid = "(removed)"

        print("{:.6f} {} {}".format(outcome[1], outcome[0], invalid))

def calculate_paired_probs(outcomes):
    """ calculate probability if we pick two outcomes using weights"""

    # scale probabilities so that they sum up to 1
    sum_probs = sum(x[1] for x in outcomes)
    probs = [0.0] * len(outcomes)
    for i in range(len(outcomes)):
        probs[i] = outcomes[i][1] / sum_probs

    # we assume that outcomes is already sorted by probbability

    paired_outcomes = []
    for i in range(len(outcomes)):
        for j in range(i+1,len(outcomes)):
            p_i = probs[i]
            p_j = probs[j]
            paired_prob = p_i * p_j * (1.0/ (1.0-p_i) + 1.0 / (1.0-p_j))
            paired_outcomes.append((outcomes[i][0], outcomes[j][0], paired_prob))

    # calculate total probabilty:
    total_prob = 0.0
    for outcome in paired_outcomes:
        total_prob += outcome[2]
    assert(abs(total_prob - 1.0) < 1e-7)

    # sort paired_outcomes by probability, starting with the largest
    paired_outcomes.sort(key=lambda x: x[2], reverse=True)

    return paired_outcomes

def calculate_triple_probs(outcomes):
    """ calculate probability if we pick three outcomes using weights """

    # scale probabilities so that they sum up to 1
    sum_probs = sum(x[1] for x in outcomes)
    probs = [0.0] * len(outcomes)
    for i in range(len(outcomes)):
        probs[i] = outcomes[i][1] / sum_probs

    # we assume that outcomes is already sorted by probbability

    triple_outcomes = []
    for i in range(len(outcomes)):
        for j in range(i+1,len(outcomes)):
            for k in range(j+1,len(outcomes)):
                p_i = probs[i]
                p_j = probs[j]
                p_k = probs[k]
                scale_p_i = 1.0 / (1.0 - p_i) * ( 1.0 / (1.0 - ( p_i + p_j )) + 1.0 / (1.0 - (p_i + p_k)) )
                scale_p_j = 1.0 / (1.0 - p_j) * ( 1.0 / (1.0 - ( p_j + p_k )) + 1.0 / (1.0 - (p_j + p_i)) )
                scale_p_k = 1.0 / (1.0 - p_k) * ( 1.0 / (1.0 - ( p_k + p_i )) + 1.0 / (1.0 - (p_k + p_j)) )
                triple_prob = p_i * p_j * p_k * (1.0/ (1.0-p_i) + 1.0 / (1.0-p_j) + 1.0 / (1.0-p_k))
                triple_prob = p_i * p_j * p_k * (scale_p_i  + scale_p_j + scale_p_k)
                triple_outcomes.append((outcomes[i][0], outcomes[j][0], outcomes[k][0], triple_prob))

    # calculate total probabilty:
    total_prob = 0.0
    for outcome in triple_outcomes:
        total_prob += outcome[3]
    assert(abs(total_prob - 1.0) < 1e-7)

    # sort triple_outcomes by probability, starting with the largest
    triple_outcomes.sort(key=lambda x: x[3], reverse=True)

    return triple_outcomes


def calculate_mixed_outcomes(outcomes, prob_pick_three):
    """ combine results of calculate_paired_probs and calculate_triple_probs.
    Here, prob_pick_three is the probability that we pick 3 markets.
    """

    # calculate probabilities for picking two outcomes
    paired_outcomes = calculate_paired_probs(outcomes)

    # calculate probabilities for picking three outcomes
    triple_outcomes = calculate_triple_probs(outcomes)

    # combine the two lists
    mixed_outcomes = []
    for outcome in paired_outcomes:
        mixed_outcomes.append(((outcome[0], outcome[1]), outcome[2] * (1.0 - prob_pick_three)))
    for outcome in triple_outcomes:
        mixed_outcomes.append(((outcome[0], outcome[1], outcome[2]), outcome[3] * prob_pick_three))

    # calculate total probabilty:
    total_prob = 0.0
    for outcome in mixed_outcomes:
        total_prob += outcome[1]
    assert(abs(total_prob - 1.0) < 1e-7)

    # sort mixed_outcomes by probability, starting with the largest
    mixed_outcomes.sort(key=lambda x: x[1], reverse=True)

    return mixed_outcomes

def get_prob_in_mixed_outcomes(outcomes, accepted_outcomes, mixed_outcomes):
    """ For each outcome, calculate the probability that it is contained in a mixed_outcome.
    We do this by summing up the probabilities of mixed outcomes that countain it."""


    # result has the same format
    result = outcomes.copy()

#     print(result)
#     print(mixed_outcomes)

    # for each outcome, calculate the probability that it is contained in a mixed_outcome
    for i in range(len(result)):
        prob = 0.0
        for mixed_outcome in mixed_outcomes:
            if result[i][0] in mixed_outcome[0]:
                prob += mixed_outcome[1]
        if outcomes[i] in accepted_outcomes:
            result[i] = (result[i][0], prob, result[i][2])
        else:
            result[i] = (result[i][0], 0.0, result[i][2])
    return result


def integer_range_outcomes(mixed_outcomes):
    """ convert probabilities to non-overlapping ranges over integers from 1 to 1 billion """
    # convert probabilities to integers

    int_outcomes = []
    used_integers = 0
    for outcome in mixed_outcomes:
        weight = int(outcome[1]*1e9)
        int_outcomes.append((outcome[0], used_integers + 1, used_integers + weight))
        used_integers += weight

    return int_outcomes

def print_integer_range_outcomes(int_outcomes):

    print("Outcomes by integer range:")
    for outcome in int_outcomes:
        moves = ", ".join(outcome[0])
        print("[{:9d}-{:9d}] {}".format(outcome[1], outcome[2], moves))

def print_short_integer_range_outcomes(int_outcomes):
    """ print a user friendly version of int_outcomes. 
    Dont print low probability stuff. 
    Print a hash of the complete table for verification. """

    min_print = 7
    max_print = 20
    dont_print_after = 90*1e7
    max_int = int_outcomes[-1][2]

    print("pick a number between 1 and {} (inclusive)".format(max_int))

    print("Outcomes by integer range:")
    last_integer = 0
    for i, outcome in enumerate(int_outcomes):

        if i >= max_print:
            break
        if outcome[1] < dont_print_after or i < min_print:
            moves = ", ".join(outcome[0])
            print("[{:9d}-{:9d}] {}".format(outcome[1], outcome[2], moves))
            last_integer = outcome[2]

    # print row for remaining integers:
    if last_integer < max_int:
        print("[{:9d}-{:9d}] {}".format(last_integer, max_int, "other"))

    # print hash of the table
    print("Hash of the complete table:")
    print(hashlib.sha256(str(int_outcomes).encode('utf-8')).hexdigest())

def ensure_data_dir():
    """ Make sure the data/ directory exists. Create it if it does not exist."""
    if not os.path.exists('data'):
        os.makedirs('data')

def save_int_outcomes_to_file(int_outcomes):
    """ writes the table of outcomes to a file.
    filename should contain the hash of its content. """

    # calculate the hash of the file content
    file_hash = hashlib.sha256(str(int_outcomes).encode('utf-8')).hexdigest()
    ensure_data_dir()
    filename = "data/outcomes_{}.txt".format(file_hash)

    with open(filename, "w") as f:
        f.write(str(int_outcomes))

def main_free_response(market):
    """ main routine when it is a dpm free response market """

    # get the outcomes
    outcomes = avg_prob.get_outcomes_data_with_avg_probs(market, time_window = 4*3600)

    # filter the outcomes
    accepted_outcomes = interactive_filter(outcomes, market)

    # calculate paired_outcomes
    # paired_outcomes = calculate_paired_probs(accepted_outcomes)

    # calculate mixed_outcomes
    mixed_outcomes = calculate_mixed_outcomes(accepted_outcomes, 0.25)

    int_outcomes = integer_range_outcomes(mixed_outcomes)

    save_int_outcomes_to_file(int_outcomes)
    print("\n" + "="*30 + "\n")
    print_outcomes_avg_probs(outcomes, accepted_outcomes)
    print("\n\n")
    print_short_integer_range_outcomes(int_outcomes)
    print("\n" + "="*30 + "\n")

    print_more = ask_yes_no("print more info?")
    if not print_more:
        return

    print_integer_range_outcomes(int_outcomes)
    print("\n" + "="*30 + "\n")
    print_outcomes_avg_probs(outcomes, accepted_outcomes)
    print("\n\n")
    print_short_integer_range_outcomes(int_outcomes)
    print("\n" + "="*30 + "\n")
    p_in_mixed = get_prob_in_mixed_outcomes(outcomes, accepted_outcomes, mixed_outcomes)
    print_outcomes_avg_probs(p_in_mixed, p_in_mixed, "Chance of candidate move")


def probabilistic_rounding(x):
    """ x should be between 0 and 1.
    rounds x up or down two precision of two digits.
    weighted random so that expected value is x.
    returns integer (percentage).
    Reproducible (seed is stored in a file)."""

    # only reproducible after move 20 or so.

    # convert x to integer
    x_int = int(x*100)

    # calculate the probability that we round up
    p_round_up = 100*x - x_int
    assert(0.0 <= p_round_up < 1.0)

    seed_file = "data/seed"
    try:
        seed = float(open(seed_file).read())
    except FileNotFoundError:
        seed = random.random()
        ensure_data_dir()
        open(seed_file, "w").write(str(seed))

    # generate random number
    random.seed(seed + x) # add x to make it less predictable
    r = random.random()

    # round up or down
    if r < p_round_up:
        return x_int + 1
    else:
        return x_int



def main_binary(market):
    """ main routine when it is a binary market """

    p = avg_prob.get_binary_probability_avg(market, time_window= 4* 3600)
    print("")
    print("Average probability: {:.6f}".format(p))
    print("")
    more = ask_yes_no("more info")
    if more:
        prob_rounded_p = probabilistic_rounding(p)
        print("probabilistic rounded: %d%%" %( prob_rounded_p))



def main_interactive():
    """ asks user for url or gets url from parameters, then does the calculations and api calls """

    # check first if first arg is a url
    if len(sys.argv) > 1:
        market_url = sys.argv[1]
    else:
        market_url = input("Enter market url: ")

    # get the market
    market = get_market_by_url(market_url)

    # print the market info
    print_market_info(market)

    time.sleep(0.4)

    if market["outcomeType"] == "FREE_RESPONSE":
        main_free_response(market)
    elif market["outcomeType"] == "BINARY":
        main_binary(market)
    else:
        print("market type unknown!!")


if __name__ == "__main__":
    main_interactive()


