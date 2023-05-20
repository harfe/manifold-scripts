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

def get_market_by_id(market_id):
    """ calls the api and gets the market data """

    # call the api
    r = requests.get(API_ENDPOINT + "/market/" + market_id)
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

    if len(outcomes)==1:
        outcome = outcomes[0]
        return [((outcome[0],), 1.0)]

    # calculate probabilities for picking two outcomes
    paired_outcomes = calculate_paired_probs(outcomes)

    if len(outcomes)==2:
        return [((x[0],x[1]), x[2]) for x in paired_outcomes]

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

    print("\n" + "-"*20 + "\n")
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


# leverage:
# - list of markets with parents and params
# - dict: [market]
#   - parent, params, moveNr, move(?), mkt_avg, 
# - para

def load_conditional_market_file():

    filename = "data/cond_markets.json"
    contents = open(filename).read()
    # you have to init the file if it does not exist yet
    return json.loads(contents)

def save_conditional_market_file(data):

    filename = "data/cond_markets.json"
    contents = json.dumps(data, indent=2)
    with open(filename, "w") as f:
        f.write(contents)

def find_best_mkt(mkts, mkt_data):
    """ returns the market id with the best mkt_avg """
    curr_best = mkts[0]
    for x in mkts:
        if mkt_data[x]["mkt_avg"] > mkt_data[curr_best]["mkt_avg"]:
            curr_best = x
    return curr_best

def suggest_market_entry(market, mkt_data):
    """ makes suggestions for an entry into the conditional markets file.
    Returns True or False to indicate success/failure"""

    market_id = market["id"]
    assert( market_id not in mkt_data )
    title = market["question"]

    # try to extract move and move number
    try:
        number_str = title.split(".")[0].split(" ")[-1]
        number = int(number_str)
        move_str = title.split(". ")[1].split(",")[0]
        move_str = number_str + ". " + move_str
    except:
        print("could not parse market title", title)
        return False

    mkt_avg = avg_prob.get_binary_probability_avg(market, time_window= 4* 3600)

    if market["closeTime"] / 1000 > time.time():
        # if market not yet closed
        mkt_avg = None

    # search for parent
    poss_parent = []
    for mkt in mkt_data:
        if mkt_data[mkt]["moveNumber"] == number -1:
            poss_parent.append(mkt)
    if len(poss_parent)==0:
        print("no parent candidates found for move", number)
        return False
    assert(len(poss_parent)<=3)
    # parent is the one with best score
    parent = find_best_mkt(poss_parent, mkt_data)

    # params hard-coding
    if number < 31:
        params = [1.0, 0.5]
    else:
        center_mkt = mkt_data[parent]["mkt_avg"]
        center_score = score(center_mkt, mkt_data[parent]["params"])
        params = [5.0, round(center_score, 2)]

    print("We suggest the following market data:")
    print("moveNumber:", number)
    print("move:", move_str)
    print("params:", params)
    print("market closing average value:", mkt_avg)
    parent_move = mkt_data[parent]["move"]
    print("parent market: move:", parent_move)
    print("parent market: closing avg:", mkt_data[parent]["mkt_avg"])

    yn = ask_yes_no("Do you accept this data?")
    if not yn:
        return False

    mkt_data[market_id] = {"parent": parent,
                           "params": params,
                           "moveNumber": number,
                           "move": move_str,
                           "mkt_avg": mkt_avg}

    save_conditional_market_file(mkt_data)
    print("data saved")
    return True


def find_children(parent, mkt_data):
    """ find all entries which have the right parent. """
    return [x for x in mkt_data if mkt_data[x]["parent"]==parent]



def score(mkt, param):
    # converts market value to score when leverage is used

    if param[0]==1.0:
        return mkt

    assert( 2.0 < param[0] <= 8.0)
    tolerance = 0.1**8

    leverage = param[0] # leverage near the center
    center = param[1] # is score that corresponds to mkt 0.5

    # interpolation point
    low_mkt = 0.1
    high_mkt = 0.9

    if center + 0.5 / leverage + tolerance > 1.0 :
        low_score = 1.0 - (1.0-low_mkt)/leverage
        high_score = 1.0 - (1.0-high_mkt)/leverage
    elif center - 0.5 / leverage < tolerance:
        low_score = low_mkt / leverage
        high_score = high_mkt / leverage
    else: 
        low_score = center - (0.5 - low_mkt) / leverage
        high_score = center + (high_mkt - 0.5) / leverage

    assert(high_score < 1.0)
    assert(low_score > 0.0)
    assert( 0.0 <= mkt <= 1.0)

    # now do the interpolation
    if mkt < low_mkt:
        return mkt * low_score / low_mkt
    elif mkt < high_mkt:
        return low_score + (mkt - low_mkt)*(high_score-low_score) / (high_mkt - low_mkt)
    else:
        return 1.0 - (1.0 - mkt) * (1.0-high_score) / (1.0 - high_mkt)

def mkt(score, param):
    # inverse function of score()

    if param[0]==1.0:
        return score

    assert( 2.0 < param[0] <= 8.0)
    tolerance = 0.1**8

    leverage = param[0] # leverage near the center
    center = param[1] # is score that corresponds to mkt 0.5

    # interpolation point
    low_mkt = 0.1
    high_mkt = 0.9

    if center + 0.5 / leverage + tolerance > 1.0 :
        low_score = 1.0 - (1.0-low_mkt)/leverage
        high_score = 1.0 - (1.0-high_mkt)/leverage
    elif center - 0.5 / leverage < tolerance:
        low_score = low_mkt / leverage
        high_score = high_mkt / leverage
    else: 
        low_score = center - (0.5 - low_mkt) / leverage
        high_score = center + (high_mkt - 0.5) / leverage

    assert(high_score < 1.0)
    assert(low_score > 0.0)
    assert( 0.0 <= score <= 1.0)

    # now do the inverse interpolation
    if score < low_score:
        return score * low_mkt / low_score
    elif score < high_score:
        return low_mkt + (score - low_score)*(high_mkt-low_mkt) / (high_score - low_score)
    else:
        return 1.0 - (1.0 - score) * (1.0-high_mkt) / (1.0 - high_score)

def print_mkt_score_table(param):
    # Print a table of example (mkt, score) pairs using the score() and mkt() functions
    # for interesting values of mkt

    print("Here is a table of the correspondence to market value and score")
    print('value  score')
    print('-----  -----')

    mkt_vals = [0.0, 0.03, 0.07, 0.1, 0.2, 0.3, 0.4]
    mkt_vals += [0.5] + sorted([ 1-x for x in mkt_vals])

    for mkt_val in mkt_vals:
        score_val = score(mkt_val, param)
        # mkt_val2 = mkt(score_val, param)
        print(f'{mkt_val:.2f}   {score_val:.3f}')

    print('----- -----')
    score_val_lo = score(0.1, param)
    score_val_hi = score(0.9, param)
    def_str = "(0.0, 0.0), (0.1, %.2f), (0.9, %.2f), (1.0)" %(score_val_lo, score_val_hi)
    print("This correspondence is defined by linearly interpolating between the points")
    print(def_str + ".")


def main_binary(market):
    """ main routine when it is a binary market """

    p = avg_prob.get_binary_probability_avg(market, time_window= 4* 3600)
    print("")
    print("Average probability: {:.6f}".format(p))
    mkt_data = load_conditional_market_file()
    if market["id"] in mkt_data:
        params = mkt_data[market["id"]]["params"]
        my_score = score(p, params)
        print("Score: {:.6f}".format(my_score))
    print("")
    more = ask_yes_no("more info")
    if not more:
        return
    mkt_data = load_conditional_market_file()
    if market["id"] not in mkt_data:
        res = suggest_market_entry(market, mkt_data)
        if not res:
            return
        time.sleep(0.5)
        mkt_data = load_conditional_market_file()

    # params = mkt_data[market["id"]]["params"]
    parent = mkt_data[market["id"]]["parent"]
    siblings = find_children(parent, mkt_data)
    # print("siblings: ", [mkt_data[x]["move"] for x in siblings])
    # siblings_all = ask_yes_no("are these conditional markets all markets in that move?")
    # if not siblings_all:
    #     return

    # update market values
    need_update = False
    for s in siblings:
        s_mkt = mkt_data[s]["mkt_avg"]

        if s_mkt != None:
            # no update needed
            continue

        s_api_data = get_market_by_id(s)
        if s_api_data["closeTime"] / 1000 > time.time():
            # if market not yet closed
            print("not all market closed. sorry")
            return
        p = avg_prob.get_binary_probability_avg(s_api_data, time_window= 4* 3600)
        need_update = True
        mkt_data[s]["mkt_avg"] = p

    if need_update:
        save_conditional_market_file(mkt_data)

    winner = find_best_mkt(siblings, mkt_data)

    print("\n" + "="*30 + "\n")
    for s in siblings:
        s_mkt = mkt_data[s]["mkt_avg"]
        assert(s_mkt != None)

        s_score = score(s_mkt, mkt_data[s]["params"])

        print(mkt_data[s]["move"] + ": Average market value: {:.6f}".format( s_mkt))
        print(mkt_data[s]["move"] + ":                score: {:.6f}".format( s_score))

    print("Winner:", mkt_data[winner]["move"])
    winning_mkt = mkt_data[winner]["mkt_avg"]
    winning_score = score(winning_mkt, mkt_data[winner]["params"])

    if parent == None:
        print("(no parent in system)")
        return

    # resolution of parent market

    print("\n" + "-"*20 + "\n")

    print("resolution of market", mkt_data[parent]["move"]+":")
    parent_params = mkt_data[parent]["params"]
    mkt_val = mkt(winning_score, parent_params)
    print("resolution score: {:.6f}".format( winning_score))
    print("corresponding market value: {:.6f}".format( mkt_val))
    prob_rounded_p = probabilistic_rounding(mkt_val)
    print("probabilistically rounded: %d%%" %( prob_rounded_p))

    print("\n" + "="*30 + "\n")

    next_move_number = mkt_data[winner]["moveNumber"] + 1
    print("Table for move", str(next_move_number) +  ":")
    if next_move_number < 31:
        next_params = [1.0, round(winning_score, 2)]
    else:
        next_params = [5.0, round(winning_score, 2)]

    print_mkt_score_table(next_params)


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


