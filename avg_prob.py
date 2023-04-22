#! /usr/bin/python3

import requests
import json
import time
import hashlib
import sys

import argparse


API_ENDPOINT="https://manifold.markets/api/v0"

""" 
Calculates the average probability in a time window before close (eg the last hour before close).
Both DPM markets and binary markets are supportet.

If market is still open, then it assumes that the current probabilities don't change until close.

Currently does not support average probability in some time window that does not end at close.
"""


def get_list_of_bets(contractId, limit, before=None):
    """ makes an api call to get a list of bets """

    # set the url
    url = API_ENDPOINT + "/bets"

    params={}
    params["contractId"] = contractId
    params["limit"] = limit
    if before:
        params["before"] = before

    # make the request
    r = requests.get(url, params=params)
    return r.json()

def get_last_bets(market, time_window):
    """ collect a list of bets that were made in the last time_window seconds before close time"""

    total_limit = 100 * 1000 # total limit of bets
    limit_per_api_call = 500

    # get closeTime
    close_time = market["closeTime"] # in milliseconds

    # get the current time
    current_time = time.time()
    # assert(close_time/1000 < current_time)

    # get the number of api calls we need to make
    max_number_of_api_calls = 1+int(total_limit/limit_per_api_call)

    # get the bets
    all_bets = []
    for i in range(max_number_of_api_calls):
        bets = get_list_of_bets(market["id"], limit_per_api_call, all_bets[-1]["id"] if all_bets else None)
        all_bets.extend(bets)

        # if last bet is more than time_window older than close_time, break
        if bets[-1]["createdTime"] < close_time - time_window*1000:
            break

    # only return those bets which are in the last time_window seconds before close time
    return [bet for bet in all_bets if bet["createdTime"] > close_time - time_window*1000]


def get_dpm_probabilites_avg(market, time_window):
    """ get average probabilities for dpm outcomes over the last time_window seconds"""

    bets = get_last_bets(market, time_window)
    for i in range(len(bets)-1): # check that they are sorted
        assert(bets[i]["createdTime"] > bets[i+1]["createdTime"])

    close_time = market["closeTime"] # in milliseconds
    tol = 1e-7 # tolerance for numerical tests

    # market should be open for longer than time_window
    assert(close_time - time_window*3600 > market["createdTime"])

    probs = [0.0] + [x["probability"] for x in market["answers"]]
    probs[0] = 1.0 - sum(probs)
    # starting_probs = probs[:]

    assert(abs(sum(probs) - 1.0) < tol)

    prob_time = [0.0]*len(probs) # added with time in ms

    t = close_time

    for bet in bets: # go backwards in time
        p_before = bet["probBefore"]
        p_after = bet["probAfter"]
        # shares = bet["shares"]
        outcome = int(bet["outcome"])

        assert(abs(p_after - probs[outcome]) < tol)

        time_diff = t - bet["createdTime"]
        t = bet["createdTime"]
        assert(time_diff >= 0)

        for i in range(len(probs)):
            prob_time[i] += probs[i] * time_diff

        # calculate probs before bet
        probs[outcome] = p_before
        scale_factor = (1-p_before)/(1-p_after)
        for i in range(len(probs)):
            if i == outcome:
                continue
            probs[i] = scale_factor * probs[i]

        assert(abs(sum(probs) - 1.0) < tol)

    # difference of time window beginning and oldest bet in time window
    time_diff = t - (close_time - time_window*1000)
    assert(time_diff >= 0)
    for i in range(len(probs)):
        prob_time[i] += probs[i] * time_diff

    avg_probs = [x / (time_window * 1000) for x in prob_time]
    return avg_probs


def get_binary_probability_avg(market, time_window):
    """ get average probability for binary markets over the last time_window seconds"""

    bets = get_last_bets(market, time_window)
    for i in range(len(bets)-1): # check that they are sorted
        assert(bets[i]["createdTime"] >= bets[i+1]["createdTime"])

    close_time = market["closeTime"] # in milliseconds
    tol = 1e-7

    # market should be open for longer than time_window
    assert(close_time - time_window*3600 > market["createdTime"])

    prob = market["probability"]

    prob_time = 0.0 # added with time in ms

    t = close_time

    for bet in bets: # go backwards in time

        p_before = bet["probBefore"]
        p_after = bet["probAfter"]
        # shares = bet["shares"]
        # outcome = int(bet["outcome"])

        if not (abs(p_after - prob) < tol):
            # print( p_after, prob)
            if bet["isRedemption"] or ( "isFilled" in bet and bet["isFilled"]):
                # ignore this probability for our calculations
                continue

            else:
                assert(False) # we should not be here, I think the other bets are in order
            
        time_diff = t - bet["createdTime"]
        t = bet["createdTime"]
        assert(time_diff >= 0)

        prob_time += prob * time_diff

        # calculate probs before bet
        prob = p_before

    # difference of time window beginning and oldest bet in time window
    time_diff = t - (close_time - time_window*1000)
    assert(time_diff >= 0)
    prob_time += prob * time_diff

    avg_prob = prob_time / (time_window * 1000)
    return avg_prob

def get_outcomes_data_with_avg_probs(market, time_window, free_response=True):
    """ get a list of outcomes/answers. 
    Each list entry is a triple with answer text, average prob, and answer id.
    Sorted with highest prob first."""

    avg_probs = get_dpm_probabilites_avg(market, time_window)
    if free_response:
        outcomes = [(x["text"], avg_probs[x["number"]], x["number"]) for x in market["answers"]]
    else: # multiple choice, discard index 0
        outcomes = [(x["text"], avg_probs[x["number"]+1], x["number"]) for x in market["answers"]]

    return sorted(outcomes, key=lambda x: -x[1])

def print_outcomes(outcomes):
    """ prints a list of outcomes, answers, as returned by get_outcomes_data_with_avg_probs(). """

    for outcome in outcomes:
        print("{:.6f} {}".format(outcome[1], outcome[0]))

# The functions below are for use of the script in command line

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
    print("type: " + market["outcomeType"])
    print("Creator: " + market["creatorUsername"])

def main():
    """ parses arguments, displays results."""

    description = """Calculates average value of a market in the last hour 
    (or the last T seconds) the market.
    If market is not closed yet, then the current value gets extended until close."""
    parser = argparse.ArgumentParser(description=description, prog="./avg_prob.py")
    parser.add_argument('url', help='the url of the market')
    parser.add_argument('-t', 
                        help='time window in seconds. If it ends with h or m, then it gets interpreted as hours or minutes.', 
                        metavar='T', default='3600')

    args = parser.parse_args()
    url = args.url
    if args.t[-1]=='h': # parse hours
        time_window = float(args.t[:-1])*3600
        time_window = int(time_window)
    elif args.t[-1]=='m': # parse minutes
        time_window = float(args.t[:-1])*60
        time_window = int(time_window)
    elif args.t[-1]=='s': # parse seconds
        time_window = float(args.t[:-1])
    else: # parse seconds
        time_window = float(args.t)

    market = get_market_by_url(url)
    print_market_info(market)
    print("Will calculate the average market values in the last", time_window, "seconds before close:")
    print("")
    if market["outcomeType"] == "FREE_RESPONSE":
        outcomes = get_outcomes_data_with_avg_probs(market, time_window, free_response=True)
        print_outcomes(outcomes)
    elif market["outcomeType"] == "MULTIPLE_CHOICE":
        outcomes = get_outcomes_data_with_avg_probs(market, time_window, free_response=False)
        print_outcomes(outcomes)
    elif market["outcomeType"] == "BINARY":
        prob = get_binary_probability_avg(market, time_window)
        print("Average probability: {:.6f}".format(prob))
    else:
        print("market type not supported.")

if __name__ == "__main__":
    main()
