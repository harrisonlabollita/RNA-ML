# REINFORCEMENT LEARNING FOR RNA FOLDING PREDICTION
# --------------------------------------------------
# Input: RNA Sequence
# Convert sequence to matrix (following Zhang et. al.)
# Use CNN Model to predict a pairing map
#
#                ACTIONS
#         (          )      .
#     ----------------------------
#  S  |   0.1  |  0.4    |  0.5  |
#  E  |   0.9  |  0.1    |  0.0  |
#  Q  |   0.6  |  0.2    |  0.2  |
#  U  |   0.2  |  0.6    |  0.2  |
#  E  |   0.3  |  0.1    |  0.6  |
#  N  |   0.6  |  0.1    |  0.3  |
#  C  |   0.5  |  0.2    |  0.3  |
#  E  |   0.2  |  0.8    |  0.0  |
#     ----------------------------
# Use this pairing map for reinforcment learning (Q learning)
#      Q(s, a) <----- Q(s, a) + alpha ( r + gamma * max( Q(s', a'))),
# where r is the immediate reward, alpha is the learning rate, gamma is the
# discount, and max(Q(s', a')) is the maximum reaward for the next state.
# Here I envision each state as a nucleotide, which is corresponding actions associated with it, pair forward,
# pair back, or do not pair.
#
# Possible loop
# 1. Start in state 0, select the action suggested by the CNN.
# 2. if action in state 0 is to pair, then find all possible pairings for state 0 (based on physical conditions)
# 3. Calculate the max ( # of possible pairings)
# 4. Fix the paired nucleotides
# 5. Repeat


def find_possible_pairings(nucl, sequence):
    # find all of the physical pairings for this nucleotide
    return

def calculate_Q(nucleotide):
    # for this state (nucleotide)
    # determine max value
    return

def calculate_next_Q(nucleotide, current_state, action):
    # find the next highest scored move to make
    return
