import os
import numpy as np
from scipy.stats import pearsonr
import random
import math
from itertools import permutations
from scipy.stats import vonmises

# ============================================================
# 1. Generate Unrelated Maps
# ============================================================

# Get monsters
n_monster = 7
monsters = list(range(n_monster))

# compute map RDM (upper triangle)
def compute_rdm(map):
    positions = {stim: idx for idx, stim in enumerate(map)}
    distances = []

    for i in range(n_monster):
        for j in range(i + 1, n_monster):
            pos_i = positions[monsters[i]]
            pos_j = positions[monsters[j]]

            d = abs(pos_i - pos_j)
            circular_distance = min(d, n_monster - d)

            distances.append(circular_distance)

    return np.array(distances)

# generate random map (permutation)
def random_map():
    map = list(range(n_monster))
    random.shuffle(map)
    return map

# check shared neighbomap pairs
def shared_structure_stats(map_one, map_two):
    directed_pairs_one = set()
    reversed_pairs_one = set()
    unordered_pairs_one = set()
    triples_one = set()

    n = len(map_one)

    for i in range(n):
        a = map_one[i]
        b = map_one[(i + 1) % n]
        c = map_one[(i + 2) % n]

        directed_pairs_one.add((a, b))
        reversed_pairs_one.add((b, a))
        unordered_pairs_one.add(frozenset([a, b]))

        triple = (a, b, c)
        triples_one.add(triple)
        triples_one.add(tuple(reversed(triple)))

    n_shared_pairs = 0
    n_same_direction_pairs = 0
    n_reversed_pairs = 0
    has_shared_three_chain = False

    for i in range(len(map_two)):
        a = map_two[i]
        b = map_two[(i + 1) % len(map_two)]
        c = map_two[(i + 2) % len(map_two)]

        if frozenset([a, b]) in unordered_pairs_one:
            n_shared_pairs += 1

        if (a, b) in directed_pairs_one:
            n_same_direction_pairs += 1

        if (a, b) in reversed_pairs_one:
            n_reversed_pairs += 1

        if (a, b, c) in triples_one:
            has_shared_three_chain = True

    return n_shared_pairs, n_same_direction_pairs, n_reversed_pairs, has_shared_three_chain

# generate unrelated map pairs
def generate_unrelated_maps():
    tolerance = 0.08   # adjust if needed

    while True:

        map_one = random_map()
        rdm_one = compute_rdm(map_one)

        # try multiple candidates for B
        for _ in range(1000):

            map_two = random_map()

            # skip identical maps
            if map_two == map_one:
                continue
            
            # 1. abs(r) < 0.08, with all current constraints, the minimal abs(r) is 0.07143
            # 2. shared adjacent pairs are allowed
            # 3. same-direction shared adjacent pairs are not allowed
            # 4. shared three-monster chains are not allowed, including reversed chains
            n_shared_pairs, n_same_direction_pairs, n_reversed_pairs, has_shared_three_chain = shared_structure_stats(map_one, map_two)

            # shared three-monster chains are not allowed, including reversed chains
            if has_shared_three_chain:
                continue

            # reject same-direction shared adjacent pairs
            if n_same_direction_pairs > 0:
                continue

            rdm_two = compute_rdm(map_two)
            r, _ = pearsonr(rdm_one, rdm_two)

            if abs(r) < tolerance:
                return map_one, map_two, r, n_shared_pairs, n_same_direction_pairs, n_reversed_pairs, has_shared_three_chain


# ============================================================
# 2. Generate Task Sequences
# ============================================================

# make design files for choice task
# no need to read maps, map index is enough
# 168 trials in each choice task with single context
# 420 trials in each choice task with double context (5 blocks)
# 84 trials in each block (10 mini-blocks)
# no repeated monster on two successive trials
class make_choice_design_file:
    def __init__(self, single=True):
        self.n_monster = 7
        self.kappa = 1
        self.n_dist = 20 if single else 10 # number of mini-blocks (different reward distributions)
        self.single = single
        self.mini_block_size = [7,8,8,9,10]
        self.n_iter = 4
        self.n_peaks = 12
        
    # assign rewards to each map based on von Mises distribution 
    # von Mises distribution is a used for a circular distribution
    # mu is the peak and kappa is the concentration. 
    # Higher kappa means more concentrated around the peak, 
    # and peak does not have to be an eaxct posotion of the monster
    # this function is adpated from Jana's code, 
    # Kappa = 1, n_monster = 7, n_dist = 10 
    # avoid same/similar peak points on two sucessive mini-blocks
    # at least one monster apart
    def generate_reward_dist(self, prev_mu = None):
        reward_dist = []
        mu_list = []
        peak_idx_list = []
        min_sep = 2 * math.pi / self.n_monster  # at least one monster apart
        peak_angles = np.linspace(0, 2 * np.pi, self.n_peaks, endpoint = False)

        for _ in range(self.n_dist):
            while True:
                peak_idx = random.randrange(self.n_peaks)
                mu = peak_angles[peak_idx]

                if prev_mu is None:
                    break
                diff = abs(mu - prev_mu)
                circ_dist = min(diff, 2 * math.pi - diff)
                if circ_dist >= min_sep:
                    break
            prev_mu = mu

            angles = [(2 * np.pi / self.n_monster) * i for i in range(self.n_monster)]
            pdf_vals = vonmises.pdf(angles, loc = mu, kappa = self.kappa)
            min_val = vonmises.pdf(mu + np.pi, loc = mu, kappa = self.kappa)
            max_val = vonmises.pdf(mu, loc = mu, kappa = self.kappa)

            rewards = [round(100*(r-min_val)/(max_val-min_val)) for r in pdf_vals]
            
            reward_dist.append(rewards)
            mu_list.append(mu)
            peak_idx_list.append(peak_idx)

        return reward_dist, mu_list, peak_idx_list, prev_mu
        
    # generate an order of 42 pairs for stimulus presentation 
    # with no shared monster on two sucessive trials
    # NOTICE: each block has 84 trials, which is 2 iterations of the 42 pairs
    def shuffle_no_adjacent_shared(self):
        all_pairs = [list(p) for p in permutations(range(self.n_monster), 2)]

        while True:  # keep restarting until success
            pairs = all_pairs.copy()
            random.shuffle(pairs)

            final_pairs = [pairs.pop()]
            success = True

            while pairs:
                a, b = final_pairs[-1]
                candidates = [
                    p for p in pairs
                    if p[0] != a and p[0] != b
                    and p[1] != a and p[1] != b
                    and not (p[0] == b and p[1] == a)
                ]

                if not candidates:
                    success = False
                    break

                next_pair = random.choice(candidates)
                final_pairs.append(next_pair)
                pairs.remove(next_pair)

            if success:
                return final_pairs
    
    # generate 10 mini-block contexts, balanced 5/5, with 4 or 5 changes
    def generate_context_order(self):
        sequence = ["map_one"] * 5 + ["map_two"] * 5
        while True:
            random.shuffle(sequence)
            changes = sum(a != b for a, b in zip(sequence, sequence[1:]))
            if changes in [4, 5]:
                return sequence

    # make a semi-block, with 5 mini-blocks, sizes from [7, 8, 8, 9, 10]
    def make_semi_block(self, block_dist, block_mu, block_peak_idx, map_label, map_structure, return_mini_blocks = False):
        pairs = self.shuffle_no_adjacent_shared()
        size = self.mini_block_size.copy()
        random.shuffle(size)
        total_trial = len(pairs)
        
        block_design_file = []
        mini_blocks = []
        
        for i in range(len(size)):
            mini_size = size[i]
            mini_dist = block_dist[i]

            for _ in range(mini_size):
                trial_design_file = {}
                pair = pairs.pop(0)
                
                trial_design_file['trialID'] = total_trial-len(pairs) # loop every 42 iterations, here is only for generation, the true trialID is assigned in js code
                trial_design_file['map'] = map_label
                trial_design_file['mu'] = block_mu[i]
                trial_design_file['peak_idx'] = block_peak_idx[i]
                trial_design_file['reward_dist'] = mini_dist
                trial_design_file['left_pos'] = pair[0]
                trial_design_file['right_pos'] = pair[1]
                trial_design_file['left_choice'] = map_structure[pair[0]]
                trial_design_file['right_choice'] = map_structure[pair[1]]
                trial_design_file['left_reward'] = mini_dist[pair[0]]
                trial_design_file['right_reward'] = mini_dist[pair[1]]
                
                block_design_file.append(trial_design_file)
            mini_blocks.append(block_design_file[-mini_size:])
        
        return mini_blocks if return_mini_blocks else block_design_file
    
    # make choice task design file
    def make_choice_task(self, which_map, map_info):

        def merge_semi_blocks(semi_blocks):
            return [semi_blocks[i] + semi_blocks[i+1] for i in range(0, len(semi_blocks), 2)]

        if self.single:
            reward_dist, mu_list, peak_idx_list, _ = self.generate_reward_dist()
            choice_design_file = []
            map_label = which_map
            map_structure = map_info[which_map]

            for _ in range(self.n_iter):
                semi_block_dist = [reward_dist.pop(0) for _ in range(5)]
                semi_block_mu = [mu_list.pop(0) for _ in range(5)]
                semi_block_peak_idx = [peak_idx_list.pop(0) for _ in range(5)]
                semi_block_design_file = self.make_semi_block(semi_block_dist, semi_block_mu, semi_block_peak_idx, map_label, map_structure)

                # avoid shared monsters across semi-block boundaries.
                if choice_design_file:
                    previous_trial = choice_design_file[-1][-1]
                    previous_pair = {previous_trial['left_choice'], previous_trial['right_choice']}
                    while previous_pair & {semi_block_design_file[0]['left_choice'], semi_block_design_file[0]['right_choice']}:
                        semi_block_design_file = self.make_semi_block(semi_block_dist, semi_block_mu, semi_block_peak_idx, map_label, map_structure)
                choice_design_file.append(semi_block_design_file)

            # merge every 2 semi-blocks
            choice_design_file = merge_semi_blocks(choice_design_file)

        else:
            n_block = 5 # 5 blocks, 2 outside scanner and 3 inside scanner
            choice_design_file = []
            previous_mu = None
            for _ in range(n_block):
                reward_dist, mu_list, peak_idx_list, previous_mu = self.generate_reward_dist(prev_mu = previous_mu)
                map_list = self.generate_context_order()

                # assign rewards in final presentation order to preserve peak separation.
                context_dist = {
                    map: [dist for context, dist in zip(map_list, reward_dist) if context == map]
                    for map in ["map_one", "map_two"]
                }

                context_mu = {
                    map: [mu for context, mu in zip(map_list, mu_list) if context == map]
                    for map in ["map_one", "map_two"]
                }

                context_peak_idx = {
                    name: [peak_idx for context, peak_idx in zip(map_list, peak_idx_list) if context == name]
                    for name in ["map_one", "map_two"]
                }

                while True:
                    mini_blocks = {
                        map: self.make_semi_block(context_dist[map], context_mu[map], context_peak_idx[map], map, map_info[map], return_mini_blocks = True)
                        for map in ["map_one", "map_two"]
                    }
                    block_design_file = []
                    for map in map_list:
                        block_design_file.extend(mini_blocks[map].pop(0))

                    # Check the final mixed order, including the previous block boundary.
                    previous_trial = choice_design_file[-1][-1:] if choice_design_file else []
                    trials = previous_trial + block_design_file
                    if all(
                        not ({a['left_choice'], a['right_choice']} &
                             {b['left_choice'], b['right_choice']})
                        for a, b in zip(trials, trials[1:])
                    ):
                        break

                choice_design_file.append(block_design_file)

        # reassign trialID for each trial in the final design file, 1-84
        for block in choice_design_file:
            for trial_id, trial in enumerate(block, start=1):
                trial['trialID'] = trial_id

        return choice_design_file
                       