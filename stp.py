import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import os
from datetime import datetime


class Belief:
    def __init__(self):
        self.successes = 1
        self.failures = 1
        self.data = []

    def update(self, sat):
        if sat:
            self.successes += 1
        else:
            self.failures += 1

    def dist(self):
        return beta(self.successes, self.failures)

    def sample(self):
        return beta(self.successes, self.failures).rvs()

    def std(self):
        return beta(self.successes, self.failures).std()

    def mode(self):
        return self.successes / (self.successes + self.failures)

    def mean(self):
        return beta(self.successes, self.failures).mean()

    def median(self):
        return beta(self.successes, self.failures).median()

    def cv(self):
        return self.std() / self.mean()

    def ppf(self, q):
        return beta(self.successes, self.failures).ppf(q)

    def print_stats(self):
        print('{:.2} {:.2} {:.2} {:.2}'.format(self.std(), self.cv(), self.median(), self.ppf(0.1)))


class Vanilla:
    def __init__(self):
        self.planning_depth = planning_depth
        self.best_plan = None
        self.best_value = 0.

    def get_random_plan(self):
        return [['up', 'down', 'left', 'right'][np.random.randint(0, 4)] for _ in range(self.planning_depth)]

    def update(self, plan, value):
        if value > self.best_value:
            self.best_plan = plan


class Planner:
    def __init__(self):
        self.planning_depth = planning_depth
        self.safety_belief_dists = [self.get_prior_safety_belief() for _ in range(0, self.planning_depth)]

    def get_plan(self, function):
        plan = []
        for d in range(0, self.planning_depth):
            values = dict()
            for action, belief in self.safety_belief_dists[d].items():
                values[action] = function(belief)
            # get action with maximum sampled value
            max_action = max(values.keys(), key=(lambda k: values[k]))
            plan.append(max_action)
        return plan

    def update_beliefs(self, plan, rewards):
        sat = np.sum(rewards) >= required_reward
        for d, action in enumerate(plan):
            self.safety_belief_dists[d][action].update(sat)

    def update_beliefs_markovian(self, plan, rewards):
        inverse_cumulative_rewards = np.cumsum(rewards[::-1])[::-1]
        for d, action in enumerate(plan):
            sat = inverse_cumulative_rewards[d] >= required_reward
            self.safety_belief_dists[d][action].update(sat)

    def update_beliefs_per_action(self, plan, rewards):
        for d, action in enumerate(plan):
            self.safety_belief_dists[d][action].update(rewards[d] + 1)

    def get_prior_safety_belief(self):
        return dict(right=Belief(), left=Belief(), up=Belief(), down=Belief())


class Agent:
    def __init__(self):
        self.pos = [0, 0]
        self.p_fail = np.random.uniform(0, 1)
        self.actions = dict(right=(1, 0), left=(-1, 0), up=(0, 1), down=(0, -1))

    def update(self, action):
        action = self.actions[action]
        if np.random.uniform(0, 1) <= self.p_fail:
            action = (action[0] * -1, action[1] * -1)
        self.pos[0] += action[0]
        self.pos[1] += action[1]


class World:
    def __init__(self, x_dim=10, y_dim=10):
        self.grid = np.random.rand(x_dim, y_dim)
        self.init_grid()
        self.agent = Agent()

    def init_grid(self):
        for x in range(len(self.grid)):
            for y in range(len(self.grid)):
                self.grid[x, y] = np.random.choice([0, -1], p=[0.8, 0.2])

    def update(self, action):
        self.agent.update(action)
        self.agent.pos[0] = np.clip(self.agent.pos[0], 0, 9)
        self.agent.pos[1] = np.clip(self.agent.pos[1], 0, 9)
        reward = self.grid[self.agent.pos[0], self.agent.pos[1]]
        return reward

    def execute_plan(self, plan):
        rewards = []
        for action in plan:
            r = self.update(action)
            rewards.append(r)
        return rewards


def estimate_satisfaction_probability(world, plan):
    plan_sats = []
    for j in range(0, 1000):
        world_copy = copy.deepcopy(world)
        rewards = world_copy.execute_plan(plan)
        sat = np.sum(rewards) >= required_reward
        plan_sats.append(sat)
    return np.mean(plan_sats)


directory = "plots/bayes_mc_discrete_seq_markov/{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(directory):
    os.makedirs(directory)

np.random.seed(424242)
# np.random.seed(123123)
plt.ion()
plt.show()

planning_depth = 10
required_reward = -2

sats = []
estimated_sats = []
estimated_sats_best = []
sampled_tails = []
best_plan_lower_tails = []
best_plan_upper_tails = []
cvs = []
modes = []
best_cvs = []
best_modes = []

world = World()
print('p fail {}'.format(world.agent.p_fail))

vanilla = Vanilla()
max_p = 0.
for i in range(100):
    print('vanilla', i, end="\r")
    plan = vanilla.get_random_plan()
    p = estimate_satisfaction_probability(world, plan)
    vanilla.update(plan, p)
    if p > max_p:
        max_p = p

planner = Planner()
for i in range(0, 10000):
    plan = planner.get_plan(Belief.sample)

    world_copy = copy.deepcopy(world)
    rewards = world_copy.execute_plan(plan)

    sat = np.sum(rewards) >= required_reward
    sats.append(sat)

    # TODO: compare Markovian vs. global (i.e. non-Markovian) belief updates
    planner.update_beliefs(plan, rewards)
    # planner.update_beliefs_markovian(plan, rewards)
    # planner.update_beliefs_per_action(plan, rewards)

    estimated_sats.append(estimate_satisfaction_probability(world, plan))
    mode = np.mean([planner.safety_belief_dists[i][a].mode() for i, a in enumerate(plan)])
    modes.append(mode)
    cv = np.mean([planner.safety_belief_dists[i][a].std() / planner.safety_belief_dists[i][a].mode() for i, a in
           enumerate(plan)])
    cvs.append(cv)

    best_plan = planner.get_plan(Belief.mode)
    estimated_sats_best.append(estimate_satisfaction_probability(world, best_plan))
    best_mode = np.mean([planner.safety_belief_dists[i][a].mode() for i, a in enumerate(best_plan)])
    best_modes.append(best_mode)
    cv = [planner.safety_belief_dists[i][a].std() / planner.safety_belief_dists[i][a].mode() for i, a in
           enumerate(best_plan)]
    cv = np.mean(cv)
    best_cvs.append(cv)

    sampled_tails.append(
        np.min([planner.safety_belief_dists[d][action].ppf(0.1) for d, action in enumerate(plan)]))
    '''
    best_plan_lower_tails.append(
        np.min([planner.safety_belief_dists[d][action].ppf(0.1) for d, action in enumerate(plan_to_evaluate)]))
    best_plan_upper_tails.append(
        np.min([planner.safety_belief_dists[d][action].ppf(0.9) for d, action in enumerate(plan_to_evaluate)]))
    '''
    best_plan_lower_tails.append(planner.safety_belief_dists[0][best_plan[0]].ppf(0.1))
    best_plan_upper_tails.append(planner.safety_belief_dists[0][best_plan[0]].ppf(0.9))

    if i % 10 == 0:
        '''
        plt.figure(1)
        plt.clf()
        plt.scatter(range(len(sats)), sats, color=(1, 0.5, 0, 0.5), label='sat')
        fractions = [x / (n + 1) for x, n in zip(np.cumsum(sats), range(len(sats)))]
        plt.scatter(range(len(sats)), fractions, color=(0, 0.5, 1, 0.5), label='P(sat)')
        sns.despine(offset=10, trim=True)
        plt.legend(loc='best', fancybox=False, framealpha=0.5)
        plt.savefig('{}/sample.png'.format(directory))
        plt.pause(0.001)

        plt.figure(2)
        '''
        plt.figure(1)
        plt.clf()
        '''
        plt.scatter(range(len(best_plan_upper_tails)), best_plan_upper_tails,
                    color=(1, 0.5, 0, 0.5), label='$\mathrm{B}^\mathrm{0}_\mathrm{0.9}$')
        plt.scatter(range(len(best_plan_lower_tails)), best_plan_lower_tails,
                    color=(0, 0.5, 1, 0.5), label='$\mathrm{B}^\mathrm{0}_\mathrm{0.1}$')
        '''
        plt.xlim((-1, len(estimated_sats)))
        # y_max = np.max(max_p, np.max(estimated_sats))
        # print(y_max)
        # plt.ylim((np.min(estimated_sats) - 0.05, y_max + 0.05))
        plt.scatter(range(len(estimated_sats)), estimated_sats, color=(1, 0.5, 0, 0.5),
                    label='$\hat{\mathrm{P}}$(sat)')
        plt.scatter(range(len(estimated_sats_best)), estimated_sats_best, color=(0, 0.5, 1, 0.5),
                    label='$\hat{\mathrm{P}}^*$(sat)')
        # sns.despine(offset=10, trim=True)
        plt.axhline(max_p, ls='--')
        plt.legend(loc='best', fancybox=False, framealpha=0.5)
        plt.savefig('{}/estimation.png'.format(directory))
        plt.pause(0.001)

        plt.figure(2)
        plt.clf()
        plt.xlim((0, len(modes)))
        plt.plot(range(len(modes)), modes, color=(1, 0.5, 0, 1), ls="--", label="sampled avg. mode")
        plt.plot(range(len(best_modes)), best_modes, color=(0, 0.5, 1, 1), ls="--", label="best avg. mode")
        plt.plot(range(len(cvs)), cvs, color=(1, 0.5, 0, 1), label="sampled avg. CV")
        plt.plot(range(len(best_cvs)), best_cvs, color=(0, 0.5, 1, 1), label="best avg. CV")
        plt.legend(loc='best')
        plt.savefig('{}/cvs.png'.format(directory))
        plt.pause(0.001)

        '''
        plt.figure(3)
        plt.clf()
        # TODO: confidence/error bounds of min
        # TODO: compare best plan belief about p_sat with best plan empirical p_sat
        d = np.array(estimated_sats) - np.array(best_plan_lower_tails)
        overestimates = np.cumsum([x < 0 for x in d])
        fraction_of_overestimates = [overestimates[i] / (i + 1) for i in range(len(overestimates))]
        plt.scatter(range(len(d)), d, color=(1, 0.5, 0, 0.5), label='P(sat) - $\mathrm{B}^\mathrm{0}_\mathrm{0.1}$')
        plt.scatter(range(len(fraction_of_overestimates)), fraction_of_overestimates, color=(0, 0.5, 1, 0.5),
                    label='P(overestimate)')
        plt.axhline(0, color=(0, 0, 0, 1), ls='--')
        plt.axhline(0.1, color=(0, 0, 0, 1), ls='--')
        sns.despine(offset=10, trim=True)
        plt.legend(loc='best', fancybox=False, framealpha=0.5)
        plt.savefig('{}/error.png'.format(directory))
        plt.pause(0.001)
        '''

plt.ioff()
plt.show()
