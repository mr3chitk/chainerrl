"""An example of training A3C against OpenAI Gym Envs.

This script is an example of training a A3C agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.

To solve CartPole-v0, run:
    python train_a3c_gym.py 8 --env CartPole-v0

To solve InvertedPendulum-v1, run:
    python train_a3c_gym.py 8 --env InvertedPendulum-v1 --arch LSTMGaussian --t-max 50  # noqa
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import argparse

import chainer
from chainer import functions as F
from chainer import links as L
import gym
import gym.wrappers
import numpy as np

from chainerrl.agents import a3c
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl import v_function

def phi(obs):
    return [obs.astype(np.float32)]

class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.SoftmaxPolicy(model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CFFMellowmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward mellowmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.MellowmaxPolicy(model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CLSTMGaussian(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):
    """An example of A3C recurrent Gaussian policy."""

    def __init__(self, obs_size, action_size, hidden_size=200, lstm_size=128):
        self.pi_head = L.Linear(obs_size, hidden_size)
        self.v_head = L.Linear(obs_size, hidden_size)
        self.pi_lstm = L.LSTM(hidden_size, lstm_size)
        self.v_lstm = L.LSTM(hidden_size, lstm_size)
        self.pi = policies.LinearGaussianPolicyWithDiagonalCovariance(lstm_size, action_size)
        self.v = v_function.FCVFunction(lstm_size)
        super().__init__(self.pi_head, self.v_head, self.pi_lstm, self.v_lstm, self.pi, self.v)

    def pi_and_v(self, state):

        def forward(head, lstm, tail):
            h = F.relu(head(state))
            h = lstm(h)
            return tail(h)

        pout = forward(self.pi_head, self.pi_lstm, self.pi)
        vout = forward(self.v_head, self.v_lstm, self.v)

        return pout, vout

#moved out and added args as an param
def make_env(process_idx, test, args):
    if args is None:
        raise Exception(__name__ + ":make_env:Invalid args")
    env = gym.make(args.env)
    if args.monitor and process_idx == 0:
        env = gym.wrappers.Monitor(env, args.outdir)
    if not test:
        misc.env_modifiers.make_reward_filtered(env, lambda x: x * args.reward_scale_factor)
    if args.render and process_idx == 0 and not test:
        misc.env_modifiers.make_rendered(env)
    return env

def make_agent(obs_size, action_space, args):
    if args.arch == 'LSTMGaussian':
        model = A3CLSTMGaussian(obs_size, action_space.low.size)
    elif args.arch == 'FFSoftmax':
        model = A3CFFSoftmax(obs_size, action_space.n)
    elif args.arch == 'FFMellowmax':
        model = A3CFFMellowmax(obs_size, action_space.n)

    opt = rmsprop_async.RMSpropAsync(lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    agent = a3c.A3C(model, opt, t_max=args.t_max, gamma=0.99, beta=args.beta, phi=phi)
    if args.load:
        agent.load(args.load)
    return agent

def main():
    import logging
    
    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--arch', type=str, default='FFSoftmax', choices=('FFSoftmax', 'FFMellowmax', 'LSTMGaussian'))
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 6)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-1)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.INFO)
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()

    #config logging
    logger = logging.getLogger()
    logger.setLevel(args.logger_level)
    logger.handlers = []
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    sample_env = gym.make(args.env)
    timestep_limit = sample_env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    obs_size = sample_env.observation_space.low.size
    action_space = sample_env.action_space

    if args.demo:
        env = make_env(0, True, args)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=make_agent(obs_size,action_space, args),
            n_runs=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_async(
            make_agent=make_agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            max_episode_len=timestep_limit,
            full_args=args,
            obs_size=obs_size, 
            action_space=action_space
            )   

if __name__ == '__main__':
    main()
