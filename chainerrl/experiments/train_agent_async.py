from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import logging
import multiprocessing as mp
import os

from chainerrl.experiments.evaluator import AsyncEvaluator
from chainerrl.misc import async_
from chainerrl.misc import random_seed
from chainerrl import mplog

def train_loop(process_idx, env, agent, steps, outdir, counter,
               episodes_counter, training_done,
               max_episode_len=None, evaluator=None, eval_env=None,
               successful_score=None, logger=None,
               global_step_hooks=[]):

    logger = logger or logging.getLogger(__name__)

    if eval_env is None:
        eval_env = env

    try:

        total_r = 0
        episode_r = 0
        global_t = 0
        local_t = 0
        global_episodes = 0
        obs = env.reset()
        r = 0
        done = False
        episode_len = 0
        successful = False

        while True:

            total_r += r
            episode_r += r

            if done or episode_len == max_episode_len:
                with episodes_counter.get_lock():
                    episodes_counter.value += 1
                    global_episodes = episodes_counter.value
                agent.stop_episode_and_train(obs, r, done)
                if process_idx == 0:
                    logger.info(
                        'outdir:%s global_step:%s local_step:%s R:%s',
                        outdir, global_t, local_t, episode_r)
                    logger.info('statistics:%s', agent.get_statistics())
                if evaluator is not None:
                    eval_score = evaluator.evaluate_if_necessary(
                        t=global_t, episodes=global_episodes,
                        env=eval_env, agent=agent)
                    if (eval_score is not None and
                            successful_score is not None and
                            eval_score >= successful_score):
                        with training_done.get_lock():
                            if not training_done.value:
                                training_done.value = True
                                successful = True
                        # Break immediately in order to avoid an additional
                        # call of agent.act_and_train
                        break
                episode_r = 0
                obs = env.reset()
                r = 0
                done = False
                episode_len = 0
            else:
                a = agent.act_and_train(obs, r)
                obs, r, done, info = env.step(a)

                # Get and increment the global counter
                with counter.get_lock():
                    counter.value += 1
                    global_t = counter.value
                local_t += 1
                episode_len += 1

                for hook in global_step_hooks:
                    hook(env, agent, global_t)

                if global_t > steps or training_done.value:
                    break

    except (Exception, KeyboardInterrupt):
        if process_idx == 0:
            # Save the current model before being killed
            dirname = os.path.join(outdir, '{}_except'.format(global_t))
            agent.save(dirname)
            logger.warning('Saved the current model to %s', dirname)
        raise

    if global_t == steps + 1:
        # Save the final model
        dirname = os.path.join(outdir, '{}_finish'.format(steps))
        agent.save(dirname)
        logger.info('Saved the final agent to %s', dirname)

    if successful:
        # Save the successful model
        dirname = os.path.join(outdir, 'successful')
        agent.save(dirname)
        logger.info('Saved the successful agent to %s', dirname)


def extract_shared_objects_from_agent(agent):
    return dict((attr, async_.as_shared_objects(getattr(agent, attr)))
                for attr in agent.shared_attributes)


def set_shared_objects(agent, shared_objects):
    for attr, shared in shared_objects.items():
        new_value = async_.synchronize_to_shared_objects(
            getattr(agent, attr), shared)
        setattr(agent, attr, new_value)

def run_func(process_idx, 
             make_env, 
             make_agent, 
             full_args, 
             agent, 
             counter,
             episodes_counter,
             steps,
             outdir,
             max_episode_len,
             successful_score,
             training_done,
             global_step_hooks,
             profile,
             shared_objects,
             step_offset,
             eval_explorer,
             obs_size,
             action_space
             ): 
    #logging
    logger = logging.getLogger(__name__)
    
    #random seed
    random_seed.set_random_seed(process_idx)
    
    #environment
    env = make_env(process_idx, test=False, args=full_args)
    
    #evaluator
    if full_args.eval_interval is None:
        evaluator = None
    else:
        if(process_idx == 0):
            evaluator = AsyncEvaluator(
                    n_runs=full_args.eval_n_runs,
                    eval_interval=full_args.eval_interval, 
                    outdir=outdir,
                    max_episode_len=max_episode_len,
                    step_offset=step_offset,
                    explorer=eval_explorer,
                    logger=logger)
        else:
            evaluator = None
    
    if evaluator is None:
        eval_env = env
    else:
        eval_env = make_env(process_idx, test=True, args=full_args)
    
    #make_agent
    if make_agent is not None:
        local_agent = make_agent(obs_size, action_space, full_args)
    else:
        local_agent = agent
    
    #always set_shared_objects cuz windows has no fork = different agents 
    if shared_objects is not None:
        set_shared_objects(local_agent, shared_objects)
    local_agent.process_idx = process_idx
    
    #train_loop
    def f():
        train_loop(
            process_idx=process_idx,
            counter=counter,
            episodes_counter=episodes_counter,
            agent=local_agent,
            env=env,
            steps=steps,
            outdir=outdir,
            max_episode_len=max_episode_len,
            evaluator=evaluator,
            successful_score=successful_score,
            training_done=training_done,
            eval_env=eval_env,
            global_step_hooks=global_step_hooks,
            logger=logger)

    #profile
    if profile:
        import cProfile
        cProfile.runctx('f()', globals(), locals(), 'profile-{}.out'.format(os.getpid()))
    else:
        f()

def train_agent_async(outdir,                #yes
                      processes,             #yes
                      make_env,              #yes
                      profile=False,         #yes
                      steps=8 * 10 ** 7,     #yes
                      eval_interval=10 ** 6, #yes
                      eval_n_runs=10,        #yes
                      max_episode_len=None,  #yes
                      step_offset=0,         #no
                      successful_score=None, #no
                      eval_explorer=None,    #no
                      agent=None,            #yes
                      make_agent=None,       #yes
                      global_step_hooks=[],  #no
                      logger=None,           #no
                      full_args=None,        #yes
                      obs_size=None,         #yes
                      action_space=None      #yes
                      ):         
    """Train agent asynchronously using multiprocessing.

    Either `agent` or `make_agent` must be specified.

    Args:
        outdir (str): Path to the directory to output things.
        processes (int): Number of processes.
        make_env (callable): (process_idx, test) -> Environment.
        profile (bool): Profile if set True.
        steps (int): Number of global time steps for training.
        eval_interval (int): Interval of evaluation. If set to None, the agent
            will not be evaluated at all.
        eval_n_runs (int): Number of runs for each time of evaluation.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        successful_score (float): Finish training if the mean score is greater
            or equal to this value if not None
        eval_explorer: Explorer used for evaluation.
        agent (Agent): Agent to train.
        make_agent (callable): (process_idx) -> Agent
        global_step_hooks (list): List of callable objects that accepts
            (env, agent, step) as arguments. They are called every global
            step. See chainerrl.experiments.hooks.
        logger (logging.Logger): Logger used in this function.

    Returns:
        Trained agent.
    """

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    counter = mp.Value('l', 0)
    episodes_counter = mp.Value('l', 0)
    training_done = mp.Value('b', False)  # bool
    
    #async evaluator
    logger = logger or logging.getLogger(__name__)
    
    if agent is None:
        assert make_agent is not None
        agent = make_agent(obs_size, action_space, full_args)
        
    shared_objects = extract_shared_objects_from_agent(agent)
    set_shared_objects(agent, shared_objects)

    logger.info("Start Training...")
    with mplog.open_queue() as log_queue:
        async.run_async(processes, 
                        run_func, 
                        make_env, 
                        make_agent, 
                        full_args, 
                        None, #agent won't be passed in since it contains locked objects
                        counter,
                        episodes_counter,
                        steps,
                        outdir,
                        max_episode_len,
                        successful_score,
                        training_done,
                        global_step_hooks,
                        profile,
                        log_queue,
                        shared_objects,
                        step_offset,
                        eval_explorer,
                        obs_size,
                        action_space
                        )

    return agent
