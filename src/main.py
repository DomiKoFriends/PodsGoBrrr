from model import DQN
from env import Env, EnvSR, EnvWithBlocker
import numpy as np
import torch
from agent import DQNAgent, DoubleDQNAgent, DoubleDQNAgentPrioritized
from os import path
import testing


def check_model_name(name):
    if path.exists(path.join(".", "models", name)):
        print(f"Model {name} already exists.\nAre you sure about this? (Y/N)")
        decision = input()
        if decision.lower() in ['n', 'no']:
            print('Quiting')
            exit(0)
        elif not decision.lower() in ['y', 'yes']:
            print('OK then...\n💀 [*] Old model\n')
            exit(0)


def train_runner():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device} ⚙️")

    metrics_data = {'Steps': [], 'Length': [], 'Checkpoints completed': [], 'Reward': [],
                    'Qs': [], 'TD-error': []}

    name = 'who cares'

    check_model_name(name)

    learn_start = 1000
    T_max = 1000000
    eval_per = 10000
    learn_per = 4
    update_frozen = 500
    render_per = T_max + 1  # eval_per * 10

    input_size = Env(device).reset().get_state().shape[0]

    agent = DoubleDQNAgentPrioritized(agent_name=name,
                                      gamma=0.95,
                                      epsilon_start=1.0,
                                      epsilon_min=0.05,
                                      epsilon_decay=1e5,
                                      lr=1e-4,
                                      n_actions=len(Env.LegalActions),
                                      input_dim=input_size,
                                      mem_size=2**18,
                                      batch_size=64,
                                      clip_value=1.0,
                                      device=device)

    # agent.load_models(1250000, 'runner4', True)

    val_mem = testing.make_evaluation_memory_for_runner(
        1000, input_size, device)

    testing.evaluate_runner(0, agent, input_size,
                            metrics_data, val_mem, device)

    env, done = Env(device).reset(), False
    observation = env.get_state()
    games_cnt = 1

    for t in range(1, T_max + 1):
        if t % 500 == 0:
            print(
                f'step = {t} eps = {agent.epsilon} games cnt = {games_cnt} 🐌')

        if done:
            env, done = env.reset(), False
            games_cnt += 1

        action = agent.act_e_greedy(observation)

        env, reward, done = env.step(action)

        next_observation = env.get_state()

        agent.store_transition(observation, action,
                               reward, next_observation, done)

        if t >= learn_start:
            if t % eval_per == 0:
                agent.eval()

                testing.evaluate_runner(t, agent, input_size, metrics_data,
                                        val_mem, device, render=(t % render_per == 0))

                agent.train()

            if t % learn_per == 0:
                agent.learn()
                agent.decrease_epsilon()

            if t % update_frozen == 0:
                agent.update_frozen_net()

        observation = next_observation


def train_blocker():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device} ⚙️")

    metrics_data = {'Steps': [], 'Length': [], 'Checkpoints completed': [], 'Qs': [],
                    'Reward blocker': [], 'Reward runner': [], 'Collisions': [],
                    'Shields': [], 'TD-error': [], 'Collision reward': []}

    name = 'blocker_good'
    runner_name = 'who cares'

    check_model_name(name)
    check_model_name(runner_name)

    learn_start = 10000
    T_max = 10 ** 8
    eval_per = 50000
    learn_per = 4
    update_frozen = 500
    render_per = eval_per * 10

    input_size_runner = EnvWithBlocker(
        device).reset().get_state_runner_against_blocker().shape[0]
    input_size_blocker = EnvWithBlocker(
        device).reset().get_state_blocker().shape[0]

    agent_runner = DoubleDQNAgentPrioritized(gamma=0.95,
                                             epsilon_start=0.01,
                                             epsilon_min=0.01,
                                             epsilon_decay=1e5,
                                             lr=1e-4,
                                             n_actions=len(
                                                 EnvWithBlocker(device).LegalActionsRunner),
                                             input_dim=input_size_runner,
                                             mem_size=2**18,
                                             batch_size=64,
                                             agent_name=runner_name,
                                             device=device)

    agent_runner.load_models(8000000, 'runner_blocker')
    agent_runner.train()

    agent_blocker = DoubleDQNAgentPrioritized(gamma=0.95,
                                              epsilon_start=1.0,
                                              epsilon_min=0.05,
                                              epsilon_decay=1e5,
                                              lr=1e-4,
                                              n_actions=len(
                                                  EnvWithBlocker(device).LegalActionsBlocker),
                                              input_dim=input_size_blocker,
                                              mem_size=2**18,
                                              batch_size=64,
                                              agent_name=name,
                                              device=device)

    val_mem = testing.make_evaluation_memory_for_blocker(
        1000, input_size_blocker, device)

    testing.evaluate_blocker(0, agent_blocker, agent_runner, input_size_runner, input_size_blocker,
                             metrics_data, val_mem, device)

    def update_collision_reward(steps):
        reward = 0.5 - 0.4 * (steps / 4e6)
        reward = max(0.01, reward)
        reward = min(0.5, reward)

        EnvWithBlocker.collision_reward = reward

    EnvWithBlocker.collision_reward = 0.5
    env, done = EnvWithBlocker(device).reset(), False
    games_cnt = 1

    for t in range(1, T_max + 1):
        if t % 500 == 0:
            print(
                f'step = {t} eps = {agent_blocker.epsilon} collision_reward = {EnvWithBlocker.collision_reward} 🐗 games cnt = {games_cnt} 🐌')

        update_collision_reward(t)

        if done:
            env, done = env.reset(), False
            games_cnt += 1

        observation_runner = env.get_state_runner_against_blocker()
        action_runner = agent_runner.act(observation_runner)

        observation_blocker = env.get_state_blocker()
        action_blocker = agent_blocker.act_e_greedy(observation_blocker)

        env, _, reward_blocker, done, _ = env.step(
            action_runner, action_blocker)

        agent_blocker.store_transition(observation_blocker, action_blocker,
                                       reward_blocker, env.get_state_blocker(), done)

        if t >= learn_start:
            if t % eval_per == 0:
                agent_blocker.eval()

                testing.evaluate_blocker(t, agent_blocker, agent_runner, input_size_runner, input_size_blocker,
                                         metrics_data, val_mem, device, render=(t % render_per == 0))

                agent_blocker.train()

            if t % learn_per == 0:
                agent_blocker.learn()
                agent_blocker.decrease_epsilon()

            if t % update_frozen == 0:
                agent_blocker.update_frozen_net()


def train_runner_against_blocker():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device} ⚙️")

    metrics_data = {'Steps': [], 'Length': [], 'Checkpoints completed': [], 'Qs': [],
                    'Reward blocker': [], 'Reward runner': [], 'Collisions': [],
                    'Shields': [], 'TD-error': []}

    runner_name = 'runner_blocker_3'
    blocker_name = 'blocker_2'

    check_model_name(runner_name)

    learn_start = 10000
    T_max = 10 ** 8
    eval_per = 50000
    learn_per = 4
    update_frozen = 500
    render_per = eval_per * 10

    input_size_runner = EnvWithBlocker(
        device).reset().get_state_runner_against_blocker().shape[0]
    input_size_blocker = EnvWithBlocker(
        device).reset().get_state_blocker().shape[0]

    agent_runner = DoubleDQNAgentPrioritized(gamma=0.95,
                                             epsilon_start=0.05,
                                             epsilon_min=0.05,
                                             epsilon_decay=1e4,
                                             lr=1e-4,
                                             n_actions=len(
                                                 EnvWithBlocker(device).LegalActionsRunner),
                                             input_dim=input_size_runner,
                                             mem_size=2**18,
                                             batch_size=64,
                                             agent_name=runner_name,
                                             device=device)

    agent_runner.load_models(3000000, 'runner_blocker', True)

    agent_blocker = DoubleDQNAgentPrioritized(gamma=0.95,
                                              epsilon_start=0.5,
                                              epsilon_min=0.01,
                                              epsilon_decay=1e5,
                                              lr=1e-5,
                                              n_actions=len(
                                                  EnvWithBlocker(device).LegalActionsBlocker),
                                              input_dim=input_size_blocker,
                                              mem_size=2**18,
                                              batch_size=64,
                                              agent_name='who cares',
                                              device=device)

    agent_blocker.load_models(5925000, blocker_name)
    agent_blocker.train()

    val_mem = testing.make_evaluation_memory_for_runner_against_blocker(
        1000, input_size_runner, device)

    testing.evaluate_runner_against_blocker(0, agent_runner, agent_blocker, input_size_runner, input_size_blocker,
                                            metrics_data, val_mem, device)

    EnvWithBlocker.collision_reward = 0.5
    env, done = EnvWithBlocker(device).reset(), False
    games_cnt = 1

    for t in range(1, T_max + 1):
        if t % 500 == 0:
            print(
                f'step = {t} eps = {agent_runner.epsilon} {agent_blocker.epsilon}games cnt = {games_cnt} 🐌')
        if done:
            env, done = env.reset(), False
            games_cnt += 1

        observation_runner = env.get_state_runner_against_blocker()
        action_runner = agent_runner.act_e_greedy(observation_runner)

        observation_blocker = env.get_state_blocker()
        action_blocker = agent_blocker.act_e_greedy(observation_blocker)

        env, reward_runner, _, done, _ = env.step(
            action_runner, action_blocker)

        agent_runner.store_transition(observation_runner, action_runner,
                                      reward_runner, env.get_state_runner_against_blocker(), done)

        if t >= learn_start:
            if t % eval_per == 0:
                agent_runner.eval()

                testing.evaluate_runner_against_blocker(
                    t, agent_runner, agent_blocker, input_size_runner, input_size_blocker, metrics_data, val_mem, device, render=(t % render_per == 0))

                agent_runner.train()

            if t % learn_per == 0:
                agent_runner.learn()
                agent_runner.decrease_epsilon()

                agent_blocker.decrease_epsilon()

            if t % update_frozen == 0:
                agent_runner.update_frozen_net()


def train_both():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device} ⚙️")

    metrics_data = {'Steps': [], 'Length': [], 'Checkpoints completed': [], 'Qs runner': [], 'Qs blocker': [],
                    'Reward blocker': [], 'Reward runner': [], 'Collisions': [], 'Collision reward': [],
                    'Shields runner': [], 'Shields blocker': [], 'TD-error runner': [], 'TD-error blocker': []}

    runner_name = 'runner_both'
    blocker_name = 'blocker_both'

    check_model_name(runner_name)
    check_model_name(blocker_name)

    learn_start = 10000
    T_max = 10 ** 8
    eval_per = 50000
    learn_per = 4
    update_frozen = 500
    render_per = eval_per * 20

    input_size_runner = EnvWithBlocker(
        device).reset().get_state_runner_against_blocker().shape[0]
    input_size_blocker = EnvWithBlocker(
        device).reset().get_state_blocker().shape[0]

    agent_runner = DoubleDQNAgentPrioritized(gamma=0.95,
                                             epsilon_start=1.0,
                                             epsilon_min=0.05,
                                             epsilon_decay=5e4,
                                             lr=1e-4,
                                             n_actions=len(
                                                 EnvWithBlocker(device).LegalActionsRunner),
                                             input_dim=input_size_runner,
                                             mem_size=2**18,
                                             batch_size=64,
                                             agent_name=runner_name,
                                             device=device)

    agent_blocker = DoubleDQNAgentPrioritized(gamma=0.95,
                                              epsilon_start=1.0,
                                              epsilon_min=0.05,
                                              epsilon_decay=1e5,
                                              lr=1e-4,
                                              n_actions=len(
                                                  EnvWithBlocker(device).LegalActionsBlocker),
                                              input_dim=input_size_blocker,
                                              mem_size=2**18,
                                              batch_size=64,
                                              agent_name=blocker_name,
                                              device=device)

    val_mem_runner = testing.make_evaluation_memory_for_runner_against_blocker(
        1000, input_size_runner, device)

    val_mem_blocker = testing.make_evaluation_memory_for_blocker(
        1000, input_size_blocker, device)

    testing.evaluate_both(0, agent_runner, agent_blocker, input_size_runner, input_size_blocker,
                          metrics_data, val_mem_runner, val_mem_blocker, device)

    def update_collision_reward(steps):
        reward = 0.5 - 0.4 * ((steps - 2e6) / 4e6)
        reward = max(0.1, reward)
        reward = min(0.5, reward)

        EnvWithBlocker.collision_reward = reward

    EnvWithBlocker.collision_reward = 0.5
    env, done = EnvWithBlocker(device).reset(), False
    games_cnt = 1

    for t in range(1, T_max + 1):
        if t % 500 == 0:
            print(
                f'step = {t} eps runner = {agent_runner.epsilon} eps blocker = {agent_blocker.epsilon} games cnt = {games_cnt} 🐌')
        if done:
            env, done = env.reset(), False
            games_cnt += 1

        observation_runner = env.get_state_runner_against_blocker()
        action_runner = agent_runner.act_e_greedy(observation_runner)

        observation_blocker = env.get_state_blocker()
        action_blocker = agent_blocker.act_e_greedy(observation_blocker)

        env, reward_runner, reward_blocker, done, _ = env.step(
            action_runner, action_blocker)

        agent_runner.store_transition(observation_runner, action_runner,
                                      reward_runner, env.get_state_runner_against_blocker(), done)

        agent_blocker.store_transition(observation_blocker, action_blocker,
                                       reward_blocker, env.get_state_blocker(), done)

        if t >= learn_start:
            if t % eval_per == 0:
                agent_runner.eval()
                agent_blocker.eval()

                testing.evaluate_both(
                    t, agent_runner, agent_blocker, input_size_runner, input_size_blocker,
                    metrics_data, val_mem_runner, val_mem_blocker, device, render=(t % render_per == 0))

                agent_runner.train()
                agent_blocker.train()

            if t % learn_per == 0:
                agent_runner.learn()
                agent_runner.decrease_epsilon()

                agent_blocker.learn()
                agent_blocker.decrease_epsilon()

            if t % update_frozen == 0:
                agent_runner.update_frozen_net()

                agent_blocker.update_frozen_net()


def train_runner_SR():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu' #? cpu is actually faster than cuda
    
    print(f"device: {device} ⚙️")

    metrics_data = {'Steps': [], 'Length': [], 'Checkpoints completed': [], 'Reward': [],
                     'TD-error': [], 'Steps sum':[], 'Percentage completed':[]}

    name = 'runner_sr_0'

    check_model_name(name)

    test_idx = 0
    learn_start = 1000
    T_max = 10**8
    eval_per = 25000
    learn_per = 4
    update_frozen = 100
    render_per = eval_per * 10

    input_size = EnvSR(device).reset().get_state().shape[0]

    agent = DoubleDQNAgentPrioritized(agent_name=name,
                                      gamma=0.95,
                                      epsilon_start=1.0,
                                      epsilon_min=0.01,
                                      epsilon_decay=5e4,
                                      lr=1e-4,
                                      n_actions=len(EnvSR.LegalActions),
                                      input_dim=input_size,
                                      mem_size=2**14,
                                      batch_size=32,
                                      clip_value=1.0,
                                      device=device)

    testing.evaluate_runner_SR(0, agent, input_size,
                            metrics_data, test_idx, device)

    env, done = EnvSR(device).reset(), False
    observation = env.get_state()
    games_cnt = 1

    for t in range(1, T_max + 1):
        if t % 500 == 0:
            print(
                f'step = {t} eps = {agent.epsilon} games cnt = {games_cnt} 🐌')

        if done:
            env, done = env.reset(), False
            games_cnt += 1

        action = agent.act_e_greedy(observation)

        env, reward, done = env.step(action)

        next_observation = env.get_state()

        agent.store_transition(observation, action,
                               reward, next_observation, done)

        if t >= learn_start:
            if t % eval_per == 0:
                agent.eval()

                testing.evaluate_runner_SR(t, agent, input_size, metrics_data,
                                        test_idx, device, render=(t % render_per == 0))

                agent.train()

            if t % learn_per == 0:
                agent.learn()
                agent.decrease_epsilon()

            if t % update_frozen == 0:
                agent.update_frozen_net()

        observation = next_observation


if __name__ == '__main__':
    train_runner()
    # train_blocker()
    # train_runner_against_blocker()
    # train_both()
    # train_runner_SR()
