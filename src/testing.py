from env import Env, EnvWithBlocker, EnvSR
import pickle
import torch
import numpy as np
from plotting import create_ploting_entry, save_ploting_data
import random

def evaluate_runner(T, agent, input_size, metrics_data, val_mem, device, render=False, evaluations_cnt=250, draw_cnt=1):
    print(f"Evaluating runner 🐛")
    metrics_data['Steps'].append(T)
    
    T_length                = [0] * evaluations_cnt
    T_checkpoints_completed = [0] * evaluations_cnt
    T_reward                = [0] * evaluations_cnt
    T_Qs = agent.evaluate_q(val_mem).cpu().detach().numpy().tolist()

    states   = [[Env(device).reset(), False] for _ in range(evaluations_cnt)]
    envs     = [pickle.loads(pickle.dumps(env)) for env, _ in states]
    done_cnt = 0

    gamma = 1.0
    steps = 0
    while done_cnt < evaluations_cnt:
        steps += 1
        inputs = torch.zeros((evaluations_cnt - done_cnt, input_size), device=device)
        cnt = 0
        for env, done in states:
            if not done:
                inputs[cnt] = env.get_state()
                cnt += 1

        actions = agent.act_many(inputs)

        cnt = 0
        for i in range(evaluations_cnt):
            env, done = states[i]
            if not done:
                T_length[i] += 1
                env, reward, done = env.step(actions[cnt])
                states[i] = [env, done]
                T_reward[i] += reward * gamma
                cnt += 1
                if done:
                    done_cnt += 1
                    T_checkpoints_completed[i] = env.pod.points

        gamma *= 0.99

    def draw(env):
        done = False

        while not done:
            state = env.get_state()
            action = agent.act(state)
            env, _, done = env.step(action)
            env.render()

    if render:
        order = np.argsort(np.array(T_reward))

        env.open()
        for i in range(draw_cnt):
            draw(envs[order[i]])

        for i in range(draw_cnt):
            draw(envs[order[-i-1]])
        env.close()

    agent.save_models(T)
    # agent.save_weights(name)

    # Append to results and save metrics
    metrics_data['Length'].append(T_length)
    metrics_data['Checkpoints completed'].append(T_checkpoints_completed)
    metrics_data['Qs'].append(T_Qs)
    metrics_data['Reward'].append(T_reward)
    metrics_data['TD-error'].append(agent.memory.get_avg_td_error())

    ploting_entries = []

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Game length', 'Steps', 'Length', x='Steps', y='Length',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Qs values', 'Steps', 'Qs', x='Steps', y='Qs',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Rewards', 'Steps', 'reward', x='Steps', y='Reward',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Checkpoints completed', 'Steps', 'Checkpoints completed', x='Steps', y='Checkpoints completed',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Average TD-error', 'Steps', 'TD-error', x='Steps', y='TD-error',
        plot_min=False, plot_max=False, plot_mean=False, plot_std=False, raw_data=True))

    save_ploting_data(agent.name, ploting_entries)

def evaluate_blocker(T, agent, runner, input_size_runner, input_size_blocker, metrics_data, val_mem, device, render=False, evaluations_cnt=250, draw_cnt=1):
    print(f"Evaluating blocker 🐛")
    metrics_data['Steps'].append(T)
    
    T_length                = [0] * evaluations_cnt
    T_checkpoints_completed = [0] * evaluations_cnt
    T_reward_runner         = [0] * evaluations_cnt
    T_reward_blocker        = [0] * evaluations_cnt
    T_collisions            = [0] * evaluations_cnt
    T_shield_blocker        = [0] * evaluations_cnt
    T_Qs = agent.evaluate_q(val_mem).cpu().detach().numpy().tolist()

    LegalActionsBlocker = EnvWithBlocker(device).LegalActionsBlocker

    states   = [[EnvWithBlocker(device).reset(), False] for _ in range(evaluations_cnt)]
    envs     = [pickle.loads(pickle.dumps(env)) for env, _ in states]
    done_cnt = 0

    gamma = 1.0
    steps = 0
    while done_cnt < evaluations_cnt:
        steps += 1
        inputs_runner = torch.zeros(
            (evaluations_cnt - done_cnt, input_size_runner), device=device)
        inputs_blocker = torch.zeros(
            (evaluations_cnt - done_cnt, input_size_blocker), device=device)
        cnt = 0
        for env, done in states:
            if not done:
                inputs_runner[cnt] = env.get_state_runner_against_blocker()
                inputs_blocker[cnt] = env.get_state_blocker()
                cnt += 1

        actions_runner = runner.act_many(inputs_runner)
        actions_blocker = agent.act_many(inputs_blocker)

        cnt = 0
        for i in range(evaluations_cnt):
            env, done = states[i]
            if not done:
                T_length[i] += 1
                env, reward_runner, reward_blocker, done, collisions_cnt = env.step(
                    actions_runner[cnt], actions_blocker[cnt])
                    
                states[i] = [env, done]
                T_reward_runner[i] += reward_runner * gamma
                T_reward_blocker[i] += reward_blocker
                T_collisions[i] += collisions_cnt

                if LegalActionsBlocker[actions_blocker[cnt]][1] == None:
                    T_shield_blocker[i] += 1
                cnt += 1
                if done:
                    done_cnt += 1
                    T_checkpoints_completed[i] = env.runner.points

        gamma *= 0.99

    def draw(env):
        done = False

        while not done:
            state_runner = env.get_state_runner_against_blocker()
            state_blocker = env.get_state_blocker()

            action_runner = runner.act(state_runner)
            action_blocker = agent.act(state_blocker)

            env, _, _, done, _ = env.step(action_runner, action_blocker)
            env.render()

    if render:
        order = np.argsort(np.array(T_reward_runner))

        env.open()
        for i in range(draw_cnt):
            draw(envs[order[i]])

        for i in range(draw_cnt):
            draw(envs[order[-i-1]])
        env.close()

    agent.save_models(T)

    # Append to results and save metrics
    metrics_data['Length'].append(T_length)
    metrics_data['Checkpoints completed'].append(T_checkpoints_completed)
    metrics_data['Qs'].append(T_Qs)
    metrics_data['Reward blocker'].append(T_reward_blocker)
    metrics_data['Reward runner'].append(T_reward_runner)
    metrics_data['Collisions'].append(T_collisions)
    metrics_data['Shields'].append(T_shield_blocker)
    metrics_data['TD-error'].append(agent.memory.get_avg_td_error())
    metrics_data['Collision reward'].append(EnvWithBlocker.collision_reward)

    ploting_entries = []

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Game length', 'Steps', 'Length', x='Steps', y='Length',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Checkpoints completed', 'Steps', 'Checkpoints completed', x='Steps', y='Checkpoints completed',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Qs blocker values', 'Steps', 'Qs', x='Steps', y='Qs',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Rewards blocker', 'Steps', 'reward', x='Steps', y='Reward blocker',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Rewards runner', 'Steps', 'reward', x='Steps', y='Reward runner',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Collisions', 'Steps', 'Collisions', x='Steps', y='Collisions',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Average Shields number', 'Steps', 'Shields', x='Steps', y='Shields',
        plot_min=False, plot_max=False, plot_mean=True, plot_std=False))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Average TD-error', 'Steps', 'TD-error', x='Steps', y='TD-error',
        plot_min=False, plot_max=False, plot_mean=False, plot_std=False, raw_data=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Collision reward', 'Steps', 'Collision reward', x='Steps', y='Collision reward',
        plot_min=False, plot_max=False, plot_mean=False, plot_std=False, raw_data=True))

    save_ploting_data(agent.name, ploting_entries)

def evaluate_runner_against_blocker(T, agent, blocker, input_size_runner, input_size_blocker, metrics_data, val_mem, device, render=False, evaluations_cnt=250, draw_cnt=1):
    print(f"Evaluating runner 🐛")
    metrics_data['Steps'].append(T)
    
    T_length                = [0] * evaluations_cnt
    T_checkpoints_completed = [0] * evaluations_cnt
    T_reward_runner         = [0] * evaluations_cnt
    T_reward_blocker        = [0] * evaluations_cnt
    T_shield_runner         = [0] * evaluations_cnt
    T_collisions            = [0] * evaluations_cnt
    T_Qs = agent.evaluate_q(val_mem).cpu().detach().numpy().tolist()

    states   = [[EnvWithBlocker(device).reset(), False] for _ in range(evaluations_cnt)]
    envs     = [pickle.loads(pickle.dumps(env)) for env, _ in states]
    done_cnt = 0

    gamma = 1.0
    steps = 0
    while done_cnt < evaluations_cnt:
        steps += 1
        inputs_runner = torch.zeros(
            (evaluations_cnt - done_cnt, input_size_runner), device=device)
        inputs_blocker = torch.zeros(
            (evaluations_cnt - done_cnt, input_size_blocker), device=device)
        cnt = 0
        for env, done in states:
            if not done:
                inputs_runner[cnt] = env.get_state_runner_against_blocker()
                inputs_blocker[cnt] = env.get_state_blocker()
                cnt += 1

        actions_runner = agent.act_many(inputs_runner)
        actions_blocker = blocker.act_many_e_greedy(inputs_blocker) #* blocker is Eps-greedy

        cnt = 0
        for i in range(evaluations_cnt):
            env, done = states[i]
            if not done:
                T_length[i] += 1
                env, reward_runner, reward_blocker, done, collisions_cnt = env.step(
                    actions_runner[cnt], actions_blocker[cnt])
                    
                states[i] = [env, done]
                T_reward_runner[i] += reward_runner * gamma
                T_reward_blocker[i] += reward_blocker
                T_collisions[i] += collisions_cnt

                if EnvWithBlocker.LegalActionsRunner[actions_runner[cnt]][1] == None:
                    T_shield_runner[i] += 1
                cnt += 1
                if done:
                    done_cnt += 1
                    T_checkpoints_completed[i] = env.runner.points

        gamma *= 0.99

    def draw(env):
        done = False

        while not done:
            state_runner = env.get_state_runner_against_blocker()
            state_blocker = env.get_state_blocker()

            action_runner = agent.act(state_runner)
            action_blocker = blocker.act(state_blocker)

            env, _, _, done, _ = env.step(action_runner, action_blocker)
            env.render()

    if render:
        order = np.argsort(np.array(T_reward_runner))

        env.open()
        for i in range(draw_cnt):
            draw(envs[order[i]])

        for i in range(draw_cnt):
            draw(envs[order[-i-1]])
        env.close()

    agent.save_models(T)

    # Append to results and save metrics
    metrics_data['Length'].append(T_length)
    metrics_data['Checkpoints completed'].append(T_checkpoints_completed)
    metrics_data['Qs'].append(T_Qs)
    metrics_data['Reward blocker'].append(T_reward_blocker)
    metrics_data['Reward runner'].append(T_reward_runner)
    metrics_data['Collisions'].append(T_collisions)
    metrics_data['Shields'].append(T_shield_runner)
    metrics_data['TD-error'].append(agent.memory.get_avg_td_error())

    ploting_entries = []

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Game length', 'Steps', 'Length', x='Steps', y='Length',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Checkpoints completed', 'Steps', 'Checkpoints completed', x='Steps', y='Checkpoints completed',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Qs runner values', 'Steps', 'Qs', x='Steps', y='Qs',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Rewards blocker', 'Steps', 'reward', x='Steps', y='Reward blocker',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Rewards runner', 'Steps', 'reward', x='Steps', y='Reward runner',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Collisions', 'Steps', 'Collisions', x='Steps', y='Collisions',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Average runner Shields number', 'Steps', 'Shields', x='Steps', y='Shields',
        plot_min=False, plot_max=False, plot_mean=True, plot_std=False))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Average TD-error', 'Steps', 'TD-error', x='Steps', y='TD-error',
        plot_min=False, plot_max=False, plot_mean=False, plot_std=False, raw_data=True))

    save_ploting_data(agent.name, ploting_entries)

def evaluate_both(T, runner, blocker, input_size_runner, input_size_blocker, metrics_data, val_mem_runner, val_mem_blocker, device, render=False, evaluations_cnt=250, draw_cnt=1):
    print(f"Evaluating both 🐛")
    metrics_data['Steps'].append(T)
    
    T_length                = [0] * evaluations_cnt
    T_checkpoints_completed = [0] * evaluations_cnt
    T_reward_runner         = [0] * evaluations_cnt
    T_reward_blocker        = [0] * evaluations_cnt
    T_shield_runner         = [0] * evaluations_cnt
    T_shield_blocker        = [0] * evaluations_cnt
    T_collisions            = [0] * evaluations_cnt
    T_Qs_runner = runner.evaluate_q(val_mem_runner).cpu().detach().numpy().tolist()
    T_Qs_blocker = blocker.evaluate_q(val_mem_blocker).cpu().detach().numpy().tolist()

    states   = [[EnvWithBlocker(device).reset(), False] for _ in range(evaluations_cnt)]
    envs     = [pickle.loads(pickle.dumps(env)) for env, _ in states]
    done_cnt = 0

    gamma = 1.0
    steps = 0
    while done_cnt < evaluations_cnt:
        steps += 1
        inputs_runner = torch.zeros(
            (evaluations_cnt - done_cnt, input_size_runner), device=device)
        inputs_blocker = torch.zeros(
            (evaluations_cnt - done_cnt, input_size_blocker), device=device)
        cnt = 0
        for env, done in states:
            if not done:
                inputs_runner[cnt] = env.get_state_runner_against_blocker()
                inputs_blocker[cnt] = env.get_state_blocker()
                cnt += 1

        actions_runner = runner.act_many(inputs_runner)
        actions_blocker = blocker.act_many(inputs_blocker)

        cnt = 0
        for i in range(evaluations_cnt):
            env, done = states[i]
            if not done:
                T_length[i] += 1
                env, reward_runner, reward_blocker, done, collisions_cnt = env.step(
                    actions_runner[cnt], actions_blocker[cnt])
                    
                states[i] = [env, done]
                T_reward_runner[i] += reward_runner * gamma
                T_reward_blocker[i] += reward_blocker * gamma
                T_collisions[i] += collisions_cnt

                if EnvWithBlocker.LegalActionsRunner[actions_runner[cnt]][1] == None:
                    T_shield_runner[i] += 1
                if EnvWithBlocker.LegalActionsBlocker[actions_blocker[cnt]][1] == None:
                    T_shield_blocker[i] += 1
                cnt += 1
                if done:
                    done_cnt += 1
                    T_checkpoints_completed[i] = env.runner.points

        gamma *= 0.99

    def draw(env):
        done = False

        while not done:
            state_runner = env.get_state_runner_against_blocker()
            state_blocker = env.get_state_blocker()

            action_runner = runner.act(state_runner)
            action_blocker = blocker.act(state_blocker)

            env, _, _, done, _ = env.step(action_runner, action_blocker)
            env.render()

    if render:
        env.open()

        order = np.argsort(np.array(T_reward_runner))
        for i in range(draw_cnt):
            draw(envs[order[i]])

        for i in range(draw_cnt):
            draw(envs[order[-i-1]])

        order = np.argsort(np.array(T_reward_blocker))
        for i in range(draw_cnt):
            draw(envs[order[i]])

        for i in range(draw_cnt):
            draw(envs[order[-i-1]])

        env.close()

    runner.save_models(T)
    blocker.save_models(T)

    # Append to results and save metrics
    metrics_data['Length'].append(T_length)
    metrics_data['Checkpoints completed'].append(T_checkpoints_completed)
    metrics_data['Qs runner'].append(T_Qs_runner)
    metrics_data['Qs blocker'].append(T_Qs_blocker)
    metrics_data['Reward blocker'].append(T_reward_blocker)
    metrics_data['Reward runner'].append(T_reward_runner)
    metrics_data['Collisions'].append(T_collisions)
    metrics_data['Shields runner'].append(T_shield_runner)
    metrics_data['Shields blocker'].append(T_shield_blocker)
    metrics_data['TD-error runner'].append(runner.memory.get_avg_td_error())
    metrics_data['TD-error blocker'].append(blocker.memory.get_avg_td_error())
    metrics_data['Collision reward'].append(EnvWithBlocker.collision_reward)

    ploting_entries = []

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Game length', 'Steps', 'Length', x='Steps', y='Length',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Checkpoints completed', 'Steps', 'Checkpoints completed', x='Steps', y='Checkpoints completed',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Qs runner values', 'Steps', 'Qs', x='Steps', y='Qs runner',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Qs blocker values', 'Steps', 'Qs', x='Steps', y='Qs blocker',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Rewards blocker', 'Steps', 'reward', x='Steps', y='Reward blocker',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Rewards runner', 'Steps', 'reward', x='Steps', y='Reward runner',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Collisions', 'Steps', 'Collisions', x='Steps', y='Collisions',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Average runner Shields number', 'Steps', 'Shields', x='Steps', y='Shields runner',
        plot_min=False, plot_max=False, plot_mean=True, plot_std=False))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Average blocker Shields number', 'Steps', 'Shields', x='Steps', y='Shields blocker',
        plot_min=False, plot_max=False, plot_mean=True, plot_std=False))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Average TD-error runner', 'Steps', 'TD-error', x='Steps', y='TD-error runner',
        plot_min=False, plot_max=False, plot_mean=False, plot_std=False, raw_data=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Average TD-error runner', 'Steps', 'TD-error', x='Steps', y='TD-error blocker',
        plot_min=False, plot_max=False, plot_mean=False, plot_std=False, raw_data=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Collision reward', 'Steps', 'Collision reward', x='Steps', y='Collision reward',
        plot_min=False, plot_max=False, plot_mean=False, plot_std=False, raw_data=True))

    save_ploting_data(runner.name, ploting_entries)

def make_evaluation_memory_for_runner(evaluation_size, input_size, device):
    val_mem = torch.zeros((evaluation_size, input_size), device=device)

    env = Env(device)

    for T in range(evaluation_size):
        env = env.reset()
        val_mem[T] = env.get_state()

    return val_mem

def make_evaluation_memory_for_runner_against_blocker(evaluation_size, input_size, device):
    val_mem = torch.zeros((evaluation_size, input_size), device=device)

    env = EnvWithBlocker(device)

    for T in range(evaluation_size):
        env = env.reset()
        val_mem[T] = env.get_state_runner_against_blocker()

    return val_mem

def make_evaluation_memory_for_blocker(evaluation_size, input_size, device):
    val_mem = torch.zeros((evaluation_size, input_size), device=device)

    env = EnvWithBlocker(device)

    for T in range(evaluation_size):
        env = env.reset()
        val_mem[T] = env.get_state_blocker()

    return val_mem

def make_evaluation_memory_for_runner_SR(evaluation_size, input_size, device):
    val_mem = torch.zeros((evaluation_size, input_size), device=device)

    env = EnvSR(device)

    for T in range(evaluation_size):
        env = env.reset()
        val_mem[T] = env.get_state()

    return val_mem

def evaluate_runner_SR(T, agent, input_size, metrics_data, test_idx, device, render=False, evaluations_cnt=50, draw_cnt=1):
    print(f"Evaluating runner 🐛")
    metrics_data['Steps'].append(T)
    
    T_length                = [0] * evaluations_cnt
    T_checkpoints_completed = [0] * evaluations_cnt
    T_percentage_completed  = [0] * evaluations_cnt
    T_steps_sum = 0

    states   = [[EnvSR(device).reset(test_idx), False] for i in range(evaluations_cnt)]
    envs     = [pickle.loads(pickle.dumps(env)) for env, _ in states]
    done_cnt = 0

    gamma = 1.0
    steps = 0
    T_actions = [[] for _ in range(evaluations_cnt)]

    while done_cnt < evaluations_cnt:
        steps += 1
        inputs = torch.zeros((evaluations_cnt - done_cnt, input_size), device=device)
        cnt = 0
        for env, done in states:
            if not done:
                inputs[cnt] = env.get_state()
                cnt += 1

        actions = agent.act_many(inputs)

        cnt = 0
        for i in range(evaluations_cnt):
            env, done = states[i]
            if not done:
                T_length[i] += 1
                env, _, done = env.step(actions[cnt])
                T_actions[i].append(actions[cnt].item())
                states[i] = [env, done]

                cnt += 1
                T_steps_sum += 1

                if done:
                    done_cnt += 1
                    T_checkpoints_completed[i] = env.pod.points
                    T_percentage_completed[i] = 100 * env.pod.points / (env.laps_count * env.checkpoint_count)

    T_reward = []
    for i in range(evaluations_cnt):
        T_reward.append(1000 * (T_percentage_completed[i] == 100) - T_length[i])

    order = np.argsort(np.array(T_reward))
    best_idx = order[-1]
    print(f"Reward = {T_reward[best_idx]}")
    print(T_actions[best_idx])

    def draw(env):
        done = False

        while not done:
            state = env.get_state()
            action = agent.act(state)
            env, _, done = env.step(action)
            env.render()

    if render:
        env.open()
        for i in range(draw_cnt):
            draw(envs[order[i]])

        for i in range(draw_cnt):
            draw(envs[order[-i-1]])
        env.close()

    agent.save_models(T)
    # agent.save_weights(name)

    # Append to results and save metrics
    metrics_data['Length'].append(T_length)
    metrics_data['Checkpoints completed'].append(T_checkpoints_completed)
    metrics_data['Reward'].append(T_reward)
    metrics_data['Steps sum'].append(T_steps_sum)
    metrics_data['TD-error'].append(agent.memory.get_avg_td_error())
    metrics_data['Percentage completed'].append(T_percentage_completed)

    ploting_entries = []

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Game length', 'Steps', 'Length', x='Steps', y='Length',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Rewards', 'Steps', 'reward', x='Steps', y='Reward',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Checkpoints completed', 'Steps', 'Checkpoints completed', x='Steps', y='Checkpoints completed',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Percentage completed', 'Steps', 'Percentage completed', x='Steps', y='Percentage completed',
        plot_min=True, plot_max=True, plot_mean=True, plot_std=True))

    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Steps sum', 'Steps', 'Steps sum', x='Steps', y='Steps sum',
        plot_min=False, plot_max=False, plot_mean=False, plot_std=False, raw_data=True))
    
    ploting_entries.append(
        create_ploting_entry(metrics_data,
        'Average TD-error', 'Steps', 'TD-error', x='Steps', y='TD-error',
        plot_min=False, plot_max=False, plot_mean=False, plot_std=False, raw_data=True))

    save_ploting_data(agent.name, ploting_entries)

