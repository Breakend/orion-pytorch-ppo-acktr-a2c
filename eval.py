import numpy as np
import torch
from torch.autograd import Variable

def evaluate_with_seeds(env, policy, cuda, eval_env_seeds):
    all_rets = []
    for seed in eval_env_seeds:
        for venv in env.venv.envs:
            venv.seed(seed)

        epoch_return = rollout_episode(env, policy, cuda)
        all_rets.append(epoch_return.cpu())

    return torch.stack(all_rets).data.numpy()

def rollout_episode(env, policy, cuda):
    num_processes = 1
    num_episodes = 1
    current_state = torch.zeros(num_processes, *env.observation_space.shape)

    def update_current_state(state):
        shape_dim0 = env.observation_space.shape[0]
        state = torch.from_numpy(state).float()
        current_state[:, -shape_dim0:] = state

    state = env.reset()
    update_current_state(state)

    if cuda: current_state = current_state.cuda()

    steps = 0
    masks = torch.FloatTensor(
        [[0.0]])

    internal_states= torch.zeros([1,policy.state_size])

    episode_return = torch.zeros([1, 1])
    rewards = []
    actions = []
    while True:
        steps += num_processes

        value, action, action_log_prob, internal_states = policy.act(Variable(current_state), Variable(internal_states), Variable(masks), deterministic=True)
        cpu_actions = action.data.squeeze(1).cpu().numpy()
        state, reward, done, info = env.step(cpu_actions)

        rewards.append(reward)
        actions.append(cpu_actions)

        # Expand the reward
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

        episode_return += reward

        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])

        # GPU mem
        if cuda:
            masks = masks.cuda()

        # may need to modify the state if we're using pixels
        if current_state.dim() == 4:
            current_state *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_state *= masks

        update_current_state(state)

        if done:
            break

    return episode_return
