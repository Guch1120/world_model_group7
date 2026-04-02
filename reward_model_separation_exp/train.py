import sys, atexit, os, random, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Normal, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
# IntrinsicRewardModelをインポート
from student_code import Agent, RSSM, Encoder, Decoder, RewardModel, IntrinsicRewardModel, DiscountModel, Actor, DiscreteActor, Critic, MSE
from exploration.cifar import create_cifar_function_simple
from exploration.noisy_wrapper import NoisyTVEnvWrapperCIFAR
import gymnasium as gym, imageio
from PIL import Image
import wandb

# NoisyTV Wrappers
class NoisyTVWrapperDiscrete(gym.Wrapper):
    def __init__(self, env, get_random_cifar_fn, trigger_prob=0.05):
        super().__init__(env)
        self.get_random_cifar = get_random_cifar_fn
        self.trigger_prob = trigger_prob
    def step(self, action):
        is_noise = random.random() < self.trigger_prob
        obs, reward, terminated, truncated, info = self.env.step(action)
        if is_noise:
            noise_img = self.get_random_cifar()
            pil_img = Image.fromarray(noise_img).resize((obs.shape[1], obs.shape[0]), Image.NEAREST)
            obs, reward = np.array(pil_img), 0.0
        info["noisy"] = is_noise
        return obs, reward, terminated, truncated, info

class NoisyTVWrapperContinuous(gym.Wrapper):
    def __init__(self, env, get_random_cifar_fn, trigger_threshold=1.5):
        super().__init__(env)
        self.get_random_cifar = get_random_cifar_fn
        self.trigger_threshold = trigger_threshold
    def step(self, action):
        is_noise = np.any(np.abs(action) > self.trigger_threshold)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if is_noise:
            noise_img = self.get_random_cifar()
            pil_img = Image.fromarray(noise_img).resize((obs.shape[1], obs.shape[0]), Image.NEAREST)
            obs, reward = np.array(pil_img), 0.0
        info["noisy"] = is_noise
        return obs, reward, terminated, truncated, info

# Configクラス
class Config:
    def __init__(self, **kwargs):
        self.buffer_size, self.batch_size, self.seq_length, self.imagination_horizon = 100_000, 16, 50, 20
        self.state_dim, self.num_classes, self.rnn_hidden_dim, self.mlp_hidden_dim = 32, 32, 400, 300
        self.model_lr, self.actor_lr, self.critic_lr, self.epsilon, self.weight_decay = 2e-4, 4e-5, 1e-4, 1e-5, 1e-6
        self.gradient_clipping, self.kl_scale, self.kl_balance, self.actor_entropy_scale = 100, 0.1, 0.8, 1e-3
        self.slow_critic_update, self.ext_reward_loss_scale, self.intr_reward_loss_scale, self.discount_loss_scale, self.update_freq = 100, 1.0, 1.0, 1.0, 80
        self.discount, self.lambda_ = 0.995, 0.95
        self.iter, self.seed_iter, self.eval_interval, self.eval_freq, self.eval_episodes = 6000, 1000, 10000, 5, 5
        self.intr_reward_scale = 1.0
        self.intr_value_scale = 0.1
        for k, v in kwargs.items(): setattr(self, k, v)

# ReplayBuffer (内発的報酬保存機能付き)
class ReplayBuffer:
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.intrinsic_rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)
        self.index, self.is_filled = 0, False
    def push(self, obs, act, rew, intr_rew, done):
        self.observations[self.index] = obs
        self.actions[self.index] = act
        self.rewards[self.index] = rew
        self.intrinsic_rewards[self.index] = intr_rew
        self.done[self.index] = done
        self.index = (self.index + 1) % self.capacity
        self.is_filled = self.is_filled or self.index == 0
    def sample(self, batch_size, chunk_length):
        episode_borders, sampled_indexes = np.where(self.done)[0], []
        for _ in range(batch_size):
            while True:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                if not np.logical_and(initial_index <= episode_borders, episode_borders < initial_index + chunk_length).any(): break
            sampled_indexes.extend(range(initial_index, initial_index + chunk_length))
        return (d[sampled_indexes].reshape(batch_size, chunk_length, *d.shape[1:]) for d in [self.observations, self.actions, self.rewards, self.intrinsic_rewards, self.done])
    def __len__(self): return self.capacity if self.is_filled else self.index

# Helper Functions
def preprocess_obs(obs): return (obs.astype(np.float32) / 255.0) - 0.5

def calculate_lambda_target(rewards, discounts, values, lambda_):
    V_lambda = torch.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        V_lambda[t] = rewards[t] + discounts[t] * (values[t] if t == rewards.shape[0]-1 else ((1-lambda_)*values[t+1] + lambda_*V_lambda[t+1]))
    return V_lambda

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

def evaluation(eval_env, policy, step, cfg):
    all_ep_rewards = []
    os.makedirs("eval_view/video", exist_ok=True)
    with torch.no_grad():
        for i in range(cfg.eval_episodes):
            obs, _ = eval_env.reset()
            policy.reset()
            done, truncated, episode_reward, frames, recon_frames = False, False, [], [], []
            while not done and not truncated:
                action, recon_img = policy(obs, eval=True)
                action_to_step = np.argmax(action) if hasattr(eval_env.action_space, 'n') else np.clip(action, eval_env.action_space.low, eval_env.action_space.high)
                obs, reward, done, truncated, info = eval_env.step(action_to_step)
                frames.append(eval_env.render())
                if recon_img is not None: recon_frames.append((recon_img * 255.0).astype(np.uint8))
                episode_reward.append(reward)
            if i == 0 and len(frames) > 0:
                video_path = f"eval_view/video/eval_iter_{step}_ep_{i}.mp4"
                try:
                    frames = [np.array(f, dtype=np.uint8) for f in frames if f is not None]
                    if len(recon_frames) > 0 and len(recon_frames) == len(frames):
                        combined_frames = []
                        for f, r in zip(frames, recon_frames):
                            f_pil, r_pil = Image.fromarray(f), Image.fromarray(r)
                            f_s = f_pil.resize((64, 64))
                            comb = Image.new('RGB', (128, 64))
                            comb.paste(f_s, (0, 0)); comb.paste(r_pil, (64, 0))
                            combined_frames.append(np.array(comb))
                        imageio.mimsave(video_path, combined_frames, fps=10)
                    else:
                       imageio.mimsave(video_path, frames, fps=10)
                except Exception as e:
                    print(f"Failed to save video: {e}")
            all_ep_rewards.append(np.sum(episode_reward))
    print(f"Eval(iter={step}) mean extrinsic reward: {np.mean(all_ep_rewards):.2f}")

def make_env(seed, name, noisy=False, wrapper_type='simple'):
    if "ALE/" in name:
        from gymnasium.wrappers import AtariPreprocessing
        import ale_py
        gym.register_envs(ale_py)
        env = AtariPreprocessing(gym.make(name, render_mode="rgb_array", frameskip=1), screen_size=64, grayscale_obs=False, terminal_on_life_loss=False)
    else:
        env = gym.make(name, render_mode="rgb_array")
    if noisy:
        cifar_fn = create_cifar_function_simple()
        if wrapper_type == 'action_space' and hasattr(env.action_space, 'n'):
            print("Using action_space extension for Noisy-TV")
            env = NoisyTVEnvWrapperCIFAR(env, cifar_fn)
        else:
            print("Using simple noise injection for Noisy-TV")
            if isinstance(env.action_space, gym.spaces.Box): env = NoisyTVWrapperContinuous(env, cifar_fn)
            else: env = NoisyTVWrapperDiscrete(env, cifar_fn)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='ALE/Breakout-v5')
    parser.add_argument('--noisy-tv', action='store_true')
    parser.add_argument('--noisy-wrapper-type', type=str, default='simple', choices=['simple', 'action_space'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', default='Dreamer-Separated-Rewards')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-run-name', default='breakout-separate-v1')
    args = parser.parse_args()
    cfg = Config(iter=args.steps, noisy_wrapper_type=args.noisy_wrapper_type)
    if args.wandb: wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(cfg), mode="online")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env, eval_env = make_env(args.seed, args.env_name, args.noisy_tv, args.noisy_wrapper_type), make_env(args.seed + 100, args.env_name, False)
    atexit.register(env.close); atexit.register(eval_env.close)
    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    replay_buffer = ReplayBuffer(cfg.buffer_size, (64, 64, 3), action_dim)
    
    rssm = RSSM(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes, action_dim).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder(cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    ext_reward_model = RewardModel(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    intr_reward_model = IntrinsicRewardModel(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    actor = DiscreteActor(action_dim, cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device) if hasattr(env.action_space, 'n') else Actor(action_dim, cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    ext_critic = Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    intr_critic = Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    target_ext_critic = Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    target_intr_critic = Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    target_ext_critic.load_state_dict(ext_critic.state_dict())
    target_intr_critic.load_state_dict(intr_critic.state_dict())
    agent = Agent(encoder, decoder, rssm, actor).to(device)
    
    wm_params = list(rssm.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) + list(ext_reward_model.parameters()) + list(intr_reward_model.parameters())
    wm_optimizer = optim.Adam(wm_params, lr=cfg.model_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.actor_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    ext_critic_optimizer = optim.Adam(ext_critic.parameters(), lr=cfg.critic_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    intr_critic_optimizer = optim.Adam(intr_critic.parameters(), lr=cfg.critic_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)

    obs, _ = env.reset()
    for _ in tqdm(range(cfg.seed_iter), desc="Pre-filling buffer"):
        action = env.action_space.sample()
        action_to_push = np.zeros(action_dim)
        action_to_push[action] = 1
        next_obs, reward, done, truncated, _ = env.step(action)
        replay_buffer.push(preprocess_obs(obs), action_to_push, reward, 0.0, done or truncated)
        obs = next_obs if not (done or truncated) else env.reset()[0]

    log_file = "intrinsic_stats_separated.csv"
    with open(log_file, "w") as f: f.write("step,actual_error,intr_reward,is_noisy\n")
    print("Starting Main Loop...")
    total_reward = []
    for iteration in tqdm(range(cfg.iter), desc="Training Steps"):
        with torch.no_grad():
            action, _ = agent(obs, eval=False)
            env_action = np.argmax(action) if hasattr(env.action_space, 'n') else np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, done, truncated, info = env.step(env_action)
            
            intr_reward = 0.0
            if agent.last_state is not None:
                t_action = torch.as_tensor(action, device=device).unsqueeze(0)
                t_next_rnn = rssm.recurrent(agent.last_state, t_action, agent.last_rnn_hidden)
                t_next_prior = rssm.get_prior(t_next_rnn)
                t_next_state = t_next_prior.mean.flatten(1)
                t_next_obs_pred = decoder(t_next_state, t_next_rnn).mean.squeeze().cpu().numpy().transpose(1, 2, 0)
                actual_error = np.mean((preprocess_obs(next_obs) - t_next_obs_pred)**2)
                intr_reward = cfg.intr_reward_scale * np.log(actual_error + 1e-6)
                with open(log_file, "a") as f: f.write(f"{iteration},{actual_error},{intr_reward},{info.get('noisy', False)}\n")
                if args.wandb: wandb.log({"Intrinsic/Actual_Error": actual_error, "Intrinsic/Intrinsic_Reward": intr_reward}, step=iteration)
            
            action_to_push = action
            replay_buffer.push(preprocess_obs(obs), action_to_push, reward, intr_reward, done or truncated)
            obs = next_obs
            total_reward.append(reward)
            if done or truncated:
                if args.wandb: wandb.log({"Episode/Extrinsic_Reward": np.sum(total_reward), "Episode/Length": len(total_reward)}, step=iteration)
                obs, _ = env.reset(); agent.reset(); total_reward = []
        
        if (iteration + 1) % cfg.eval_interval == 0: evaluation(eval_env, agent, iteration, cfg)
        
        if (iteration + 1) % cfg.update_freq == 0 and len(replay_buffer) > cfg.batch_size:
            observations, actions, ext_rewards, intr_rewards, done_flags = replay_buffer.sample(cfg.batch_size, cfg.seq_length)
            observations, actions, ext_rewards, intr_rewards, done_flags = (torch.as_tensor(d, device=device) for d in (observations, actions, ext_rewards, intr_rewards, 1-done_flags))
            observations, actions, ext_rewards, intr_rewards, done_flags = torch.permute(observations, (1,0,4,2,3)), actions.transpose(0,1), ext_rewards.transpose(0,1), intr_rewards.transpose(0,1), done_flags.transpose(0,1).float()

            emb_observations = encoder(observations.reshape(-1, 3, 64, 64)).view(cfg.seq_length, cfg.batch_size, -1)
            state, rnn_hidden = torch.zeros(cfg.batch_size, cfg.state_dim*cfg.num_classes, device=device), torch.zeros(cfg.batch_size, cfg.rnn_hidden_dim, device=device)
            states, rnn_hiddens, kl_loss = torch.zeros(cfg.seq_length, *state.shape, device=device), torch.zeros(cfg.seq_length, *rnn_hidden.shape, device=device), 0
            for i in range(cfg.seq_length-1):
                rnn_hidden = rssm.recurrent(state, actions[i], rnn_hidden)
                next_state_prior, next_detach_prior = rssm.get_prior(rnn_hidden, detach=True)
                next_state_posterior, next_detach_posterior = rssm.get_posterior(rnn_hidden, emb_observations[i+1], detach=True)
                state = next_state_posterior.rsample().flatten(1)
                rnn_hiddens[i+1], states[i+1] = rnn_hidden, state
                kl_loss += (cfg.kl_balance * torch.mean(kl_divergence(next_detach_posterior, next_state_prior))) + ((1 - cfg.kl_balance) * torch.mean(kl_divergence(next_state_posterior, next_detach_prior)))
            kl_loss /= (cfg.seq_length - 1)
            
            flatten_rnn_hiddens, flatten_states = rnn_hiddens[1:].reshape(-1, cfg.rnn_hidden_dim), states[1:].reshape(-1, cfg.state_dim*cfg.num_classes)
            
            obs_dist = decoder(flatten_states, flatten_rnn_hiddens)
            ext_reward_dist = ext_reward_model(flatten_states, flatten_rnn_hiddens)
            intr_reward_dist = intr_reward_model(flatten_states, flatten_rnn_hiddens)

            obs_loss = -torch.mean(obs_dist.log_prob(observations[1:].reshape(-1, 3, 64, 64)))
            ext_reward_loss = -torch.mean(ext_reward_dist.log_prob(ext_rewards[:-1].reshape(-1, 1)))
            intr_reward_loss = -torch.mean(intr_reward_dist.log_prob(intr_rewards[:-1].reshape(-1, 1)))
            wm_loss = obs_loss + cfg.ext_reward_loss_scale * ext_reward_loss + cfg.intr_reward_loss_scale * intr_reward_loss + cfg.kl_scale * kl_loss
            
            wm_optimizer.zero_grad(); wm_loss.backward(); clip_grad_norm_(wm_params, cfg.gradient_clipping); wm_optimizer.step()
            if args.wandb: wandb.log({"Train/WM_Loss": wm_loss.item(), "Train/Obs_Loss": obs_loss.item(), "Train/Ext_Reward_Loss": ext_reward_loss.item(), "Train/Intr_Reward_Loss": intr_reward_loss.item(), "Train/KL_Loss": kl_loss.item()}, step=iteration)
            
            # --- Actor Critic Update ---
            # Refactored to avoid in-place operations
            imagined_states_list = []
            imagined_rnn_hiddens_list = []
            imagined_action_entropys_list = []

            # Initial states for imagination, detached from previous graph
            current_imagined_state = states.flatten(0,1).detach()
            current_imagined_rnn_hidden = rnn_hiddens.flatten(0,1).detach()
            
            # Store initial state (step 0 of imagination)
            imagined_states_list.append(current_imagined_state)
            imagined_rnn_hiddens_list.append(current_imagined_rnn_hidden)
            
            # Imagination Loop
            for i in range(cfg.imagination_horizon): # Loop for H steps (0 to H-1)
                # Actor computes action from the CURRENT state
                i_actions, _, i_action_entropys = actor(current_imagined_state, current_imagined_rnn_hidden)
                imagined_action_entropys_list.append(i_action_entropys) # List has H elements

                # World model predicts the NEXT state
                current_imagined_rnn_hidden = rssm.recurrent(current_imagined_state, i_actions, current_imagined_rnn_hidden)
                current_imagined_state = rssm.get_prior(current_imagined_rnn_hidden).rsample().flatten(1)
                
                # Append the NEXT state to the lists
                imagined_states_list.append(current_imagined_state)
                imagined_rnn_hiddens_list.append(current_imagined_rnn_hidden)
            
            # Stack the lists of tensors into final tensors
            # Resulting tensors will have shape (H+1, B*S_len, ...) for states/hiddens
            # and (H, B*S_len, ...) for entropies
            imagined_states_full = torch.stack(imagined_states_list) # (H+1, B*S_len, ...)
            imagined_rnn_hiddens_full = torch.stack(imagined_rnn_hiddens_list) # (H+1, B*S_len, ...)
            imagined_action_entropys_full = torch.stack(imagined_action_entropys_list) # (H, B*S_len)
            
            # Loss Calculation (using H steps, from step 1 to H of imagination)
            # Need to slice [1:] for states/hiddens to get the H predicted steps
            flatten_imagined_states = imagined_states_full[1:].reshape(-1, cfg.state_dim * cfg.num_classes) # (H * B*S_len, ...)
            flatten_imagined_rnn_hiddens = imagined_rnn_hiddens_full[1:].reshape(-1, cfg.rnn_hidden_dim) # (H * B*S_len, ...)
            
            # Rewards
            imagined_ext_rewards = ext_reward_model(flatten_imagined_states, flatten_imagined_rnn_hiddens).mean.view(cfg.imagination_horizon, -1) # (H, B*S_len)
            imagined_intr_rewards = intr_reward_model(flatten_imagined_states, flatten_imagined_rnn_hiddens).mean.view(cfg.imagination_horizon, -1) # (H, B*S_len)
            discount_arr = torch.full_like(imagined_ext_rewards, cfg.discount, device=device) # (H, B*S_len)
            
            # Target Values
            target_ext_values = target_ext_critic(flatten_imagined_states, flatten_imagined_rnn_hiddens).view(cfg.imagination_horizon, -1).detach() # (H, B*S_len)
            target_intr_values = target_intr_critic(flatten_imagined_states, flatten_imagined_rnn_hiddens).view(cfg.imagination_horizon, -1).detach() # (H, B*S_len)

            lambda_target_ext = calculate_lambda_target(imagined_ext_rewards, discount_arr, target_ext_values, cfg.lambda_) # (H, B*S_len)
            lambda_target_intr = calculate_lambda_target(imagined_intr_rewards, discount_arr, target_intr_values, cfg.lambda_) # (H, B*S_len)
            
            combined_lambda_target = lambda_target_ext + cfg.intr_value_scale * lambda_target_intr # (H, B*S_len)
            
            # Weights for Actor/Critic losses (H steps)
            # Need weights for H steps of rewards/entropies
            weights = torch.cumprod(torch.cat([torch.ones(1, imagined_ext_rewards.shape[1], device=device), discount_arr], dim=0), dim=0).detach() # (H+1, B*S_len)
            
            # Actor Loss
            actor_loss = -(weights[:-1] * (combined_lambda_target + cfg.actor_entropy_scale * imagined_action_entropys_full.unsqueeze(1))).mean()
            actor_optimizer.zero_grad(); actor_loss.backward(retain_graph=True); clip_grad_norm_(actor.parameters(), cfg.gradient_clipping); actor_optimizer.step()
            
            # Critic Losses
            # Note: Critic loss calculation uses H steps of values/lambda_target
            # And H+1 steps for weights. weights[0] is 1.0.
            flatten_imagined_states_for_critic = imagined_states_full.detach().reshape(-1, cfg.state_dim * cfg.num_classes)
            flatten_imagined_rnn_hiddens_for_critic = imagined_rnn_hiddens_full.detach().reshape(-1, cfg.rnn_hidden_dim)

            value_pred_ext_full = ext_critic(flatten_imagined_states_for_critic, flatten_imagined_rnn_hiddens_for_critic).view(cfg.imagination_horizon + 1, -1) # (H+1, B*S_len)
            ext_critic_loss = -(weights[:-1] * MSE(value_pred_ext_full[:-1]).log_prob(lambda_target_ext.detach())).mean() # Compare H steps
            ext_critic_optimizer.zero_grad(); ext_critic_loss.backward(retain_graph=True); clip_grad_norm_(ext_critic.parameters(), cfg.gradient_clipping); ext_critic_optimizer.step()

            value_pred_intr_full = intr_critic(flatten_imagined_states_for_critic, flatten_imagined_rnn_hiddens_for_critic).view(cfg.imagination_horizon + 1, -1) # (H+1, B*S_len)
            intr_critic_loss = -(weights[:-1] * MSE(value_pred_intr_full[:-1]).log_prob(lambda_target_intr.detach())).mean() # Compare H steps
            intr_critic_optimizer.zero_grad(); intr_critic_loss.backward(); clip_grad_norm_(intr_critic.parameters(), cfg.gradient_clipping); intr_critic_optimizer.step()
            
            if (iteration + 1) % cfg.slow_critic_update == 0:
                target_ext_critic.load_state_dict(ext_critic.state_dict())
                target_intr_critic.load_state_dict(intr_critic.state_dict())

    torch.save(agent.state_dict(), "agent_separate_rewards.pth")

if __name__ == "__main__":
    main()