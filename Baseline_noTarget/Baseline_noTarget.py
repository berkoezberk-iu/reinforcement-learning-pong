# train_dqn.py
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing, TransformReward
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cv2
import os
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import ale_py

# === Hyperparameters ===
GAMMA = 0.99
LR = 0.00025
BATCH_SIZE = 32
REPLAY_SIZE = 100_000
START_TRAINING = 10_000
#TARGET_UPDATE_FREQ = 200

EPS_START = 1.0
EPS_END = 0.01

EPS_DECAY = 30_000
STACK_SIZE = 4
MODEL_SAVE_PATH = "RESULTS"
MODEL_PATH = MODEL_SAVE_PATH + "/" + "dqn_pong.pt"
BEST_MODEL_PATH = MODEL_SAVE_PATH + "/" + "dqn_pong_best.pt"
REWARDS_PLOT = MODEL_SAVE_PATH + "/" + "rewards_plot.png"
MAX_EPISODES = 800

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# === DQN Network ===
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()

        self.num_actions = num_actions
        self.conv = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = x / 255.0  # Scale to [0, 1]
        x = (x - 0.5) / 0.5  # Normalize to [-1, 1]
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)        

        return self.fc(x)

    # === Epsilon-greedy policy ===
    def epsilon_greedy(self, state, epsilon):
        if random.random() < epsilon:
            action = random.randrange(self.num_actions)
        else:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(DEVICE)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        return action

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)

# === Live Plot Rewards and Loss ===
def live_plot(stp_done,rewards, losses, qvals):
    plt.clf()
    plt.suptitle("Training Progress")    
    plt.subplot(2, 1, 1)
    plt.title(f'Total frames {stp_done}. Avg reward over last 10 episodes: {np.mean(rewards[-10:])}')    
    plt.plot(rewards, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))     

    plt.subplot(2, 1, 2)
    plt.title("Losses")    
    
    plt.plot(qvals, label="Loss", color="orange")  # Use the index as x-axis
    plt.xlabel("Episode")
    plt.ylabel("Loss variation over Episodes")
    plt.grid(True)

    plt.tight_layout()
    plt.pause(0.01)

def compute_loss(model, replay_buffer, batch_size, gamma):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(np.float32(state)).to(DEVICE)
    next_state = torch.FloatTensor(np.float32(next_state)).to(DEVICE)
    action = torch.LongTensor(action).to(DEVICE)
    reward = torch.FloatTensor(reward).to(DEVICE)
    done = torch.FloatTensor(done).to(DEVICE)

    q_values_old = model(state)
    next_q_values = model(next_state).detach()

    q_value_old = q_values_old.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value_old - expected_q_value.detach()).pow(2).mean()

    return loss

# === Training ===
def train(env, model, optimizer, replay_buffer):

    writer = SummaryWriter()

    total_steps = 0
    episode = 0
    rewards = []
    losses = []
    best_reward = float('-inf')
    model.train()
    plt.ion()
    plt.figure(figsize=(10, 8))
    q_values_all =[]
    while episode < MAX_EPISODES:
        state = env.reset()[0]
        q_values_this_episode = [] 
        episode_reward = 0
        done = False
        loss_list = []

        while not done:

            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(- total_steps / EPS_DECAY)
            action = model.epsilon_greedy(state, epsilon)

            # Log Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(DEVICE)
                q_values = model(state_tensor)
                mean_q = q_values.mean().item()
                q_values_this_episode.append(mean_q)

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done) 

            state = next_state
            episode_reward += reward

            loss = 0
            if len(replay_buffer) >= START_TRAINING:

                loss = compute_loss(model, replay_buffer, BATCH_SIZE, GAMMA)
                optimizer.zero_grad()
                loss.backward()

                # Sanity Check: Print gradient norms once
                if total_steps == START_TRAINING:
                #if 1:                    
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    print(f"[Sanity Check] Gradient Norm at step {total_steps}: {total_norm:.4f}")

                optimizer.step()




            total_steps += 1

            if done:
                if len(replay_buffer) >= START_TRAINING:

                   losses.append(loss.item())

                rewards.append(episode_reward)

                
                
        avg_q = np.mean(q_values_this_episode)
        q_values_all.append(avg_q)
        writer.add_scalar("Q_value/Average", avg_q, episode)
        print(f"Episode {episode}, Avg Q: {avg_q:.4f}")

        if (episode+1) % 100 == 0:
            path = os.path.join(MODEL_SAVE_PATH, f"Pong_v5_episode_{episode+1}.pth")
            print(f"Saving weights at Episode {episode+1} ...")
            torch.save(model.state_dict(), path)   


        # Save best model
        if episode_reward >= best_reward:
            best_reward = episode_reward
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        # TensorBoard logging
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Loss", loss, episode)
        writer.add_scalar("Epsilon", epsilon, episode)

        live_plot(total_steps, rewards, losses,q_values_all)
        
        print(f"Episode {episode}, Reward: {episode_reward}, Loss: {loss:.4f}, Epsilon: {epsilon:.3f}")
        episode += 1

    print("Training complete. Final model saved.")
    print(f"Best model with reward {best_reward:.2f} saved to '{BEST_MODEL_PATH}'.")

    path = os.path.join(MODEL_SAVE_PATH, f"Pong_v5_episode_{episode+1}.pth")
    print(f"Saving weights at Episode {episode+1} ...")
    torch.save(model.state_dict(), path)   

    # Save final plot
    plt.ioff()
    plt.savefig(REWARDS_PLOT)
    print(f"Final reward/loss plot saved to '{REWARDS_PLOT}'")

    env.close()

def play_pong_live(env, model, model_path=BEST_MODEL_PATH, episodes=5, render=True, fps=30):
    import time

    # Load the trained model
    num_actions = env.action_space.n
    #model = DQN(num_actions).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print(f"[Live] Loaded model from {model_path}. Running {episodes} episode(s)...")

    for ep in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                q_values = model(state_tensor)
                action = q_values.argmax(1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if render:
                time.sleep(1 / fps)

        print(f"[Episode {ep + 1}] Total Reward: {total_reward:.2f}")

    env.close()
    print("[Live] All episodes complete.")

if __name__ == "__main__":
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)    
    
    live = False

    if (live):
        ren_mode = "human"
    else:
        ren_mode = None
    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode=ren_mode)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False)
    env = TransformReward(env, lambda x: np.clip(x, -1, 1))
    
    env = FrameStackObservation(env, 4)

    num_actions = env.action_space.n
    model = DQN(num_actions).to(DEVICE)
    
    optimizer = optim.RMSprop(model.parameters(), lr=LR, alpha=0.95, eps=0.01)
    
    replay = ReplayBuffer(REPLAY_SIZE)
    
    if (live):
        play_pong_live(env, model)
    else:
        train(env, model, optimizer, replay)
    