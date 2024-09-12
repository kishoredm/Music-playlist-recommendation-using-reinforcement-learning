import numpy as np
import pandas as pd
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Constants
NUM_FEATURES = 17  # Number of song features
BATCH_SIZE = 32
MEMORY_SIZE = 10000
GAMMA = 0.95  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Exploration rate decay
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

file_path="C:\\Users\\kisho\\OneDrive\\Documents\\RL\\project\\dataset.csv"
# Load the song data from CSV
def load_songs(file_path):
    df = pd.read_csv(file_path)
    song_titles = df[['track_id', 'track_name']].values
    song_features = df[['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                        'time_signature']].values
    return song_titles, song_features

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, num_features):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_features * 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Action Head Deep Q-Network
class AH_DQN:
    def __init__(self, num_songs, num_features):
        self.num_songs = num_songs
        self.num_features = num_features
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = DQN(num_features).to(DEVICE)
        self.target_model = DQN(num_features).to(DEVICE)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.song_features = None

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_virtual_user):
        if is_virtual_user or np.random.rand() <= EPSILON:
            return random.randrange(self.num_songs)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        song_features = torch.tensor(self.song_features, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            act_values = self.model(state.repeat(self.num_songs, 1), song_features).squeeze().cpu().numpy()
        return np.argmax(act_values)

    def replay(self, batch_size):
        global EPSILON
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        q_values = []
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)
            song_features = torch.tensor(self.song_features, dtype=torch.float32).to(DEVICE)
            target = torch.tensor(reward, dtype=torch.float32).to(DEVICE)
            if not done:
                with torch.no_grad():
                    target += GAMMA * self.target_model(next_state.repeat(self.num_songs, 1), song_features).max()
            target_f = self.model(state.repeat(self.num_songs, 1), song_features)
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state.repeat(self.num_songs, 1), song_features), target_f)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            q_values.append(target.item())
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY
        return np.mean(losses), np.mean(q_values)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

# World Model (Virtual Users)
class WorldModel:
    def __init__(self, song_features, num_features):
        self.num_songs = song_features.shape[0]
        self.num_features = num_features
        self.song_features = song_features
        self.user_preferences = np.random.rand(self.num_features)

    def get_reward(self, song_features):
        return np.dot(self.user_preferences, song_features)

# Actual User
class ActualUser:
    def __init__(self, root, num_features):
        self.root = root
        self.num_features = num_features
        self.user_preferences = None
        self.pref_window = None

    def set_preferences(self):
        self.pref_window = tk.Toplevel(self.root)
        self.pref_window.title("Music Preferences")
        
        label = tk.Label(self.pref_window, text="Select song features that you like:", font=("Helvetica", 12))
        label.pack()

        self.check_vars = []
        features = ["popularity", "duration_ms", "explicit", "danceability", "energy", "key", "loudness", "mode", 
                    "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", 
                    "time_signature"]
        for feature in features:
            var = tk.IntVar()
            self.check_vars.append(var)
            check_button = tk.Checkbutton(self.pref_window, text=feature, variable=var, font=("Helvetica", 10))
            check_button.pack(anchor=tk.W)

        submit_button = tk.Button(self.pref_window, text="Submit", command=self.submit_preferences, font=("Helvetica", 12))
        submit_button.pack(pady=10)

    def submit_preferences(self):
        self.user_preferences = [var.get() for var in self.check_vars]
        if sum(self.user_preferences) == 0:
            messagebox.showwarning("Warning", "Please select at least one feature.")
        else:
            self.pref_window.destroy()
            self.display_playlist()

    def display_playlist(self):
        playlist = generate_playlist(agent, self)

        result_window = tk.Toplevel(self.root)
        result_window.title("Playlist for Actual User")

        result_text = tk.Text(result_window, width=40, height=10)
        result_text.pack(pady=10, padx=10)

        for song in playlist:
            result_text.insert(tk.END, f"Song {song_titles[song][1]}\n")  # Display track_name instead of track_id

# Training
def train(num_episodes, song_features):
    world_model = WorldModel(song_features, NUM_FEATURES)
    agent = AH_DQN(song_features.shape[0], NUM_FEATURES)
    agent.song_features = world_model.song_features

    episode_rewards = []
    episode_losses = []
    episode_q_values = []

    for episode in range(num_episodes):
        state = world_model.user_preferences
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state, is_virtual_user=True)
            song_features = world_model.song_features[action]
            reward = world_model.get_reward(song_features)
            next_state = state
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            done = True
            if done:
                agent.update_target_model()
                if len(agent.memory) > BATCH_SIZE:
                    loss, avg_q_value = agent.replay(BATCH_SIZE)
                    episode_losses.append(loss)
                    episode_q_values.append(avg_q_value)
                else:
                    episode_losses.append(0)
                    episode_q_values.append(0)
                episode_rewards.append(total_reward)
                print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward}, Loss: {episode_losses[-1]}, Q-value: {episode_q_values[-1]}")
                break

    agent.save("ah_dqn_weights.pth")
    print("Training completed and weights saved.")

    # Plotting
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_episodes + 1), episode_rewards, label='Episode vs Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Episode vs Total Reward')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_episodes + 1), episode_losses, label='Episode vs Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Episode vs Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_episodes + 1), episode_q_values, label='Episode vs Q-value')
    plt.xlabel('Episodes')
    plt.ylabel('Average Q-value')
    plt.title('Episode vs Q-value')
    plt.legend()

    plt.show()

# Generate Playlist
def generate_playlist(agent, user):
    if user.user_preferences is None:
        user.set_preferences()
        return []

    user_preferences = np.array(user.user_preferences).astype(float)
    state = user_preferences
    playlist = []

    for _ in range(10):
        action = agent.act(state, is_virtual_user=False)
        playlist.append(action)

    return playlist

# GUI
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return
    global song_titles, song_features
    song_titles, song_features = load_songs(file_path)
    messagebox.showinfo("Success", "CSV file loaded successfully!")

def start_training():
    try:
        num_episodes = int(num_episodes_entry.get())
        train(num_episodes, song_features)
        messagebox.showinfo("Success", "Training completed successfully!")
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid number of episodes.")

def show_recommendations():
    global agent
    agent = AH_DQN(song_features.shape[0], NUM_FEATURES)
    agent.load("ah_dqn_weights.pth")
    actual_user = ActualUser(root, NUM_FEATURES)
    actual_user.set_preferences()

root = tk.Tk()
root.title("Music Recommendation System")
root.geometry("400x300")

main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill="both", expand=True)

title_label = ttk.Label(main_frame, text="Music Recommendation System", font=("Helvetica", 16))
title_label.pack(pady=10)

load_button = ttk.Button(main_frame, text="Load Songs CSV", command=open_file)
load_button.pack(pady=10)

num_episodes_label = ttk.Label(main_frame, text="Number of Episodes:", font=("Helvetica", 12))
num_episodes_label.pack(pady=5)
num_episodes_entry = ttk.Entry(main_frame)
num_episodes_entry.pack(pady=5)

train_button = ttk.Button(main_frame, text="Train Model", command=start_training)
train_button.pack(pady=10)

recommend_button = ttk.Button(main_frame, text="Show Recommendations", command=show_recommendations)
recommend_button.pack(pady=10)

root.mainloop()
