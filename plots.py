import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('DuelingDQN_training.csv')
mc = pd.read_csv("manual_control_results.csv")
hp = pd.read_csv("baseline_results.csv")
duel_dqn = pd.read_csv("dueling_dqn_results.csv")
double_dqn = pd.read_csv("double_dqn_results.csv")
dqn = pd.read_csv("dqn_results.csv")

plt.plot(dqn["Episode"], dqn["Return"], label="Vanilla DQN", marker="o")
plt.plot(double_dqn["Episode"], double_dqn["Return"], label="Double DQN", marker="o")
plt.plot(duel_dqn["Episode"], duel_dqn["Return"], label="Dueling DQN", marker="o")
plt.plot(mc["Episode"], mc["Return"], label="Manual Control", marker="o")
plt.plot(hp["Episode"], hp["Return"], label="Heuristic Baseline", marker="o")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.show()


df["Smoothed_Return"] = df["Return"].rolling(window=100).mean()
plt.figure(figsize=(10, 5))
plt.plot(df["Episode"], df["Smoothed_Return"], color='red', label="Smoothed Return (Moving Avg)")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Smoothed Return Across 100 Episodes")
plt.legend()
plt.grid(True)
plt.show()