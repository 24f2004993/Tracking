import pandas as pd
import matplotlib.pyplot as plt

# Use full path
csv_path = '/Users/tanishq/Desktop/particleeeee1/trajectories.csv'
df = pd.read_csv(csv_path)

# Example: plot trajectories
plt.figure(figsize=(10,8))
for tid, group in df.groupby("ID"):
    plt.plot(group["X"], group["Y"], marker="o", label=f"ID {tid}")

plt.gca().invert_yaxis()  # because image origin is top-left
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Particle Trajectories")
plt.legend()
plt.show()