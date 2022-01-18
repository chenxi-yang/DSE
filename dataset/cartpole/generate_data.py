import pickle
from imitation.data import rollout

with open("final.pkl", "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

f = open('../cartpole-v1_0.418.txt', 'w')
f.write("trajectory_list\n")

for trajectory in trajectories:
    # print(len(trajectory.obs), len(trajectory.acts))
    for idx, obs in enumerate(trajectory.obs[:-1]):
        action = [trajectory.acts[idx]]
        state = (list(obs), action)
        f.write(f"{state};")
    f.write(f"\n")
f.close()
        
        

