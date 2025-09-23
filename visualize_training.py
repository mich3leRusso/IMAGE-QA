from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

# Path to your event file or its folder
logdir = "/davinci-1/home/micherusso/PycharmProjects/IMAGE-QA/runs/SWIN_train_KADID10K_0"

# Load the event file
ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()

# Print available scalar tags
print("Available metrics:", ea.Tags()['scalars'])

# Create output folder for images
os.makedirs("training_plots/Loss/", exist_ok=True)

# Plot each scalar
for tag in ea.Tags()['scalars']:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    plt.figure()
    plt.plot(steps, values, marker='o')
    plt.title(tag)
    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"training_plots/{tag}.png")
    plt.close()

print("All plots saved in ./training_plots/")
