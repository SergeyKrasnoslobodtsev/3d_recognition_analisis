import json
import matplotlib.pyplot as plt

def plot_history(history_json_path: str, out_png: str = "training_curves.png"):
    with open(history_json_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train_loss")
    ax[0].plot(hist["val_loss"],   label="val_loss")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()

    ax[1].plot(hist["train_acc"], label="train_acc")
    ax[1].plot(hist["val_acc"],   label="val_acc")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()

    fig.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)