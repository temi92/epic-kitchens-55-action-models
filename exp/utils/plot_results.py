import matplotlib.pyplot as plt
import numpy as np

def create_bar_plot(models_noun_acc, models_verb_acc, models_noun_loss, 
                        models_verb_loss, models_fps, labels=["tsn", "trn"]):
    ind = np.arange(len(labels))
    width = 0.35
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.set_title("Accuracy@5 comparison between models")
    ax1.set_ylabel("accuracy")
    ax1.bar(ind-width/2, models_noun_acc, width, label="noun")
    ax1.bar(ind+width/2, models_verb_acc, width, label="verb")

    ax1.set_xticks(ind)
    ax1.set_xticklabels(labels)
    ax1.legend()

    ax2.set_title("Loss comparison between models")
    ax2.set_ylabel("loss")
    ax2.bar(ind-width/2, models_noun_loss, width, label="noun")
    ax2.bar(ind+width/2, models_verb_loss, width, label="verb")

    ax2.set_xticks(ind)
    ax2.set_xticklabels(labels)
    ax2.legend()

    ax3.set_title("fps comparison between models")
    ax3.set_ylabel("frame per sec")
    ax3.bar(labels, models_fps, width)

    fig.tight_layout()
    plt.show()


def create_acc_loss_curve(results): 
    plt.style.use("ggplot")

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ind = np.arange(0, len(results["train_loss"]))
    ax1.plot(ind, results["train_loss"], label="train_loss")
    ax1.plot(ind, results["val_loss"], label="val_loss")

    ax1.set_xticks(ind)


    ax1.set_title("Loss comparison between train and val sets")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    ax2.plot(ind, results["train_acc"], label="train_acc")
    ax2.plot(ind, results["val_acc"], label="val_acc")
    ax2.set_title("Accuracy comparison between train and val sets")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Acc")
    ax2.legend()
    plt.show()
