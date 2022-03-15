import matplotlib.pyplot as plt
import pandas as pd
import sys

def main():
    if not len(sys.argv) > 1:
        print("Pass loss file as argument")
        return
    data = pd.read_csv(sys.argv[1], header=None)

    # extract the data
    epoch = data[0]
    training_loss = data[1]
    validation_loss = data[2]
    training_accuracy = data[3]
    validation_accuracy = data[4]

    # plot the data
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].plot(epoch, training_loss, label="training_loss")
    axs[0].set_title("training vs validation loss")
    axs[0].plot(epoch, validation_loss, label="validation_loss")
    axs[1].plot(epoch, training_accuracy, label="training_acc")
    axs[1].plot(epoch, validation_accuracy, label="validation_acc")
    axs[1].set_title("training vs validation accuracy")

    plt.legend()
    plt.savefig("loss.jpg")
    

if __name__ == "__main__":
    main()
