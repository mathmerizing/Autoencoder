import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

if __name__ == "__main__":
    file_name = "train_loss.txt" if len(sys.argv) < 2 else sys.argv[1]

    fig = plt.figure()
    fig.canvas.set_window_title(file_name)
    ax1 = fig.add_subplot(1,1,1)

    def animate(i):
        lines = open(file_name, "r").readlines()
        xs, ys = [], []
        for line in lines[1:]:
            x, y = line.split(",")
            xs.append(int(x))
            ys.append(float(y))
        ax1.clear()
        ax1.plot(xs,ys)

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()
