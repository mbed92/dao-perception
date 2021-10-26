import matplotlib.animation as animation
import matplotlib.pyplot as plt


def visualize_rod_3d(rod_positions: list, output_filename: str = 'sacatter.gif'):
    assert len(rod_positions) > 0

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)

    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_zlim([0, 2])

    def update_scatter(i):
        ax.clear()
        rod_pose = rod_positions[i]
        ax.scatter(rod_pose[0], rod_pose[2], rod_pose[1])

    ani = animation.FuncAnimation(fig, update_scatter, frames=len(rod_positions), interval=100)
    ani.save(output_filename, writer='pillow')
