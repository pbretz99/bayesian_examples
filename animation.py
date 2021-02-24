from matplotlib import pyplot as plt
from celluloid import Camera
import numpy as np

def animate_heat_eqn(state_grid, obs_grid, theta, Y, title, filename, c_scale):
    length = Y.shape[0]
    size = int(np.sqrt(state_grid[0].size))
    x, y = state_grid[0].reshape((size, size)), state_grid[1].reshape((size, size))
    #th_min, th_max = np.min(theta), np.max(theta)
    # create figure object
    fig = plt.figure()
    # load axis box
    ax = plt.axes()
    camera = Camera(fig)
    for i in range(length):
        th = theta[i,::].reshape((size, size))
        ax.contourf(x, y, th, 20, cmap='cool', vmin=c_scale[0], vmax=c_scale[1])
        ax.scatter(state_grid[0], state_grid[1], facecolors='none', edgecolors='black', alpha=0.25)
        ax.scatter(obs_grid[0], obs_grid[1], c=Y[i,::], edgecolors='black', s=250, cmap='cool', vmin=c_scale[0], vmax=c_scale[1])
        ax.title.set_text(title)
        plt.pause(0.1)
        camera.snap()
    animation = camera.animate()
    animation.save(filename, writer='Pillow', fps=2)

def animate_heat_eqn_wrapper(state_grid, obs_grid, theta, fit_values, Y):
    error = np.abs(fit_values-theta)
    th_scale =[np.min(theta), np.max(theta)]
    err_scale = [0, np.max(error)]
    animate_heat_eqn(state_grid, obs_grid, theta, Y, 'Temperature Simulation', 'simulation.gif', th_scale)
    animate_heat_eqn(state_grid, obs_grid, fit_values, Y, 'Temperature Fit', 'fit.gif', th_scale)
    animate_heat_eqn(state_grid, obs_grid, error, np.zeros(Y.shape), 'Fit Error', 'error.gif', err_scale)