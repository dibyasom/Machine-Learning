import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SimplePendulum:

    def __init__(self,
                 init_angle=50,
                 L=1.0,  # length of pendulum in m
                 G=9.8,  # acceleration due to gravity, in m/s^2
                 origin=(0,0)): 
        self.L = L
        self.G = G
        self.origin = origin
        self.time_elapsed = 0
        self.init_angle = init_angle * np.pi / 180.
        self.angle = self.init_angle
    
    def position(self):
        """compute the current x,y position of the pendulum"""
        x = np.cumsum([self.origin[0], self.L * np.sin(self.angle)])
        y = np.cumsum([self.origin[1], -self.L * np.cos(self.angle)])
        return (x, y)

    def step(self, dt):
        """execute one time step of length dt and update state"""
        self.time_elapsed += dt
        self.angle = self.init_angle * np.cos(np.sqrt( self.G / self.L) * self.time_elapsed)

#------------------------------------------------------------
# set up initial state and global variables
pendulum = SimplePendulum(40)
dt = 1./30 # 30 fps

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    """perform animation step"""
    global pendulum, dt
    pendulum.step(dt)
    
    line.set_data(*pendulum.position())
    time_text.set_text('time = %.1f' % pendulum.time_elapsed)
    return line, time_text

# choose interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig,
                              animate,
                              frames=300,
                              interval=interval,
                              blit=False,
                              init_func=init)

plt.show()