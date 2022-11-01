import taichi as ti 
ti.init(arch=ti.cuda, default_fp=ti.f32)
import numpy as np
from sim import mpm
from sim.loadRobot import Robot

robotFile = "DillerBot/DillerBot.yaml"
print("Generating Particles....")
r = Robot(robotFile, ppm3=1e14, scale=1e-3)

# Simulation Parameters
grid_size = 25e-3 # Width and Height of window in metres
dx = 0.1e-3 # Size of grid cell in meters
g = np.array([0, 0, -9.81])
gamma = 2e1 # Damping Constant
offset = np.array([grid_size/4, grid_size/4, 0.6e-3]) #Make sure robot is within grid
dt = 7e-7
# Initialise Variables
print("Initialising Variables...")
mpm.init(r, scale=1, grid_size_=grid_size, dx_=dx, g_=g, gamma_=gamma, offset=offset)
# Visualisation
window = ti.ui.Window("Window", (1024, 1024))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0, 10, 10)
camera.lookat(5, 0, 5)

field = np.array([0, 0 ,0])
t = 0
p_size = 0.01
rotate = False
angluar_velocity = 1000
theta = 0
magnitude = 0

timesteps = 0
renderEvery = 10

gui = window.get_gui()
while window.running:
    t += dt
    timesteps += 1
    mpm.advance(dt)
    if timesteps % renderEvery == 0:
        camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        mpm.getParticles(1e3)
        scene.particles(mpm.x_visualise,
                        radius=p_size,
                        color=(0.5, 0.42, 0.8),
                        per_vertex_color=mpm.colors)
        canvas.scene(scene)
        if rotate:
            theta -= angluar_velocity*dt
        rot_m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        field_vector = rot_m@np.array([[0], [magnitude]])
        field[0] = 0.0
        field[1] = field_vector[0]
        field[2] = field_vector[1]
        mpm.setField(field * 1e-3)
        with gui.sub_window("Field", 0.05, 0.8, 0.2, 0.1) as w:
            magnitude = w.slider_float("Magnitude (mT)", magnitude, -20, 20)
            w.text(str(field))
        with gui.sub_window("Particle Size", 0.05, 0.9, 0.2, 0.05) as w:
            p_size = w.slider_float("Size", p_size, 0.01, 0.2)
        with gui.sub_window("Rotate Field", 0.05, 0.1, 0.25, 0.2) as w:
            clicked =  w.button("Enable/Disable")
            angluar_velocity = w.slider_float("omega", angluar_velocity, 0, 100)
            mpm.gamma =  w.slider_float("gamma", mpm.gamma, 1e5, 100e5)
            if clicked and rotate:
                rotate = False
            elif clicked:
                rotate = True
        window.show()
