import taichi as ti 
ti.init(arch=ti.cuda, default_fp=ti.f32)
import numpy as np
from sim.mpm_class import magneticMPM
from sim.loadRobot import Robot
from colour_palette import generic_palette

robotFile = "SmallScaleBot/SmallScaleBot.yaml"
print("Generating Particles....")
r = Robot(robotFile, ppm3=1e13, scale=1e-3)

# Simulation Parameters
grid_size = 25e-3 # Width and Height of window in metres
dx = 0.1e-3 # Size of grid cell in meters
g = np.array([0, 0, -9.81])
gamma = 180 # Damping Constant
offset = np.array([3e-3, 3e-3, 0.2e-3]) #Make sure robot is within grid
dt = 7e-7
# Initialise Variables
print("Initialising Variables... (this may take a while)")
mpm = magneticMPM(r, scale=1, grid_size=grid_size, dx=dx, g=g, gamma=gamma, offset=offset, colour_palette=generic_palette)
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
magnitude = 0
theta = np.pi
angle = 0
rotate = False
frq = 1

timesteps = 0
renderEvery = 20

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
            alpha = (t % (1/frq))*frq*np.deg2rad(45)
            magnitude = (t % (1/frq)*frq)*8
            theta = np.pi + alpha
        rot_m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        field_vector = rot_m@np.array([[magnitude], [0]])
        field[0] = 0.0
        field[1] = field_vector[0]
        field[2] = field_vector[1]
        mpm.setField(field * 1e-3)
        with gui.sub_window("Field", 0.05, 0.7, 0.3, 0.1) as w:
            w.text(str(field))
            w.text(str(np.rad2deg(theta)))
        with gui.sub_window("Particle Size", 0.05, 0.9, 0.25, 0.055) as w:
            p_size = w.slider_float("Size", p_size, 0.01, 0.2)
        with gui.sub_window("Rotate Field", 0.05, 0.1, 0.25, 0.2) as w:
            clicked =  w.button("Enable/Disable")
            frq = w.slider_float("freq", frq, 0.01, 50)
            if clicked and rotate:
                rotate = False
            elif clicked:
                rotate = True
        window.show()
