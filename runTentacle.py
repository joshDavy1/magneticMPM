from matplotlib.pyplot import plot
import taichi as ti 
import numpy as np
from sim.loadRobot import Robot
ti.init(arch=ti.cuda, default_fp=ti.f32)
from sim.mpm_class import magneticMPM
from colour_palette import tentacle_palette

robotFile = "Tentacle/tentacleF.yaml"
print("Generating Particles....")
r = Robot(robotFile, ppm3=5e10, scale=1e-3)

grid_size = 200e-3 # Width and Height of window in metres
dx = 1e-3 # Size of grid cell in meters
g = np.array([0, 0, -9.81])
gamma = 200 # Damping Constant
offset = np.array([grid_size/2, grid_size/2, 30e-3])

print("Initialising Variables.... (This may take a while)")
mpm = magneticMPM(r, scale=1, grid_size=grid_size, dx=dx, g=g, gamma=gamma, offset=offset, colour_palette=tentacle_palette)

window = ti.ui.Window("Window", (1024, 1024))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(200, 200, 200)
camera.lookat(100, 0, 100)

field = np.array([0, 0 ,0])

t = 0
dt = 2e-5
p_size = 0.5
vertices = ti.Vector.field(3, dtype=float, shape=(2))

gui = window.get_gui()
while window.running:
    t += dt
    mpm.advance(dt)
    camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    mpm.getParticles(1e3)
    scene.particles(mpm.x_visualise,
                    radius=p_size,
                    per_vertex_color=mpm.colors)
    #########################
    canvas.scene(scene)
    with gui.sub_window("Field", 0.05, 0.8, 0.2, 0.1) as w:
        field[0] = w.slider_float("x (mT)", field[0], -50, 50)
        field[1] = w.slider_float("y (mT)", field[1], -50, 50)
        field[2] = w.slider_float("z (mT)", field[2], -50, 50)
        mpm.setField(field * 1e-3)
    with gui.sub_window("Particle Size", 0.05, 0.9, 0.2, 0.1) as w:
        p_size = w.slider_float("Size", p_size, 0.01, 1)
        mpm.gamma = w.slider_float("Gamma", mpm.gamma, 0, 2e3)
    window.show()
