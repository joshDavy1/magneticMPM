import numpy as np
import taichi as ti

# Simulation Parameters
window_size = 50e-3 # Width and Height of window in metres
dx = 0.3e-3 # Size of grid cell in meters
ppm = 26e3 # particles per meter
inv_mu0 = 1/(4*np.pi*1e-7) #1/(H/m)

# Static
n_grid = int(window_size / dx)
print("n_grid", n_grid)
inv_dx = 1/dx

#Material Parameters 
flux_density = ti.Vector([0.143, 0.0])
G = 303e3
K = 500*G
density = 5e3 * 1 # kgm-3

dt = 1e-6
g = 9.81

# Offset of tentacle
offset = 20e-3

width = 0.84e-3 
length = 17.2e-3

# Particle  volume
p_vol = 1

# MIT Deflection  Branch
ti.init(arch=ti.gpu, default_fp=ti.f64) 

n_particles_y = int(length*ppm)
n_particles_x = int(width*ppm)
n_particles = n_particles_x * n_particles_y
print("n_particles", n_particles)

c = 2e2
# Particle Properties
p_mass = density*p_vol 

#Variable Setup
x = ti.Vector.field(2, dtype=float, shape=(n_particles))     # position
v = ti.Vector.field(2, dtype=float, shape=(n_particles))     # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=(n_particles))  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=(n_particles))  # deformation gradient

grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))      # grid node momentum
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))                   # grid node mass
grid_pin = ti.field(dtype=int, shape=(n_grid, n_grid))

applied_field = ti.Vector.field(2, dtype=float, shape=())
avg = ti.Vector.field(2, dtype=float, shape=())
tip_particles = ti.field(dtype=int, shape=(n_particles)) 

# Initalise beam given properties
n_tip_segments = 0
def reset():
    """ Draws the beam"""
    global n_tip_segments
    n_tip_segments = 0
    n = 0
    n_particles_x = int(length*ppm)
    n_particles_y = int(width*ppm)
    # Create tentacle
    for i in range(n_particles_x):
        for j in range(n_particles_y):
            x_ = offset +  i/ppm + (0.5/ppm)
            y_ =  window_size/2 + j/ppm - width/2 + (0.5/ppm)
            x[n] = [x_, y_]
            if i == n_particles_x-1:
                tip_particles[n] = 1
                n_tip_segments += 1
            else:
                tip_particles[n] = 0
            n += 1
            if i < 0.01*n_particles_x:
                x_pin  = int(x_*inv_dx)
                y_pin = int(y_*inv_dx)
                grid_pin[x_pin, y_pin] = 1
    # Other variable initialisation
    for i in range(n_particles):
        v[i] = [0.0, 0.0]
        F[i] = [[1.0, 0.0], [0.0, 1.0]]
        C[i] = [[0.0, 0.0],[0.0, 0.0]]

@ti.kernel
def clear_grid():
    # Clear Grid
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0

@ti.kernel
def p2g():
    ######## PARTICLE TO GRID ############
    for p in range(n_particles): 
        # Calculate Stress
        f = F[p]
        J = f.determinant()
        I1 = (f.transpose()@f).trace()
        inv_F_T = (f.inverse()).transpose()
        P_elastic = G*(J**(-2/3)) * (f - (I1/3)*inv_F_T) + K*J*(J-1)*inv_F_T
        P_magnetic = -inv_mu0*applied_field[None].outer_product(flux_density)
        P = P_elastic + P_magnetic
        affine = -4*inv_dx*inv_dx*dt*p_vol*P@f.transpose()  + p_mass*C[p]
        # Quadratic Interpolation
        base = ((x[p]) * inv_dx - 0.5).cast(int)
        fx = (x[p]) * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j]) 
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass* v[p] + affine @ dpos) #momentum (uses same grid as velocity)
            grid_m[base + offset] += weight * p_mass


@ti.kernel
def grid_op():
     ############ GRID OPERATIONS ##############
    for i, j in grid_m:
        inv_m = 1 / (grid_m[i, j] + 1e-10)
        v_out = inv_m * grid_v[i, j] # Momentum to velocity
        v_out -= c*v_out*dt
        if grid_pin[i, j] == 1:
            v_out = [0.0, 0.0]
        if i < n_grid*0.01 and v_out[0] < 0:
            v_out[0] = 0  
        if i > n_grid*99 and v_out[0] > 0:
                v_out[0] = 0
        if j < n_grid*0.01 and v_out[1] < 0:
                v_out[1] = 0
        if j > n_grid*99 and v_out[1] > 0:
                v_out[1] = 0
        grid_v[i, j] = v_out

@ti.kernel
def g2p():
     ######### GRID TO PARTICLE #################
    for p in range(n_particles): 
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        base = ((x[p]) * inv_dx - 0.5).cast(int)
        fx = (x[p]) * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v 
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p] = new_v
        C[p] = new_C
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]  # deformation gradient update
        x[p] = x[p] + dt*v[p]  # advection
    
@ti.kernel
def calcAvg():
    for p in range(n_particles):
        if tip_particles[p] == 1:
            avg[None] += x[p] / n_tip_segments

def advance():
    clear_grid()
    p2g()
    grid_op()
    g2p()

def visualise(filename=None):
    gui.circles(x.to_numpy() / window_size, radius=1.5)
    gui.circle(avg.to_numpy() / window_size, color=0xFF0000, radius=5)
    gui.show(filename)

######################### Main ############################
gui = ti.GUI("Tentacle", res=800, background_color=0x888888)
fXLabel = gui.label('Field X (mT)')
timePassed = gui.label("Time passed (ms)")
deflection = gui.label("delta/L")
finish = gui.button("Finish")

renderEvery = 100

fields = np.array([20e-3, 5e-3, 7e-3, 10e-3, 15e-3, 25e-3, 40e-3])
fields = np.array([10e-3])
deflections = np.zeros_like(fields)
settle_time = 1


for j in range(fields.shape[0]):
    field = fields[j]
    applied_field[None] = [-field, 0]
    fXLabel.value = -field * 1e3
    reset()
    for i in range(int(settle_time/dt)):
        if i < 1e3:
            applied_field[None][1] = 3e-3
        else:
            applied_field[None][1] = 0e-3
        advance()
        if i % renderEvery == 0:
            timePassed.value = i*dt*1e3
            deflection.value = ((avg[None][1] - window_size/2)/length)
            avg[None] = [0.0, 0.0]
            calcAvg()
            visualise()
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == finish:
                break
    avg[None] = [0.0, 0.0]
    calcAvg()
    deflections[j] = avg[None][1] - window_size/2

y_axis = (deflections/length)
print(y_axis)
np.savetxt("mpm_data.csv", y_axis, delimiter=",")
