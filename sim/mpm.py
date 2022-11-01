import numpy as np
import taichi as ti


inv_mu0 = 1/(4*np.pi*1e-7) #1/(H/m)
p_vol = 1


def init(robot, scale=1, grid_size_=100e-3, dx_=1e-3, g_=np.array([0, 0, -9.81]), gamma_=20, offset=np.array([0, 0, 0]), colour_palette=None):
    global x, v, C, F, n_particles, mag, x_visualise, \
        grid_size, dx, g, gamma, n_grid, inv_dx, grid_v, grid_m, applied_field, \
        particle_type, grid_static, fixed_particles, colors, particle_mass, bulk_modulus, palette
    
    grid_size = grid_size_
    dx = dx_
    g = ti.Vector(g_)
    gamma = gamma_
    palette = colour_palette

    X = np.concatenate(robot.particles, axis=0, dtype=np.float32)
    n_particles = X.shape[0]
    print("n_particles", n_particles)

    x = ti.Vector.field(3, dtype=float, shape=(n_particles))     # position
    v = ti.Vector.field(3, dtype=float, shape=(n_particles))     # velocity
    mag = ti.Vector.field(3, dtype=float, shape=(n_particles))
    C = ti.Matrix.field(3, 3, dtype=float, shape=(n_particles))  # affine velocity field
    F = ti.Matrix.field(3, 3, dtype=float, shape=(n_particles))  # deformation gradient
    particle_type = ti.field(dtype=int, shape=(n_particles))
    fixed_particles = ti.field(dtype=int, shape=(n_particles))

    particle_mass = ti.field(dtype=float, shape=(n_particles))
    bulk_modulus = ti.field(dtype=float, shape=(n_particles))

    x_visualise = ti.Vector.field(3, dtype=float, shape=(n_particles))
    colors = ti.Vector.field(3, dtype=float, shape=(n_particles))

    x.from_numpy(X * scale) 
    n = 0
    for i in range(robot.n_segments):
        for _ in range(robot.particles[i].shape[0]):
            particle_type[n] = i
            colors[n] = palette[i, :]
            particle_mass[n] = p_vol*robot.density[i]
            bulk_modulus[n] = robot.bulk_modulus[i]
            if robot.fixedSegments[i] == True:
                fixed_particles[n] = 1
            else:
                fixed_particles[n] = 0
            mag[n] = robot.magnetisations[i]*robot.remenance
            n += 1

    init_in_taichi(ti.Vector(offset))

    # Simulation Parameters
    n_grid = int(grid_size / dx)
    print("n_grid", n_grid)
    inv_dx = 1/dx
   
    grid_v = ti.Vector.field(3, dtype=float, shape=(n_grid, n_grid, n_grid))      # grid node momentum
    grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))                # grid node mass
    grid_static = ti.field(dtype=int, shape=(n_grid, n_grid, n_grid))
    applied_field = ti.Vector.field(3, dtype=float, shape=())

    

@ti.kernel
def init_in_taichi(offset: ti.template()):
    for i in range(n_particles):
        x[i] += offset
        v[i] = [0.0, 0.0, 0.0]
        C[i] = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]]
        F[i] = [[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]]

#########################################################################################################

@ti.kernel
def clear_grid():
    # Clear Grid
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0.0, 0.0, 0.0]
        grid_m[i, j, k] = 0.0
        grid_static[i, j, k] = 0

@ti.kernel
def p2g(dt: float):
    ######## PARTICLE TO GRID ############
    for p in range(n_particles): 
        G = bulk_modulus[p]
        K = 20*G
        p_mass = particle_mass[p]
        # Calculate Stress
        f = F[p]
        J = f.determinant()
        I1 = (f.transpose()@f).trace()
        inv_F_T = (f.inverse()).transpose()
        P_elastic = G*(J**(-2/3)) * (f - (I1/3)*inv_F_T) + K*J*(J-1)*inv_F_T
        P_magnetic = -inv_mu0*applied_field[None]@mag[p].transpose()
        P = P_elastic + P_magnetic
        affine = -4*inv_dx*inv_dx*dt*p_vol*P@f.transpose()  + p_mass*C[p]
        # Quadratic Interpolation
        base = ((x[p]) * inv_dx - 0.5).cast(int)
        fx = (x[p]) * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  # Loop over 3x3x3 grid node neighborhood
            offset = ti.Vector([i, j, k]) 
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offset] += weight * (p_mass* v[p] + affine @ dpos) #momentum (uses same grid as velocity)
            grid_m[base + offset] += weight * p_mass
            if fixed_particles[p] == 1:
                grid_static[base + offset] = 1

@ti.kernel
def grid_op(dt: float):
     ############ GRID OPERATIONS ##############
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            inv_m = 1 / (grid_m[i, j, k])
            v_out = inv_m * grid_v[i, j, k] # Momentum to velocity
            v_out -= gamma*v_out*dt
            v_out += g*dt
            if k <= 1 and v_out[2] < 0:
                v_out = [0.0, 0.0, 0.0]  
            if grid_static[i, j, k] == 1:
                v_out = [0.0, 0.0, 0.0]
            grid_v[i, j, k] = v_out

@ti.kernel
def g2p(dt: float):
     ######### GRID TO PARTICLE #################
    for p in range(n_particles): 
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        base = ((x[p]) * inv_dx - 0.5).cast(int)
        fx = (x[p]) * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j, k]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v 
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p] = new_v
        C[p] = new_C
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]  # deformation gradient update
        x[p] = x[p] + dt*v[p]  # advection


def setField(field):
    applied_field[None] = field
    
@ti.kernel
def getParticles(scale: float):
    for i in range(n_particles):
        x_visualise[i][0] = x[i][0] * scale
        x_visualise[i][1] = x[i][2] * scale
        x_visualise[i][2] = x[i][1] * scale

def advance(dt):
    clear_grid()
    p2g(dt)
    grid_op(dt)
    g2p(dt)

