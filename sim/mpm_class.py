import numpy as np
import taichi as ti
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

inv_mu0 = 1/(4*np.pi*1e-7) #1/(H/m)
p_vol = 1

@ti.data_oriented
class magneticMPM:
    def __init__(self, robot, scale=1, grid_size=100e-3, dx=1e-3, g=np.array([0, 0, -9.81]), gamma=20, offset=np.array([0, 0, 0]), colour_palette=None):
        self.robot = robot
        self.grid_size = grid_size
        self.dx = dx
        self.g = ti.Vector(g)
        self.gamma = gamma
        self.palette = colour_palette

        self.X = np.concatenate(robot.particles, axis=0, dtype=np.float32)
        self.n_particles = self.X.shape[0]
        print("n_particles", self.n_particles)

        self.x = ti.Vector.field(3, dtype=float, shape=(self.n_particles))     # position
        self.v = ti.Vector.field(3, dtype=float, shape=(self.n_particles))     # velocity
        self.mag = ti.Vector.field(3, dtype=float, shape=(self.n_particles))
        self.C = ti.Matrix.field(3, 3, dtype=float, shape=(self.n_particles))  # affine velocity field
        self.F = ti.Matrix.field(3, 3, dtype=float, shape=(self.n_particles))  # deformation gradient
        self.particle_type = ti.field(dtype=int, shape=(self.n_particles))
        self.fixed_particles = ti.field(dtype=int, shape=(self.n_particles))

        self.x_initial = ti.Vector.field(3, dtype=float, shape=(self.n_particles))     # position
        self.v_initial = ti.Vector.field(3, dtype=float, shape=(self.n_particles))     # velocity
        self.C_initial = ti.Matrix.field(3, 3, dtype=float, shape=(self.n_particles))  # affine velocity field
        self.F_initial = ti.Matrix.field(3, 3, dtype=float, shape=(self.n_particles))  # deformation gradient

        self.particle_mass = ti.field(dtype=float, shape=(self.n_particles))
        self.shear_modulus = ti.field(dtype=float, shape=(self.n_particles))

        self.x_visualise = ti.Vector.field(3, dtype=float, shape=(self.n_particles))
        self.colors = ti.Vector.field(3, dtype=float, shape=(self.n_particles))

        self.epm_pos = ti.Vector.field(3, dtype=float, shape=())
        self.epm_moment = ti.Vector.field(3, dtype=float, shape=())

        self.segments = robot.n_segments
        self.particles_per_seg = ti.field(dtype=int, shape=(robot.n_segments))

        self.seg_magnetisations = ti.Vector.field(3, dtype=float, shape=(self.robot.n_segments))
        self.seg_magnetisations_avg = ti.Vector.field(3, dtype=float, shape=(self.robot.n_segments))


        self.velocity = ti.field(dtype=float, shape=())
        self.velocity[None] = 0
        

        self.remenance = ti.field(dtype=float, shape=())
        self.remenance[None] = self.robot.remenance

        self.x.from_numpy(self.X * scale) 
        print("a")
        n = 0
        for i in range(robot.n_segments):
            self.particles_per_seg[i] = robot.particles[i].shape[0]
            for _ in range(robot.particles[i].shape[0]):
                self.particle_type[n] = i
                self.colors[n] = self.palette[i, :]
                self.particle_mass[n] = p_vol*robot.density[i]
                self.shear_modulus[n] = robot.shear_modulus[i]
                if robot.fixedSegments[i] == True:
                    self.fixed_particles[n] = 1
                else:
                    self.fixed_particles[n] = 0
                n += 1
        print("b")
        self.init_in_taichi(ti.Vector(offset))
        self.magnetise(self.robot.magnetisations)

        # Simulation Parameters
        self.n_grid = int(self.grid_size / self.dx)
        print("n_grid", self.n_grid)
        self.inv_dx = 1/dx
    
        self.grid_v = ti.Vector.field(3, dtype=float, shape=(self.n_grid, self.n_grid, self.n_grid))      # grid node momentum
        self.grid_m = ti.field(dtype=float, shape=(self.n_grid, self.n_grid, self.n_grid))                # grid node mass
        self.grid_mu = ti.Vector.field(3, dtype=float, shape=(self.n_grid, self.n_grid, self.n_grid))   
        self.grid_static = ti.field(dtype=int, shape=(self.n_grid, self.n_grid, self.n_grid))
        self.field = ti.Vector.field(3, dtype=float, shape=())
    
    @ti.kernel
    def init_in_taichi(self, offset: ti.template()):
        for i in range(self.n_particles):
            self.x[i] += offset
            self.v[i] = [0.0, 0.0, 0.0]
            self.C[i] = [[0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]]
            self.F[i] = [[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]
        for i in range(self.n_particles):
            self.x_initial[i] = self.x[i]
            self.v_initial[i] = self.v[i]
            self.C_initial[i] = self.C[i]
            self.F_initial[i] = self.F[i]

    

    @ti.kernel
    def clear_grid(self):
        # Clear Grid
        for i, j, k in self.grid_m:
            self.grid_v[i, j, k] = [0.0, 0.0, 0.0]
            self.grid_mu[i, j, k] = [0.0, 0.0, 0.0]
            self.grid_m[i, j, k] = 0.0
            self.grid_static[i, j, k] = 0

    @ti.kernel
    def p2g(self, dt: float):
        ######## PARTICLE TO GRID ############
        for p in range(self.n_particles): 
            G = self.shear_modulus[p]
            K = 20*G
            p_mass = self.particle_mass[p]
            magnetisation = self.mag[p]*self.remenance[None]
            # Calculate Stress
            f = self.F[p]
            J = f.determinant()
            I1 = (f.transpose()@f).trace()
            inv_F_T = (f.inverse()).transpose()
            P_elastic = G*(J**(-2/3)) * (f - (I1/3)*inv_F_T) + K*J*(J-1)*inv_F_T
            P_magnetic = -inv_mu0*self.field[None].outer_product(magnetisation)
            P = P_elastic + P_magnetic
            affine = -4*self.inv_dx*self.inv_dx*dt*p_vol*P@f.transpose()  + p_mass*self.C[p]
            # Quadratic Interpolation
            base = ((self.x[p]) * self.inv_dx - 0.5).cast(int)
            fx = (self.x[p]) * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            flux_density_rot = (1/J)*f@magnetisation
            moment = inv_mu0*flux_density_rot*p_vol
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  # Loop over 3x3x3 grid node neighborhood
                offset = ti.Vector([i, j, k]) 
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                self.grid_v[base + offset] += weight * (p_mass* self.v[p] + affine @ dpos) #momentum (uses same grid as velocity)
                self.grid_m[base + offset] += weight * p_mass
                self.grid_mu[base + offset] += weight * moment
                if self.fixed_particles[p] == 1:
                    self.grid_static[base + offset] = 1

    @ti.kernel
    def grid_op(self, dt: float):
        ############ GRID OPERATIONS ##############
        for i, j, k in self.grid_m:
            if self.grid_m[i, j, k] > 0:
                inv_m = 1 / (self.grid_m[i, j, k])
                v_out = inv_m * self.grid_v[i, j, k] # Momentum to velocity
                v_out -= self.gamma*v_out*dt  # damping
                v_out += self.g*dt # gravity

                if i <= 1 and v_out[0] < 0:
                    v_out = [0.0, 0.0, 0.0]  
                if j <= 1 and v_out[1] < 0:
                    v_out = [0.0, 0.0, 0.0]  
                if k <= 1 and v_out[2] < 0:
                    v_out = [0.0, 0.0, 0.0]  
                if i >= self.n_grid-1 and v_out[0] > 0:
                    v_out = [0.0, 0.0, 0.0]  
                if j >= self.n_grid-1 and v_out[1] > 0:
                    v_out = [0.0, 0.0, 0.0]  
                if k >= self.n_grid-1 and v_out[2] > 0:
                    v_out = [0.0, 0.0, 0.0]  

                if self.grid_static[i, j, k] == 1:
                    v_out = [0.0, 0.0, 0.0]
                if k >= self.n_grid/2:
                    v_out = [0.0, 0.0, self.velocity[None]]
                self.grid_v[i, j, k] = v_out

    @ti.kernel
    def g2p(self, dt: float):
        ######### GRID TO PARTICLE #################
        for p in range(self.n_particles): 
            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)
            base = ((self.x[p]) * self.inv_dx - 0.5).cast(int)
            fx = (self.x[p]) * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j, k]).cast(float) - fx
                g_v = self.grid_v[base + ti.Vector([i, j, k])]
                weight = w[i][0] * w[j][1] * w[k][2]
                new_v += weight * g_v 
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p] = new_v
            self.C[p] = new_C
            self.F[p] = (ti.Matrix.identity(float, 3) + dt * self.C[p]) @ self.F[p]  # deformation gradient update
            self.x[p] = self.x[p] + dt*self.v[p]  # advection

        
    @ti.kernel
    def getParticles(self, scale: float):
        for i in range(self.n_particles):
            self.x_visualise[i][0] = self.x[i][0] * scale
            self.x_visualise[i][1] = self.x[i][2] * scale
            self.x_visualise[i][2] = self.x[i][1] * scale

    def advance(self, dt):
        self.clear_grid()
        self.p2g(dt)
        self.grid_op(dt)
        self.g2p(dt)

    @ti.kernel
    def __calc_avg_positions(self):
        for n in range(self.n_particles):
            for i in ti.static(range(self.segments)):
                if self.particle_type[n] == i:
                    self.avg_positions[i] += self.x[n] / self.particles_per_seg[i]

    def get_avg_positions(self):
        for i in range(self.segments):
            self.avg_positions[i] = [0.0, 0.0, 0.0]
        self.__calc_avg_positions()
        return self.avg_positions

    @ti.kernel
    def reset(self):
        for i in range(self.n_particles):
            self.x[i] = self.x_initial[i]
            self.v[i] = self.v_initial[i]
            self.C[i] = self.C_initial[i]
            self.F[i] = self.F_initial[i]

    @ti.kernel
    def __calc_avg_magnetisations(self):
         for n in range(self.n_particles):
            for i in ti.static(range(self.segments)):
                if self.particle_type[n] == i:
                    f = self.F[n]
                    J = f.determinant()
                    mag_rot = (1/J)*f@self.mag[n]
                    self.seg_magnetisations_avg[i] += mag_rot / self.particles_per_seg[i]
    
    def get_avg_magnetisations(self):
        for i in range(self.segments):
            self.seg_magnetisations_avg[i] = [0.0, 0.0, 0.0]
        self.__calc_avg_magnetisations()
        return self.seg_magnetisations_avg
        
    @ti.kernel
    def __magnetiseTaichi(self):
        for p in range(self.n_particles):
            for i in ti.static(range(self.segments)):
                if self.particle_type[p] == i:
                    self.mag[p] =  self.seg_magnetisations[i]
    
    def magnetise(self, segment_magnetisations):
        for i in range(self.segments):
            self.seg_magnetisations[i] = segment_magnetisations[i]
        self.__magnetiseTaichi()


    def setField(self, field):
        self.field[None] = field

