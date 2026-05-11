

class Forced_2D_NS:
    L = 2 * jnp.pi

    def __init__(self, Re, n, N, double=False):
        self.Re = Re
        self.M = self.get_dealias_mask(N)      
        KX, KY = self.get_K(N)
        self.dxop = 1j * KX
        self.dyop = 1j * KY

        forcing = self.kolmogorov_vorticity_forcing(self.L, N, n)  
        self.forcing_hat = jnp.fft.rfft2(forcing)  
        self.M = self.M.astype(jnp.float32)  
        if not double:
            self.dxop = self.dxop.astype(jnp.complex64)  
            self.dyop = self.dyop.astype(jnp.complex64)  
            self.forcing_hat = self.forcing_hat.astype(jnp.complex64)  

        laplacian_op = (self.dxop**2 + self.dyop**2)
        self.diff_op = (1 / Re) * laplacian_op
        self.laplacian_op_safe = laplacian_op.at[0, 0].set(1)

    @staticmethod
    def get_K(N):
        L = Forced_2D_NS.L
        dx = L / N
        kx = 2 * jnp.pi * jnp.fft.rfftfreq(N, d=dx)
        ky = 2 * jnp.pi * jnp.fft.fftfreq(N, d=dx)
        KX, KY = jnp.meshgrid(kx, ky, indexing="xy")
        return KX, KY
    
    @staticmethod
    def kolmogorov_vorticity_forcing(L, N, n):
        y = jnp.linspace(0.0, L, N, endpoint=False)
        x = jnp.linspace(0.0, L, N, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="xy")
        return -n * jnp.cos(n * Y)

    @staticmethod
    def get_dealias_mask(N):
        mx_full = jnp.fft.fftfreq(N) * N
        my_full = jnp.fft.fftfreq(N) * N
        MX, MY = jnp.meshgrid(mx_full, my_full, indexing="xy")
        M_full = (jnp.abs(MX) <= N/3) & (jnp.abs(MY) <= N/3)
        return M_full[:, :N//2 + 1]

    def vort_hat_2_vel_hat(self, omega_hat):
        psi_hat = omega_hat / self.laplacian_op_safe   
        u_hat = self.dyop * psi_hat
        v_hat = -self.dxop * psi_hat
        return u_hat, v_hat

    def explicit_term(self, omega_hat):
        u_hat, v_hat = self.vort_hat_2_vel_hat(omega_hat)
        u = jnp.fft.irfft2(u_hat)
        v = jnp.fft.irfft2(v_hat)

        dw_dx = jnp.fft.irfft2(self.dxop * omega_hat)
        dw_dy = jnp.fft.irfft2(self.dyop * omega_hat)

        adv = -(u * dw_dx + v * dw_dy)
        adv_hat = jnp.fft.rfft2(adv)
        adv_hat = adv_hat * self.M
        return adv_hat + self.forcing_hat, u, v
    
    def implicit_term(self, omega_hat):
        return self.diff_op * omega_hat
    
    def implicit_solve(self, omega_hat, mu):
        return omega_hat / (1 - mu * self.diff_op)
    
class KF_Stepper:
    alpha = [0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1]
    beta  = [0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]
    gamma = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]

    def __init__(self, Re, n, N, dt, double=True):
        self.NS = Forced_2D_NS(Re, n, N, double=double)
        self.dt = dt

    def calc_h(self, g, h, i):
        return g + self.beta[i] * h 

    def calc_mu(self, i):
        return 0.5 * self.dt * (self.alpha[i+1] - self.alpha[i])
    
    def calc_imp_rhs(self, h, mu, u, i):
        return u + self.gamma[i] * self.dt * h + mu * self.NS.implicit_term(u)

    def __call__(self, u_n): 
        u = u_n
        h = jnp.zeros_like(u_n)
        for i in range(5):
            g, _, _ = self.NS.explicit_term(u)   # g(u, t_k)
            h = self.calc_h(g, h, i)             # h <- g + beta*h

            mu = self.calc_mu(i)
            rhs = self.calc_imp_rhs(h, mu, u, i)
            u = self.NS.implicit_solve(rhs, mu)

        return u
    
class Tracer_Evolution:
    @staticmethod
    def part_pos_update(h_p, pos, fluid_vel, beta_coef, gamma_coef, dt):
        h_p = fluid_vel + beta_coef * h_p
        pos = pos + gamma_coef * dt * h_p
        return pos, h_p
    
    def __call__(
            self, 
            xp, yp, h_xp, h_yp, 
            up, vp, h_up, h_vp,
            u_fluid, v_fluid, 
            beta_coef, gamma_coef, dt
    ):
        xp, h_xp = self.part_pos_update(h_xp, xp, u_fluid, beta_coef, gamma_coef, dt)
        yp, h_yp = self.part_pos_update(h_yp, yp, v_fluid, beta_coef, gamma_coef, dt)
        return xp, yp, h_xp, h_yp, up, vp, h_up, h_vp


#KF and Tracer Particles
class KF_TP_Stepper(KF_Stepper):
    def __init__(self, Re, n, N, dt, St, beta, npart, double=True):
        super().__init__(Re, n, N, dt, double=double)
        if St == 0 and beta == 0:
            self.p_ev = Tracer_Evolution()
        else:
            self.p_ev = Inertial_Evolution(St)
        
        self.h = jnp.zeros((N, N//2+1))
        if double:
            self.h = self.h.astype(jnp.complex128)
        else:
            self.h = self.h.astype(jnp.complex64)
        self.h_xp = jnp.zeros(npart)
        self.h_yp = jnp.zeros(npart)
        self.h_up = jnp.zeros(npart)
        self.h_vp = jnp.zeros(npart)

    def part_pos_update(self, i, h_p, pos, fluid_vel):
        h_p = fluid_vel + self.beta[i] * h_p
        pos = pos + self.gamma[i] * self.dt * h_p
        return pos, h_p

    def __call__(self, omega_hat, xp, yp, up, vp):
        h = self.h * 0
        h_xp = self.h_xp * 0
        h_yp = self.h_yp * 0
        h_up = self.h_up * 0
        h_vp = self.h_vp * 0
        for i in range(5):
            g, u_grid, v_grid = self.NS.explicit_term(omega_hat)
            h = self.calc_h(g, h, i)

            mu = self.calc_mu(i)
            rhs = self.calc_imp_rhs(h, mu, omega_hat, i)
            omega_hat = self.NS.implicit_solve(rhs, mu)

            # sample fluid velocity at particle positions
            u = bilinear_sample_periodic(u_grid, xp, yp, self.NS.L, self.NS.L)
            v = bilinear_sample_periodic(v_grid, xp, yp, self.NS.L, self.NS.L)

            xp, yp, h_xp, h_yp, up, vp, h_up, h_vp = self.p_ev(
                xp, yp, h_xp, h_yp, 
                up, vp, h_up, h_vp,
                u, v, 
                self.beta[i], self.gamma[i], self.dt
            )

        xp = jnp.mod(xp, self.NS.L)
        yp = jnp.mod(yp, self.NS.L)
        return omega_hat, xp, yp, up, vp




