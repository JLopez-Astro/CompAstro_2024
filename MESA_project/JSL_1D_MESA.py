#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from scipy.interpolate import interp2d
import warnings
import time


# In[5]:


class Model:

    # Define constants, initial parameters, flags of the model.
    def __init__(self):
        self.G = const.G.value
        self.pi = np.pi
        self.k_B = const.k_B.value
        self.sigma = const.sigma_sb.value
        self.c = const.c.value
        self.m_u = const.u.value
        self.M_sun = const.M_sun.value
        self.R_sun = const.R_sun.value
        self.L_sun = const.L_sun.value
        self.rho_sun = 1.408*1e3 #kg m^-3
        X = 0.7
        Y = 0.29 + 1e-10
        Z = 1 - (X + Y)
        self.mu = (2*X + 3*Y/4 + Z/2)**-1
        self.C_p = (5*self.k_B)/(2*self.mu*self.m_u)
        
        # Define initial parameters
        # self.L_0 = self.L_sun
        # self.R_0 = self.R_sun
        # self.M_0 = self.M_sun
        # self.rho_0 = 1.0*1.42*1e-7*self.rho_sun #kg m^-3 [SI]
        # self.T_0 = 5770 #Kelvin
        # self.P_0 = self.P(self.T_0, self.rho_0)

        # Testing initial parameters
        self.L_0 = self.L_sun
        self.R_0 = 1.0*self.R_sun
        self.M_0 = 1.0*self.M_sun
        self.rho_0 = 1.0*1.42*1e-7*self.rho_sun #kg m^-3 [SI]
        self.T_0 = 5770 #Kelvin
        self.P_0 = self.P(self.T_0, self.rho_0)

        # BEST?
        # self.L_0 = self.L_sun
        # self.R_0 = 0.9 * self.R_sun
        # self.M_0 = 0.95 * self.M_sun
        # self.rho_0 = 54 * 1.*1.42*1e-7*self.rho_sun #kg m^-3 [SI]
        # self.T_0 = 5770 #Kelvin
        # self.P_0 = self.P(self.T_0, self.rho_0)
    
        # Enable sanity check
        self.opacity_sanity_check_option = False
        self.epsilon_sanity_check_option = False
        self.fin_step = 0
    
    # Define functions
    def P(self, T, rho):
        left = (4*self.sigma*(T**4))/(3*self.c)
        right = (rho*self.k_B*T)/(self.mu*self.m_u)
        return left+right

    def rho(self, P, T):
        left = (4*self.sigma*(T**4))/(3*self.c)
        right = (self.mu*self.m_u)/(self.k_B*T)
        return (P-left)*right
        
    def g(self, m, r):
        numerator = self.G*m
        denominator = r**2
        return numerator/denominator
    
    def H_p(self, T, g):
        numerator = self.k_B*T
        denominator = self.mu*self.m_u*g
        return numerator/denominator

    def flux(self, r, L):
        denominator = 4*self.pi*(r**2)
        return L/denominator

    def rad_flux(self, rho, kappa, T, H_p, temp_grad):
        numerator = 16*self.sigma*(T**4)
        denominator = 3*rho*kappa*H_p*temp_grad
        return numerator/denominator

    def conv_flux(self, flux, rad_flux):
        return flux - rad_flux
    
    def grad_stable(self, kappa, H_p, L, r, T, rho): # From project pdf
        numerator = 3*kappa*H_p*L*rho
        denominator = 64*self.pi*(r**2)*self.sigma*(T**4)
        return numerator/denominator

    # def grad_stable(self, kappa, H_p, L, r, T, rho): # From project pdf
    #     numerator = 3*kappa*H_p*L*rho
    #     denominator = 64*self.pi*(r**2)*self.sigma*(T**4)
    #     return numerator/denominator
    
    # def grad_stable(self, kappa, T, P, L, m, rho): # From Onno Pols
    #     numerator = 3*kappa*L*P
    #     denominator = 64*self.pi*self.sigma*self.G*m*(T**4)

    def grad_adiabatic(self, P, T, rho):
        numerator = P
        denominator = T*rho*self.C_p
        return numerator/denominator

    def grad_star(self, T, H_p, g, kappa, rho, grad_stable, grad_adiabatic):
        U_numerator = 64*self.sigma*(T**3)*np.sqrt(H_p/g)
        U_denominator = 3*kappa*(rho**2)*self.C_p
        U = U_numerator/U_denominator

        l_m = H_p
        omega = 4/l_m

        coeff = [1, U/(l_m**2), (U**2 * omega)/(l_m**3), (U/l_m**2)*(grad_adiabatic - grad_stable)]

        roots = np.roots(coeff)
        min_imag_root = min(roots, key=lambda r: np.abs(np.imag(r)))
        xi = np.real(min_imag_root)

        gradStar = xi**2 + ((U*omega)/l_m)*xi + grad_adiabatic

        return gradStar

    # Define ODE's
    def dr_dm(self, r, rho):
        denominator = 4*self.pi*(r**2)*rho
        return 1/denominator

    def dP_dm(self, r, m):
        numerator = -self.G*m
        denominator = 4*self.pi*(r**4)
        return numerator/denominator

    def dT_dm(self, kappa, L, r, T): # For when convection is NOT stable
        numerator = -3*kappa*L
        denominator = 256*(self.pi**2)*self.sigma*(r**4)*(T**3)
        return numerator/denominator
        
    def read_opacity(self): # CORRECT BUT REWRITE
            with open("opacity.txt", "r") as file:
                log_R = np.asarray(file.readline().split()[1:], dtype=np.float64)
    
                file.readline()
    
                log_T = []
                log_kappa = []
    
                for line in file:
                    log_T.append(float(line.split()[0]))
                    log_kappa.append(line.split()[1:])
                file.close()
    
            log_T = np.array(log_T)
            log_kappa = np.array(log_kappa)
    
            return log_R, log_T, log_kappa
            # print(f"log_R: {log_R}, log_T: {log_T}, log_kappa: {log_kappa}") #Check if reading properly

    
    def calc_opacity(self, T, rho): # CORRECT BUT REWRITE
        """
        This takes the input T and rho to interpolate the opacity, kappa. If the input values are not found
        within the opacity.txt file, linear 2D interpolation is performed to find the best kappa.
        Input:
            T - float, input temperature
            rho - float, input density
        Output:
            kappa_fin - float, opacity
        """
        log_R, log_T, log_kappa = self.read_opacity()

        warnings.filterwarnings('ignore')
        interp = interp2d(log_R, log_T, log_kappa, kind='linear')
        log_R_input = np.log10( (rho*1e-3)/ (T / 1e6)**3)
        log_T_input = np.log10(T)
        kappa = interp(log_R_input, log_T_input)[0]

        if log_T_input < log_T.min() or log_T_input > log_T.max():
            print("Warning: Opacity input values out of Temperature bounds. Extrapolating...")
        if log_R_input < log_R.min() or log_R_input > log_R.max():
            print("Warning: Opacity input values out of Radius bounds. Extrapolating...")
        kappa_fin = 10**kappa * 0.1

        if self.opacity_sanity_check_option:
            self.opacity_sanity_check()
        
        return kappa_fin

    def opacity_sanity_check(self):
        T_log10_ref = [3.750, 3.755, 3.755, 3.755, 3.755, 3.770, 3.780, 3.795, 3.770, 3.775, 3.780, 3.795, 3.800]
        R_cgs_ref = [-6.00, -5.95, -5.80, -5.70, -5.55, -5.95, -5.95, -5.95, -5.80, -5.75, -5.70, -5.55, -5.50]
        kappa_cgs_ref = [-1.55, -1.51, -1.57, -1.61, -1.67, -1.33, -1.20, -1.02, -1.39, -1.35, -1.31, -1.16, -1.11]
        kappa_SI_ref = [2.84, 3.11, 2.68, 2.46, 2.12, 4.70, 6.25, 9.45, 4.05, 4.43, 4.94, 6.89, 7.69]

        log_R, log_T, log_kappa = self.read_opacity()
        interp = interp2d(log_R, log_T, log_kappa, kind='linear')

        kappa_cgs_interp = []
        kappa_SI_interp = []
        for R_cgs_input, T_log10_input in zip(R_cgs_ref, T_log10_ref):
            kap_cgs = interp(R_cgs_input, T_log10_input)[0]
            kappa_cgs_interp.append(kap_cgs)
            kappa_SI_interp.append(10**kap_cgs * 0.1 *1e3)

        for i in range(len(kappa_cgs_ref)):
            cgs_check = np.abs(kappa_cgs_interp[i] - kappa_cgs_ref[i]) <= 0.05 * np.abs(kappa_cgs_ref[i])
            SI_check = np.abs(kappa_SI_interp[i] - kappa_SI_ref[i]) <= 0.05 * np.abs(kappa_SI_ref[i])

            if cgs_check and SI_check:
                print(f"PASSED: Calculations within 5%.")
                print(f"For log10(T) = {T_log10_ref[i]} and log10(R) = {R_cgs_ref[i]}:")
                print(f"INTERPOLATED Opacity: [cgs] = {kappa_cgs_interp[i]} and [SI] = {kappa_SI_interp[i]}")
                print(f"REFERENCE Opacity: [cgs] = {kappa_cgs_ref[i]} and [SI] = {kappa_SI_ref[i]}")
                print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            else:
                print(f"FAILED: Calculations NOT within 5%.")
                print(f"For log10(T) = {T_log10_ref[i]} and log10(R) = {R_cgs_ref[i]}:")
                print(f"INTERPOLATED Opacity: [cgs] = {kappa_cgs_interp[i]} and [SI] = {kappa_SI_interp[i]}")
                print(f"REFERENCE Opacity: [cgs] = {kappa_cgs_ref[i]} and [SI] = {kappa_SI_ref[i]}")
                print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def read_epsilon(self): # CORRECT BUT REWRITE
            with open("epsilon.txt", "r") as file:
                log_R = np.asarray(file.readline().split()[1:], dtype=np.float64)
    
                file.readline()
    
                log_T = []
                log_epsilon = []
    
                for line in file:
                    log_T.append(float(line.split()[0]))
                    log_epsilon.append(line.split()[1:])
                file.close()
    
            log_T = np.array(log_T)
            log_epsilon = np.array(log_epsilon)
    
            return log_R, log_T, log_epsilon
            # print(f"log_R: {log_R}, log_T: {log_T}, log_epsilon: {log_epsilon}") #Check if reading properly

    def calc_epsilon(self, T, rho): # CORRECT BUT REWRITE 
        """
        This takes the input T and rho to interpolate the energy released from fusion reactions in the core,
        epsilon. If the input values are not found within the epsilon.txt file, linear 2D interpolation is
        performed to find the best epsilon.
        Input:
            T - float, input temperature
            rho - float, input density
        Output:
            epsilon_fin - float, epsilon
        """
        log_R, log_T, log_epsilon = self.read_epsilon()

        warnings.filterwarnings('ignore')
        interp = interp2d(log_R, log_T, log_epsilon, kind='linear')
        log_R_input = np.log10( (rho*1e-3)/ (T / 1e6)**3)
        log_T_input = np.log10(T)
        epsilon = interp(log_R_input, log_T_input)[0]

        if log_T_input < log_T.min() or log_T_input > log_T.max():
            print("Warning: Epsilon input values out of Temperature bounds. Extrapolating...")
        if log_R_input < log_R.min() or log_R_input > log_R.max():
            print("Warning: Epsilon input values out of Radius bounds. Extrapolating...")
        
        epsilon_fin = 10**epsilon * 1e-4

        if self.epsilon_sanity_check_option:
            self.epsilon_sanity_check()
        
        return epsilon_fin


    def epsilon_sanity_check(self):
        T_log10_ref = [3.750, 3.755]
        R_cgs_ref = [-6.00, -5.95]
        epsilon_cgs_ref = [-87.995, -87.623]
        epsilon_SI_ref = [1.012, 2.415]
        
        log_R, log_T, log_epsilon = self.read_epsilon()
        interp = interp2d(log_R, log_T, log_epsilon, kind='linear')

        epsilon_cgs_interp = []
        epsilon_SI_interp = []
        for R_cgs_input, T_log10_input in zip(R_cgs_ref, T_log10_ref):
            eps_cgs = interp(R_cgs_input, T_log10_input)[0]
            epsilon_cgs_interp.append(eps_cgs)
            epsilon_SI_interp.append((10**eps_cgs) * 1e-4 * 1e92)

        for i in range(len(epsilon_cgs_ref)):
            cgs_check = np.abs(epsilon_cgs_interp[i] - epsilon_cgs_ref[i]) <= 0.05 * np.abs(epsilon_cgs_ref[i])
            SI_check = np.abs(epsilon_SI_interp[i] - epsilon_SI_ref[i]) <= 0.05 * np.abs(epsilon_SI_ref[i])

            if cgs_check and SI_check:
                print(f"PASSED: Calculations within 5%.")
                print(f"For log10(T) = {T_log10_ref[i]} and log10(R) = {R_cgs_ref[i]}:")
                print(f"INTERPOLATED Epsilon: [cgs] = {epsilon_cgs_interp[i]} and [SI] = {epsilon_SI_interp[i]}")
                print(f"REFERENCE Epsilon: [cgs] = {epsilon_cgs_ref[i]} and [SI] = {epsilon_SI_ref[i]}")
                print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            else:
                print(f"FAILED: Calculations NOT within 5%.")
                print(f"For log10(T) = {T_log10_ref[i]} and log10(R) = {R_cgs_ref[i]}:")
                print(f"INTERPOLATED Epsilon: [cgs] = {epsilon_cgs_interp[i]} and [SI] = {epsilon_SI_interp[i]}")
                print(f"REFERENCE Epsilon: [cgs] = {epsilon_cgs_ref[i]} and [SI] = {epsilon_SI_ref[i]}")
                print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def mass_step(self, L, T, r, P, epsilon, dr_dm, dP_dm, dT_dm):
        step_size = np.array([epsilon, dT_dm, dr_dm, dP_dm])
        step_size = np.abs(step_size)

        vals = np.array([L, T, r, P])
        m_step = np.min(0.01 * vals/step_size)
        if m_step < 1e19:
            m_step = 1e19
        # print(f"Step size: {step_size}")
        return m_step
    
    def ODE_euler(self):

        init_time = time.time()
        g = self.g(self.M_0, self.R_0) # Test
        H_p = self.H_p(self.T_0, g) # Test
        
        num_steps = int(1e5)
        r, m, rho, P = np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)
        r[0], m[0], rho[0], P[0] = self.R_0, self.M_0, self.rho_0, self.P_0
        
        F, L, T, epsilon = np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)
        L[0], T[0], epsilon[0] = self.L_0, self.T_0, self.calc_epsilon(self.T_0, self.rho_0)
        
        grad_temp, grad_stable, grad_adiabatic, grad_star = np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)
        grad_stable[0] = self.grad_stable(self.calc_opacity(self.T_0, self.rho_0), self.T_0, self.P_0, self.L_0, self.M_0, self.rho_0)
        grad_adiabatic[0] = self.grad_adiabatic(self.P_0, self.T_0, self.rho_0)
        # Test
        grad_star[0] = self.grad_star(T[0], H_p, g, self.calc_opacity(self.T_0, self.rho_0), self.rho_0, grad_stable[0], grad_adiabatic[0])
    # def grad_star(self, T, H_p, g, kappa, rho, grad_stable, grad_adiabatic):

        for i in np.arange(num_steps):
            if rho[i]<1e-4 or r[i]<1e-4 or P[i]<1e-4 or L[i]<1e-4 or T[i]<1e-4:
                print(f">>> ERROR: Variable reached 0 at step: {i}! <<<")
                print(f"Density = {rho[i]}")
                print(f"Pressure = {P[i]}")
                print(f"Temperature = {T[i]}")
                print(f"Mass = {m[i]/self.M_0:.2f} Msol")
                print(f"Radius = {r[i]/self.R_0:.2f} Rsol")
                print(f"Luminosity = {L[0]/self.L_0:.2f} Lsol")
                self.fin_step = i
                break
            elif m[i]<1e-4:
                print(f">>> SUCCESS: Mass reached 0 at step: {i}! <<<")
                print(f"Density = {rho[i]}")
                print(f"Pressure = {P[i]}")
                print(f"Temperature = {T[i]}")
                print(f"Mass = {m[i]/self.M_0:.2f} Msol")
                print(f"Radius = {r[i]/self.R_0:.2f} Rsol")
                print(f"Luminosity = {L[0]/self.L_0:.2f} Lsol")
                print("""
               *STAR BUILT*
                    A
                ___/_\___
                 ',. .,'
                 /.'^'.\ 
                /'     '\ 
                ~~~~~~~~~ """)
                
                self.fin_step = i
                break
            else:
                kappa = self.calc_opacity(T[i], rho[i])
                g = self.g(m[i], r[i])
                H_p = self.H_p(T[i], g)
                F[i] = self.flux(r[i], L[i])

                grad_adiabatic[i] = self.grad_adiabatic(P[i], T[i], rho[i])
                grad_stable[i] = self.grad_stable(kappa, H_p, L[i], r[i], T[i], rho[i]) # Project pdf
                # grad_stable[i] = self.grad_stable(kappa, T[i], P[i], L[i], m[i], rho[i]) # Onno

                grad_star[i] = self.grad_star(T[i], H_p, g, kappa, rho[i], grad_stable[i], grad_adiabatic[i])

                if grad_stable[i] > grad_adiabatic[i]: # Check for convection stability
                    grad_temp[i] = grad_star[i]
                    dT_dm = (T[i]/P[i])*self.dP_dm(r[i], m[i])
                else:
                    grad_temp[i] = grad_stable[i]
                    dT_dm = self.dT_dm(kappa, L[i], r[i], T[i])

                dr_dm = self.dr_dm(r[i], rho[i]) # Calculate dr_dm
                dP_dm = self.dP_dm(r[i], m[i]) # Calculate dP_dm
                curr_eps = self.calc_epsilon(T[i], rho[i]) # Current step epsilon
                
                mass_step = self.mass_step(L[i], T[i], r[i], P[i], curr_eps, dr_dm, dP_dm, dT_dm)
                
                m[i+1] = m[i] - mass_step
                r[i+1] = r[i] - self.dr_dm(r[i], rho[i])*mass_step
                P[i+1] = P[i] - self.dP_dm(r[i], m[i])*mass_step
                T[i+1] = T[i] - dT_dm*mass_step
                rho[i+1] = self.rho(P[i+1], T[i+1])
                L[i+1] = L[i] - curr_eps*mass_step
                epsilon[i+1] = self.calc_epsilon(T[i+1], rho[i+1])

                if self.fin_step == 0:
                    self.fin_step = num_steps
                
                # print(f"pressure: {P[i]} | temperature: {T[i]}")
                # print(f"mass step: {mass_step}")
                # print(f"luminosity = {L[i]}")
                # print(f"epsilon = {curr_eps}")
                # print(f"GRAD STABLE = {grad_stable[i]}")
                # print(f"---------------------------")

        return r, m, rho, P, F, L, T, epsilon, grad_temp, grad_stable, grad_adiabatic, grad_star


    def gen_plots(self, r, m, rho, P, L, T, grad_stable, grad_adiabatic, grad_star):
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (12,8))
        fig.subplots_adjust(wspace=0.4,hspace=0.3)
        
        ax1.set_title("Mass")
        ax1.set_xlabel(r"$\frac{r}{R_{0}}$", fontsize = 15)
        ax1.set_ylabel(r"$\frac{m}{M_{0}}$", fontsize = 15)
        ax1.plot(r[0:self.fin_step]/self.R_0, m[0:self.fin_step]/self.M_0, color='purple')
        ax1.grid()
        
        ax2.set_title("Density")
        ax2.set_xlabel(r"$\frac{r}{R_{0}}$", fontsize = 15)
        ax2.set_ylabel(r"$\frac{\rho}{\rho_{0}}$", fontsize = 15)
        ax2.plot(r[0:self.fin_step]/self.R_0, rho[0:self.fin_step]/self.rho_0, color='purple')
        ax2.set_yscale("symlog")
        ax2.grid()

        ax3.set_title("Pressure")
        ax3.set_xlabel(r"$\frac{r}{R_{0}}$", fontsize = 15)
        ax3.set_ylabel(r"$\frac{P}{P_{0}}$", fontsize = 15)
        ax3.plot(r[0:self.fin_step]/self.R_0, P[0:self.fin_step]/self.P_0, color='purple')
        ax3.set_yscale("symlog")
        ax3.grid()

        ax4.set_title("Luminosity")
        ax4.set_xlabel(r"$\frac{r}{R_{0}}$", fontsize = 15)
        ax4.set_ylabel(r"$\frac{L}{L_{0}}$", fontsize = 15)
        ax4.plot(r[0:self.fin_step]/self.R_0, L[0:self.fin_step]/self.L_0, color='purple')
        ax4.grid()

        ax5.set_title("Temperature")
        ax5.set_xlabel(r"$\frac{r}{R_{0}}$", fontsize = 15)
        ax5.set_ylabel("$T$", fontsize = 15)
        ax5.plot(r[0:self.fin_step]/self.R_0, T[0:self.fin_step], color='purple') #*1e-6 M K
        ax5.grid()

        ax6.set_title("Temperature Gradients (for p = 0.01)")
        ax6.set_xlabel(r"$\frac{r}{R_{0}}$", fontsize = 15)
        ax6.set_ylabel(r"$\nabla$", fontsize = 15)
        ax6.plot(r[0:self.fin_step]/self.R_0, grad_stable[0:self.fin_step], label="Grad_stable")
        ax6.plot(r[0:self.fin_step]/self.R_0, grad_star[0:self.fin_step], label="Grad_star")
        ax6.plot(r[0:self.fin_step]/self.R_0, grad_adiabatic[0:self.fin_step], label="Grad_adiabatic")
        ax6.set_yscale("symlog")
        ax6.legend()
        ax6.grid()
        
        fig.show()

    def plot_cross_star(self, r, L, grad_stable, grad_adiabatic):
    
        core_bounds = 0.995*np.max(L)
        r = r/self.R_0
    
        core_radii = r[np.where(L<core_bounds)[0]]
        shell_radii = r[np.where(L>core_bounds)[0]]
    
        grad_st_core = grad_stable[np.where(L<=core_bounds)[0]]
        grad_st_shell = grad_stable[np.where(L>core_bounds)[0]]
    
        grad_ad_core = grad_adiabatic[np.where(L<=core_bounds)[0]]
        grad_ad_shell = grad_adiabatic[np.where(L>core_bounds)[0]]
    
        fig = plt.figure(figsize=(6,6))
        ax=plt.gca()
    
        shell_conv_radii = shell_radii[np.where(grad_st_shell>grad_ad_shell)[0]]
        red = plt.Circle((0,0), np.max(shell_conv_radii), color='red',fill=True) #, lw = 2
        ax.add_patch(red)
        
        shell_ad_radii = shell_radii[np.where(grad_st_shell<=grad_ad_shell)[0]]
        yellow = plt.Circle((0,0), np.max(shell_ad_radii), color='yellow', fill=True)
        ax.add_patch(yellow)
        
        core_ad_radii = core_radii[np.where(grad_st_core<=grad_ad_core)[0]]
        cyan = plt.Circle((0,0), np.max(core_ad_radii), color='cyan', fill=True)
        ax.add_patch(cyan)
        
        core_conv_radii = core_radii[np.where(grad_st_core>grad_ad_core)[0]]
        blue = plt.Circle((0,0), np.max(core_conv_radii), color='blue', fill=True)
        ax.add_patch(blue)
    
        core = r[self.fin_step]
        white = plt.Circle((0,0), core, color='white', fill=True)
        ax.add_patch(white)
    
        labels = ['Convection: Outside Core','Radiation: Outside Core','Radiation: Inside Core','Convection: Inside Core']
        plt.legend([red,yellow,cyan,blue], labels)

        r_max = 1.2*np.max(r)
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.set_xlabel(r'$r/R_{0}$')
        ax.set_ylabel(r'$r/R_{0}$')
        ax.set_title('Cross Section of Star')
        plt.show()
        return len(shell_radii), len(grad_st_shell), len(grad_ad_shell)
        
    
        # for plotting the white center
        
            #split the total r into two groups, where L<0.995*max(L) and the other half
            #where L is less than 0.995*Lmax, this is the core, the rest is the rest of the star
            #for each of those 2 sub categories, split those into two, convection/radiation
            #through gradient comparison
    
            #convection occurs where grad_stable > grad_ad
            


# In[6]:


if __name__ == "__main__":
    # Create an instance of the Model class
    model = Model()
    
    r, m, rho, P, F, L, T, epsilon, grad_temp, grad_stable, grad_adiabatic, grad_star = model.ODE_euler()
    model.gen_plots(r, m, rho, P, L, T, grad_stable, grad_adiabatic, grad_star)
    model.plot_cross_star(r, L, grad_stable, grad_adiabatic)
    # Sanity Check Test
    # T_example = 5770
    # rho_example = 1.42*1e-7*model.rho_sun 

    # kappa = model.calc_opacity(T_example, rho_example)
    # print(f"Calculated opacity: {kappa}") # Opacity

    # epsilon = model.calc_epsilon(T_example, rho_example)
    # print(f"Calculated epsilon: {epsilon}") # Epsilon


# In[8]:


print(f"grad_stable: {list(grad_stable)}")
print(f"grad_adiabatic: {grad_adiabatic}")
print(f"grad_star: {grad_star}")

