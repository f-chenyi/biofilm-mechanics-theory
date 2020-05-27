#!/usr/bin/env python3
# -*- coding: utf-8 -*-




#################################################
### 1. Required packages to run the simulation ###
import random
from dolfin import *
import ufl
import numpy as np
from math import pi
import os
from itertools import chain
import time
set_log_active(False)
np.set_printoptions(precision=4)
#################################################




#################################################
### 2. Parameters ###

# (+) Define file name

myname = "data/"
date   = "/"
trial  = 1
debug_text = "Insert simulation conditions here."


# (+) Model parameters

ksi_list   = [36.0]              # dimensionless friction
ac_list    = [0.5]               # width of the nutrient-rich annulus
Q0_list    = [0.8]               # maximum uptake
kr_list    = [0.15]              # residual growth rate
Eps_c_list = [9.0]               # critical stress for wrinkling
                                 # Note: no wrinkling model is achieved by assigning a large value to Eps_c

ksi   = Expression('my_ksi', degree=0, my_ksi = 1.0)
kr    = Expression('my_kr',  degree=0, my_kr  = 0.0)
Eps_c = Expression('my_Nc',  degree=0, my_Nc  = 2.0)
Dn    = Expression('my_Dn',  degree=0, my_Dn  = 0.5)
Q0    = Expression('my_Q0',  degree=0, my_Q0  = 2.0)

#  nutrient field
c0    = 1.0                         # constant supply at the biofilm edge
rho0  = 1.0
chalf = 0.5*c0

# growth
k_growth_o = (chalf + c0)/c0
k_growth = 1.5*(1.0-kr)

Rc = 1.0                            # radius of the circle



# (+) Time evolution

t     = 0.0                                       # initial time
dt    = Expression("dt",degree = 0, dt = 0.01)
T     = 6.0                                       # max simulation time
dt_of = 0.05                                      # output time interval

#################################################





#################################################
### 3. Create mesh and function space ###

# (+) Mesh

mesh = Mesh("1d_line.xml")
d    = mesh.geometry().dim()


# (+) Function space

P1      = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # Scalar element
P2      = VectorElement("Lagrange", mesh.ufl_cell(), 1) # Vector element
element = MixedElement([P1, P1, P1, P1, P1, P1])
ME      = FunctionSpace(mesh, element)
V1      = FunctionSpace(mesh,'P',1)


# (+) Define functions
# ur    = radial displacement,
# lmda  = growth stretch
# nc    = nutrient concentration
# gamma = vertical stretch
# Amp   = wrinkle amplitude
# Sp    = wrinkle shape
du    = TrialFunction(ME)
u_    = TestFunction(ME)
(ur_, lmda_, nc_, gamma_, Amp_, Sp_) = split(u_)

u   = Function(ME)  # current solution
u0  = Function(ME)  # solution from previous converged step
(ur, lmda, nc, gamma, Amp, Sp) = split(u)
(ur0, lmda0, nc0, gamma0, Amp0, Sp0) = split(u0)

#################################################






#################################################
### 4. Initial Conditions, Boundary conditions and Problem ###

# (+) initial conditions

class InitialConditions(Expression):
    def __init__(self, **kwargs):
        return
    def eval(self, values, x):
        values[0] = 0.0;
        values[1] = 1.0;
        values[2] = c0;
        values[3] = 1.0;
        values[4] = 0.0;
        values[5] = 0.0;
    def value_shape(self):
        return (6,)


# (+) boundary conditions

def left_boundary(x, on_boundary):
    return on_boundary and near(x[0],0.);
def edge_boundary(x, on_boundary):
    return on_boundary and near(x[0],Rc);

def assemble_boundary_conditions(comp):    
    # no radial displacement at r = 0
    bc_left  = DirichletBC(ME.sub(0), Constant(comp), left_boundary)
    # fixed concentration on the circle boundary
    bc_circle = DirichletBC(ME.sub(2), c0, edge_boundary)
    return [bc_left,  bc_circle]

bcs = assemble_boundary_conditions(0.)
            
            
            
# (+) problem
class GrowthProblem(NonlinearProblem):
    def __init__(self, L, a, bc):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs
        
    def F(self, b, x):
        assemble(self.L, tensor=b)
        for bc in self.bcs:
            bc.apply(b,x)
            
    def J(self, A, x):
        assemble(self.a, tensor=A)
        for bc in self.bcs:
            bc.apply(A)

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 4
#################################################







#################################################
### 5. Nutrient dependent growth and uptake   ###

def phi_c(xo):
    return xo/(xo+chalf)
    
def cal_qc(xo):
    phio = phi_c(xo)
    return Q0*phio

def cal_kc(xo):
    phio = phi_c(xo)
    return k_growth*(phio) + kr

#################################################







#################################################
### 6. Weak Form of the elasto-growth problem ###

# Define R0 (radial coordinate in the initial configuartion)
myR = Expression('x[0]',degree = 1)
myR_c = Expression('x[0] > 0 ? x[0] : eps',degree = 1, eps = 1E-12)

myr = myR + ur                       # r (current configuration)
Frr = 1 + grad(ur)[0]                # F
Ftt = 1 + ur/myR_c
Arr = Frr / lmda                     # Fe
Att = Ftt / lmda
Ja_tilde = Arr * Att
Jf = Frr * Ftt

Frr0 = 1 + grad(ur0)[0]
Ftt0 = 1 + ur0/myR_c
Arr0 = Frr0 / lmda0
Att0 = Ftt0 / lmda0


# (+) Direct implementation of the analytical results

Delta       = (gamma ** 2 - Att ** 2) - Eps_c
Delta12     = (Arr ** 2 - Att ** 2)
Delta_old   = (gamma0 ** 2 - Att0 ** 2) - Eps_c
Delta12_old = (Arr0 ** 2 - Att0 ** 2)

delta_c = 0.0

# \tilde{S}
Ssq = (Delta - 2.0*Delta12)/(Delta + Delta12)
Ssq_Expression  = conditional(ge(Delta_old - 2.0*Delta12_old,delta_c), Ssq ,0.0)
# \tilde{A}
Asq = (Delta - Delta12*Ssq_Expression/(1+Ssq_Expression))/(1 - Ssq_Expression/((1+Ssq_Expression)**2))
Asq_Expression = conditional(ge(Delta_old,delta_c), Asq ,0.0)

Asq_positive = conditional(ge(Asq_Expression,delta_c),Asq_Expression,0.0)
Ssq_positive = conditional(ge(Ssq_Expression,delta_c),Ssq_Expression,0.0)


# (+) Normalized cauchy stress
srr = (Arr ** 2 - gamma ** 2)  + Asq_positive*(0.5+1.0*Ssq_positive)/(1.0+Ssq_positive)
stt = (Att ** 2 - gamma ** 2)  + Asq_positive*(1.0+0.5*Ssq_positive)/(1.0+Ssq_positive)


kofc = cal_kc(nc)
qofc = cal_qc(nc)

# (+) Governing equations (Lagrangian):
# 1. force balance; 2. nutrient dependent growth; 3. nutrient diffusion; 4. plane-stress constraint; 5. winkling

f1 = (1+0.5*Asq_positive/gamma/gamma) * ksi * (ur- ur0) * myr * Frr * ur_  + dt * (gamma * stt * Frr * ur_  + gamma * srr * grad(ur_)[0] * myr)
L1 = f1*dx

f2 = (lmda - lmda0)*lmda_ - dt*kofc*lmda*lmda_
L2 = f2*myr*Frr*dx

f3 = (nc - nc0)*Frr*nc_  - grad(nc)[0] * (ur - ur0) * nc_ + dt*Dn*grad(nc)[0]*grad(nc_)[0]/Frr + dt*rho0/Ja_tilde*qofc*Frr*nc_
L3 = f3*myr*dx

f4 = (gamma*Ja_tilde - 1.0) * gamma_
L4 = f4*myr*Frr*dx

L5 = ((Amp - Asq_positive)*Amp_ + (Sp - Ssq_positive)*Sp_)*myr*Frr*dx

L_tot = L1 + L2 + L3 + L4 + L5
Jacob = derivative(L_tot, u, du)
problem = GrowthProblem(L_tot,Jacob,bcs)
solver = NewtonSolver()
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-8
solver.parameters["maximum_iterations"] = 500
#solver.parameters["monitor_convergence"] = False
solver.parameters["report"] = False


# (+) define output variable
# cauchy stress
srr_of = project(srr,V1)           # radial
stt_of = project(stt,V1)           # circumferential
frr_of = project(Frr,V1)
ftt_of = project(Ftt,V1)
vr = project((ur-ur0)/dt,V1)       # velocity

srr_of.rename("sigma_rr","")
stt_of.rename("sigma_tt","")

#################################################







#################################################
### 7. Solving the problem ###
print ("Program started:")

for ksi_val in ksi_list:
    ksi.my_ksi = ksi_val

    for kr_val in kr_list:
        kr.my_kr = kr_val

        for Nc_val in Eps_c_list:
            Eps_c.my_Nc = Nc_val
            
            # ====================================================================== #
            print("Preparing output file......")
            directorymain = myname + date + "/" + str(trial) +"/"
            if not os.path.exists(directorymain):
                os.makedirs(directorymain)
            filesum = open(directorymain+"sum_"+str(trial)+".txt",'w')
            fileinfo = open(directorymain+"info_"+str(trial)+".txt",'w')

            fileinfo.write(debug_text+'\n' + '\n')
            fileinfo.close()
            # ====================================================================== #
            
            
            # ====================================================================== #
            print("Initializing......")
            u_init = InitialConditions(degree = 0)
            u.interpolate(u_init)
            # ====================================================================== #
            
            
            
            # ====================================================================== #
            print ("Start growth...")
            t = 0.0
            t_of = dt_of
            start = time.time()

            while t < T :
                
                u0.vector()[:] = u.vector()
                solver.solve(problem, u.vector())
                
                (ur, lmda, nc, gamma, Amp, Sp) = u.split()
                (ur0, lmda0, nc0, gamma0, Amp0, Sp0) = u0.split()
                
                (ur_p, lmda_p, nc_p, gamma_p, Amp_p, Sp_p) = u.split(deepcopy = True)
                
                srr_of.vector()[:]   = project(srr,V1).vector()
                stt_of.vector()[:]   = project(stt,V1).vector()
                frr_of.vector()[:]   = project(Frr,V1).vector()
                ftt_of.vector()[:]   = project(Ftt,V1).vector()
                vr.vector()[:]       = project((ur-ur0)/dt,V1).vector()
                
                t += dt.dt
                print("t = " + str(round(t,2)) + "; max(A) = " + str(round(np.amax(Amp_p.vector().get_local()),2)) + "; max(S) = " + str(round(np.amax(Sp_p.vector().get_local()),2)) + ".\n")
                
                
                if t >= t_of:
                    t_of += dt_of
                    # write data to txt file
                    filesum.write(str(t) + ' ' + str(Rc + ur(Rc)) + ' ' + str(kr.my_kr) + ' ' + str(Eps_c.my_Nc) + ' ' + \
                                  str(0.0) + ' ' + str(0.0) + ' ' + str(0.0) + ' ' + str(0.0) + '\n')
                    for x_ in np.linspace(0,Rc,801):
                        filesum.write(str( x_  + ur(x_) )+ '  ' + str(nc(x_)) + '  ' + str(srr_of(x_))  + '  ' + str(stt_of(x_))  + '  ' + \
                                      str(Amp(x_)) + '  ' + str(Sp(x_)) + '  ' + str(vr(x_)) + ' ' + str(gamma(x_)) +'\n' )

            filesum.close()
            trial = trial + 1

end = time.time()
print("Completed! Total running time: " + str(round(end - start,2)) + " s.")
#################################################
