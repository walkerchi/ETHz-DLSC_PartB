# Neural Operators and Operator Networks vs Parametric Approach

## Task 

1. a standard feed forward neural network to approximate the function $g$

$$
g(x,y,\mu ) = u(T, x, y, \mu )
$$

2. an operator network (DeepONet) or Neural Operator to approximate the operator $\mathcal G$

$$
G(u_0)(x,y) = u(T,x,y)
$$

## Equation

### Heat  Equation

#### PDE equation

$$
u_t = \nabla u\quad t \in [0,T],(x_1,x_2)\in [-1,1]^2, \mu\in [-1,1]^d
$$

$$
\mu\sim Unif([-1,1]^d)
$$

#### initial condition

$$
u(0,x_1,x_2,\mu) = -\frac{1}{d}\sum_{m=1}^d  \mu_m sin(\pi m x_1)sin(\pi m x_2)/\sqrt m
$$
#### boundary condition

$$
u(t,\\\{-1,1\\\},\\\{-1,1\\\},\mu) = 0
$$

#### solution 

$$
u(t,x_1,x_2,\mu) = -\frac{1}{d}\sum_{m=1}^d \frac{\mu_m}{\sqrt{m}} e^{-2m^2\pi^2t} sin(\pi m  x_1)sin(\pi mx_2)
$$

### Wave Equation

### Poisson Equation

## Model

### MLP

### DeepONet

### FNO

### CNO

