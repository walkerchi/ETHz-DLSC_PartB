# Neural Operators and Operator Networks vs Parametric Approach

## Task 

1. a standard feed forward neural network to approximate the function $g$

$$
g(x,y,\mu) = u(T, x, y, \mu)
$$

2. an operator network (DeepONet) or Neural Operator to approximate the operator $\mathcal G$
   $$
    G(u_0)(x,y) = u(T,x,y)
   $$

## Equation

heat equation
$$
u_t = \nabla u\quad t \in[0,T],(x_1,x_2)\in[-1,1]^2,\mu\in[-1,1]^d
$$
$\mu\sim Unif([-1,1]^d)$

initial condition
$$
u(0,x_1,x_2,\mu) = -\frac{1}{d}\sum_{m=1}^d  \mu_m sin(\pi m x_1)sin(\pi m x_2)/\sqrt m
$$
boundary condition

zero Dirichlet boundary condition

solution
$$
u(x,t) = \frac{1}{(4\pi)^{n/2}}\int_{R^2}e^{-\frac{|x-y|^2}{4}}\left(-\frac{1}{d}\sum_{m=1}^d \mu_m sin(\pi my_1)sin(\pi my_2)\right)dy
$$

$$
\int_{-1}^1e^{\frac{(x-y)^2}{4}} sin(\pi m y)dy
$$
