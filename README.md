# Neural Operators and Operator Networks vs Parametric Approach

## Usage

The default configuration is inside the `add_arguments` function inside ``config.py`

### prerequisite

1. if you are using windows,
   install `visual studio` to default location
2. download the `ninja` and add it to path

### Look at how the equations evolve over tim
the command below will generate the evolution video from all equations under different parameters
```bash
python gen_video.py
```
the result is inside `videos/` folder


### generate your own configuration
by running the command below, it will generate folders of toml file based on the marco in `config.py`
and the availability of GPU and the memory you passed.
```bash
python gen_config.py
```

### run the trainig script
the training script is automatically restore from the last interrupt state, just run 
```bash
python run_train.py
```
every time you are available

### run the plotting script
the  plotting  script is automatically restore from the last interrupt state, just run
```bash
python run_plot.py
```


### run all the training files
by running this command, it will run recursively all the configuration file under the config folder

```bash
python run_folder.py config
```

### run with single configuration file

the file and be `toml/json/yaml` file

you simply give it after the argument of `run_file.py`

for example, if you want to train the deeponet on heat equation

```bash
python run_file.py config/train/heat_d=1/deeponet.toml
```

### run with command line

for example, if you want to train the deeponet on heat equation

```bash
python main.py --task train --model deeponet --equation heat
```

## Task 

1. a standard feed forward neural network to approximate the function $g$

$$
g(x,y,\mu  ) = u(T, x, y, \mu  )
$$

2. an operator network (DeepONet) or Neural Operator to approximate the operator $\mathcal G$

$$
G(u_0)(x,y) = u(T,x,y)
$$

$$
G(u_0)(x,y) = u(T,x,y)
$$

## Equation

### Heat  Equation

#### PDE equation

$$
u_t = \nabla u\quad t \in  [0,T],(x_1,x_2)\in  [-1,1]^2,  \mu\in  [-1,1]^d
$$

$$
\mu\sim Unif([-1,1]^d)
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
u(t,\\\\\{-1,1\\\\\},\\\\\{-1,1\\\\\},\mu) = 0
$$

#### solution 

$$
u(t,x_1,x_2,\mu) = -\frac{1}{d}\sum_{m=1}^d \frac{\mu_m}{\sqrt{m}} e^{-2m^2\pi^2t} sin(\pi m  x_1)sin(\pi mx_2)
$$

### Wave Equation

##### PDE Equation

$$ u_{tt} - c^2 \Delta u = 0 (u_{tt} - c^2(u_{xx} + u_{yy})) \quad (x, y) \in [0, 1]^2, t \in [0, T], c = 0.1 $$   

##### Initial Condition

$$u(0, x, y, a) = \frac{\pi}{K^2} \sum_{i,j=1}^{K} a_{ij} \cdot (i^2 + j^2)^{-r} sin(\pi ix) sin(\pi jy) \quad \forall x,y \in [0, 1]$$

##### Solution

$$u(t, x, y, a) = \frac{\pi}{K^2} \sum_{i,j=1}^{K} a_{ij} \cdot (i^2 + j^2)^{-r} sin(\pi ix) sin(\pi jy) cos(c\pi t \sqrt{i^2 + j^2}), \forall x,y \in [0, 1]$$

##### PDE Equation

$$ u_{tt} - c^2 \Delta u = 0 (u_{tt} - c^2(u_{xx} + u_{yy})) \quad (x, y) \in [0, 1]^2, t \in [0, T], c = 0.1 $$   

##### Initial Condition

$$u(0, x, y, a) = \frac{\pi}{K^2} \sum_{i,j=1}^{K} a_{ij} \cdot (i^2 + j^2)^{-r} sin(\pi ix) sin(\pi jy) \quad \forall x,y \in [0, 1]$$

##### Solution

$$u(t, x, y, a) = \frac{\pi}{K^2} \sum_{i,j=1}^{K} a_{ij} \cdot (i^2 + j^2)^{-r} sin(\pi ix) sin(\pi jy) cos(c\pi t \sqrt{i^2 + j^2}), \forall x,y \in [0, 1]$$

### Poisson Equation

##### PDE Equation

$$ -\Delta u = -(u_{xx} + u_{yy}) = f \quad (x, y) \in [0, 1]^2 $$   

##### Boundary Condition

$$u\vert_{\partial D} = 0 $$

##### Solution

$$u(x, y) = \frac{1}{\pi\cdot K^2} \sum_{i,j=1}^{K} a_{ij} \cdot (i^2 + j^2)^{r-1} sin(\pi ix) sin(\pi jy),\quad \forall (x,y) \in D$$

with respect to source function

$$f=\frac{\pi}{K^2} \sum_{i,j=1}^{K} a_{ij} \cdot (i^2 + j^2)^{r} sin(\pi ix) sin(\pi jy),\quad \forall (x,y) \in D$$

## Model

### MLP

$$
H^{l+1} = \sigma(W H^l + b)
$$

### DeepONet

$$
\tilde u( T, x_1, x_2, \mu )_{jd} = 
\mathcal B \left( [ y_1, y_2, u ( 0, y_1, y_2, \mu ) ] \right) _ { idh } 
\mathcal T (  [ T, x_1, x_2 ] ) _ { jdh }
$$

### FNO

### CNO

