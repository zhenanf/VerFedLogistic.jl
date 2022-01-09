# VerFedLogistic.jl

`VerFedLogistic.jl` is a Julia package for solving the vertical federated multinomial logistic regression. We implement both synchornous and asynchornous mini-bath stochastic gradient descent methods. 


## Installation
To install, just call
```julia
Pkg.add("https://github.com/ZhenanFanUBC/VerFedLogistic.jl.git")
```

## Data
Experiments are run on Adult, which is downloaded from [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). You can also mannually download other data sets from the website, and call the function `read_libsvm` to load the data. 

## Examples
* Example for synchronous SGD is contained in `examples/syn_adult.jl`
* Example for asynchronous SGD is contain in `examples/asyn_adult.jl`. Note that for asynchronous SGD, we need to start Julia with multiple threads, i.e. 
```juila -t M```
, where `M` is equal to the number of clients. 

