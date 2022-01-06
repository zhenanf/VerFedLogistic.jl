module VerFedLogistic

using LinearAlgebra
using Printf
using SparseArrays
using Random
using SharedArrays
using Distributed
using Combinatorics
using StatsBase

export Client, AsynClient
export Server, AsynServer
export connect
export update_batch
export send_embedding, update_embedding
export update_model, update_grads, compute_mini_batch_gradient, send_gradient
export eval
export softmax, neg_log_loss
export load_data, split_data, generate_batches
export vertical_lr_train, evaluation
export read_libsvm

include("./utils.jl")
include("./client.jl")
include("./server.jl")
include("./training.jl")



end # module