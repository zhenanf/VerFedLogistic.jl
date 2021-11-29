module VerFedLogistic

using LinearAlgebra
using Printf
using SparseArrays
using Random
using Distributed
using HDF5


export Client
export Server
export connect
export update_batch
export send_embedding, update_embedding
export update_model, update_grads, compute_mini_batch_gradient
export eval
export softmax, neg_log_loss
export load_data, split_data, generate_batches
export vertical_lr_train, evaluation

include("src/client.jl")
include("src/server.jl")
include("src/utils.jl")
include("src/training.jl")



end # module