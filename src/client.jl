########################################################################
# Client
########################################################################

mutable struct Client{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Matrix{Float64}, T5<:Vector{Int64}}
    id::T1                                  # client index
    Xtrain::T3                              # training data
    Xtest::T3                               # test data
    num_classes::T1                         # number of classes
    num_clients::T1                         # number of clients
    num_epoches::T1                         # number of epoches
    batch_size::T1                          # number of batches
    learning_rate::T2                       # learning rate
    W::T4                                   # client model
    batch::T5                               # mini-batch
    grads::T4                               # gradient information
    function Client(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Xtest::SparseMatrixCSC{Float64, Int64}, config::Dict{String, Union{Int64, Float64}})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        num_epoches = config["num_epoches"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]
        dm = size(Xtrain, 1)
        W = zeros(Float64, num_classes, dm)
        batch = zeros(Int64, batch_size)
        grads = zeros(Float64, num_classes, batch_size)
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Matrix{Float64}, Vector{Int64}}(id, Xtrain, Xtest, num_classes, num_clients, num_epoches, batch_size, learning_rate, W, batch, grads)
    end
end

# update batch 
function update_batch(c::Client, batch::Vector{Int64})
    c.batch .= batch
end

# update gradient information
function update_grads(c::Client, grads::Matrix{Float64})
    c.grads .= grads
end

# update client model W
function update_model(c::Client)
    Xbatch = c.Xtrain[:, c.batch]
    Wgrad = (c.grads * Xbatch') ./ c.batch_size
    c.W .-= c.learning_rate * Wgrad 
end
