########################################################################
# Client for synchronous learning
########################################################################
mutable struct Client{T1<:Int64, T2<:Float64, T3<:Vector{T1}, T4<:Matrix{T2}, T5<:SparseMatrixCSC{T2, T1}, T6<:Flux.Chain}
    id::T1                                  # client index
    Xtrain::T5                              # training data
    Xtest::T5                               # test data
    num_classes::T1                         # number of classes
    num_clients::T1                         # number of clients
    num_epoches::T1                         # number of epoches
    batch_size::T1                          # number of batches
    learning_rate::T2                       # learning rate
    W::Union{T4, T6}                        # client model
    batch::T3                               # mini-batch
    grads::T4                               # gradient information
    back                                    # backward operator
    function Client(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Xtest::SparseMatrixCSC{Float64, Int64}, config::Dict{String, Union{Int64, Float64, String}})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        num_epoches = config["num_epoches"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]
        dm = size(Xtrain, 1)
        if config["local_model"] == "linear"
            W = zeros(Float64, num_classes, dm)
        elseif config["local_model"] == "mlp"
            W = Chain(Dense(dm, 32, σ), Dense(32, num_classes))
        end
        back = undef
        batch = zeros(Int64, batch_size)
        grads = zeros(Float64, num_classes, batch_size)
        new{Int64, Float64, Vector{Int64}, Matrix{Float64}, SparseMatrixCSC{Float64, Int64},Flux.Chain}(id, Xtrain, Xtest, num_classes, num_clients, num_epoches, batch_size, learning_rate, W, batch, grads, back)
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
    if typeof(c.W) <: Matrix{Float64}
        Xbatch = c.Xtrain[:, c.batch]
        Wgrad = (c.grads * Xbatch') ./ c.batch_size
        c.W .-= c.learning_rate * Wgrad
    else
        gs = c.back(c.grads)
        w = params(c.W)
        for i = 1:length(w)
            w[i] .-= (c.learning_rate / c.batch_size) * gs[w[i]]
        end
    end  
end

########################################################################
# Client for asynchronous learning
########################################################################
mutable struct AsynClient{T1<:Int64, T2<:Float64, T3<:Matrix{T2}, T4<:SparseMatrixCSC{T2, T1}, T5<:Flux.Chain}
    id::T1                                  # client index
    Xtrain::T4                              # training data
    Xtest::T4                               # test data
    num_classes::T1                         # number of classes
    num_clients::T1                         # number of clients
    batch_size::T1                          # number of batches
    learning_rate::T2                       # learning rate
    W::Union{T3, T5}                        # client model
    grads::T3                               # gradient information
    ts::T2                                  # time gap between successive communications
    num_commu::T1                           # number of communication rounds
    back                                    # backward operator
    function AsynClient(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Xtest::SparseMatrixCSC{Float64, Int64}, ts::Float64, config::Dict{String, Union{Int64, Float64, String}})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]
        dm = size(Xtrain, 1)
        if config["local_model"] == "linear"
            W = zeros(Float64, num_classes, dm)
        elseif config["local_model"] == "mlp"
            W = Chain(Dense(dm, 32, σ), Dense(32, num_classes))
        end
        back = undef
        grads = zeros(Float64, num_classes, batch_size )
        num_commu = 0
        new{Int64, Float64, Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Flux.Chain}(id, Xtrain, Xtest, num_classes, num_clients, batch_size, learning_rate, W, grads, ts, num_commu, back)
    end
end

function update_grads(c::AsynClient, grads::Matrix{Float64})
    @printf "Client %i finish downloading gradient \n" c.id
    c.num_commu += 1
    c.grads = grads
end

function update_model(c::AsynClient, batch::Vector{Int64})
    if typeof(c.W) <: Matrix{Float64}
        Xbatch = c.Xtrain[:, batch]
        Wgrad = (c.grads * Xbatch') ./ c.batch_size
        c.W .-= c.learning_rate * Wgrad
    else
        gs = c.back(c.grads)
        w = params(c.W)
        for i = 1:length(w)
            w[i] .-= (c.learning_rate / c.batch_size) * gs[w[i]]
        end
    end   
end

