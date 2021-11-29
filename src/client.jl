########################################################################
# Client
########################################################################

mutable struct Client
    id::Int64
    Xtrain::Matrix{Float64}
    Xtest::Matrix{Float64}
    num_classes::Int64
    num_clients::Int64
    num_epoches::Int64
    batch_size::Int64
    learning_rate::Float64
    W::Matrix{Float64}
    batch::Vector{Int64}
    grads::Matrix{Float64}
    function Client(id::Int64, Xtrain::Matrix{Float64}, Xtest::Matrix{Float64}, config::Dict{String, Union{Int64, Float64}})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        num_epoches = config["num_epoches"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]
        dm, num_data = size(Xtrain)
        W = rand(num_classes, dm)
        batch = zeros(Int64, batch_size)
        grads = zeros(Float64, num_classes, batch_size)
        new(id, Xtrain, Xtest, num_classes, num_clients, num_epoches, batch_size, learning_rate, W, batch, grads)
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
