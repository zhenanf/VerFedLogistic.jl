########################################################################
# Server for synchronous learning
########################################################################
mutable struct Server{T1<:Int64, T2<:Float64, T3<:Vector{T2}, T4<:Vector{T1}, T5<:Matrix{T2}, T6<:Array{T2, 3}, T7<:Vector{Client}} 
    Ytrain::T4                       # training label
    Ytest::T4                        # test label
    num_classes::T1                  # number of classes
    num_clients::T1                  # number of clients
    num_epoches::T1                  # number of epoches
    batch_size::T1                   # batch size
    learning_rate::T2                # learning rate
    clients::T7                      # set of clients
    b::T3                            # server model
    embeddings::T6                   # embeddings for batches
    train_embeddings::T6             # embeddings for all training data (used for final evaluation)
    test_embeddings::T6              # embeddings for all test data (used for final evaluation)
    batch::T4                        # mini-batch
    grads::T5                        # gradient information
    function Server(Ytrain::Vector{Int64}, Ytest::Vector{Int64}, config::Dict{String, Union{Int64, Float64, String}})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        num_epoches = config["num_epoches"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]
        clients = Vector{Client}(undef, num_clients)
        b = zeros(Float64, num_classes)
        embeddings = zeros(Float64, num_clients, num_classes, batch_size)
        train_embeddings = zeros(Float64, num_clients, num_classes, length(Ytrain))
        test_embeddings = zeros(Float64, num_clients, num_classes, length(Ytest))
        batch = zeros(Int64, batch_size)
        grads = zeros(Float64, num_classes, batch_size)
        new{Int64, Float64, Vector{Float64}, Vector{Int64}, Matrix{Float64}, Array{Float64, 3}, Vector{Client}}(Ytrain, Ytest, num_classes, num_clients, num_epoches, batch_size, learning_rate, clients, b, embeddings, train_embeddings, test_embeddings, batch, grads)
    end
end

# connect with client
function connect(s::Server, c::Client)
    s.clients[c.id] = c
end

# update batch 
function update_batch(s::Server, batch::Vector{Int64})
    s.batch .= batch
end

# send embeddings to server
function send_embedding(c::Client, s::Server; tag = "batch")
    if tag == "batch"
        Xbatch = c.Xtrain[:, c.batch]
        if typeof(c.W) <: Matrix{Float64}
            embedding = c.W * Xbatch
        else
            embedding, c.back = Zygote.pullback(()->c.W(Xbatch), params(c.W))
        end
    elseif tag == "training"
        if typeof(c.W) <: Matrix{Float64}
            embedding = c.W * c.Xtrain
        else
            embedding = c.W(c.Xtrain)
        end
    else
        if typeof(c.W) <: Matrix{Float64}
            embedding = c.W * c.Xtest
        else
            embedding = c.W(c.Xtest)
        end
    end
    update_embedding(s, c.id, embedding, tag=tag)
end

# update embedding
function update_embedding(s::Server, id::Int64, embedding::Matrix{Float64}; tag = "batch")
    if tag == "batch"
        s.embeddings[id,:,:] .= embedding
    elseif tag == "training"
        s.train_embeddings[id,:,:] .= embedding
    else
        s.test_embeddings[id,:,:] .= embedding
    end
end

# compute mini-batch gradient
function compute_mini_batch_gradient(s::Server)
    batch_size = s.batch_size
    num_classes = s.num_classes
    sum_embeddings = reshape( sum( s.embeddings, dims=1), num_classes, batch_size )
    loss = 0.0
    grads = zeros( num_classes, batch_size )
    # compute mini-batch gradient
    for i = 1:batch_size
        y = s.Ytrain[ s.batch[i] ]
        emb = sum_embeddings[:, i] + s.b
        pred = softmax(emb)
        loss += neg_log_loss(pred, y)
        grads[:, i] .= pred
        grads[y, i] -= 1.0
    end
    # update local gradient information 
    s.grads .= grads
    # send gradient information to clients
    for c in s.clients
        update_grads(c, grads)
    end
    # return mini-batch loss
    return loss / batch_size
end

# update server model b
function update_model(s::Server)
    bgrad = sum(s.grads, dims=2) ./ s.batch_size
    s.b .-= s.learning_rate * bgrad[:]
end


########################################################################
# Server for asynchronous learning
########################################################################
mutable struct AsynServer{T1<:Int64, T2<:Float64, T3<:Vector{T2}, T4<:Vector{T1}, T5<:SharedArray{T2, 3}, T6<:Vector{AsynClient}} 
    Ytrain::T4                       # training label
    Ytest::T4                        # test label
    num_classes::T1                  # number of classes
    num_clients::T1                  # number of clients
    learning_rate::T2                # learning rate
    clients::T6                      # set of clients
    b::T3                            # server model
    embeddings::T5                   # latest embeddings
    train_embeddings::T5             # embeddings for all training data (used for final evaluation)
    test_embeddings::T5              # embeddings for all test data (used for final evaluation)
    function AsynServer(Ytrain::Vector{Int64}, Ytest::Vector{Int64}, config::Dict{String, Union{Int64, Float64, String}})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        learning_rate = config["learning_rate"]
        clients = Vector{AsynClient}(undef, num_clients)
        b = zeros(Float64, num_classes)
        embeddings = SharedArray{Float64}(num_clients, num_classes, length(Ytrain))
        train_embeddings = SharedArray{Float64}(num_clients, num_classes, length(Ytrain))
        test_embeddings = SharedArray{Float64}(num_clients, num_classes, length(Ytest))
        new{Int64, Float64, Vector{Float64}, Vector{Int64}, SharedArray{Float64, 3}, Vector{AsynClient}}(Ytrain, Ytest, num_classes, num_clients, learning_rate, clients, b, embeddings, train_embeddings, test_embeddings)
    end
end

# connect with client
function connect(s::AsynServer, c::AsynClient)
    s.clients[c.id] = c
end

# send embeddings to server
function send_embedding(c::AsynClient, s::AsynServer; tag = "batch")
    if tag == "batch"
        num_data = length(s.Ytrain)
        if c.num_commu == 0
            batch = collect(1:num_data)
        else
            batch = sample(collect(1:num_data), c.batch_size, replace=false)
        end
        Xbatch = c.Xtrain[:, batch]
        if typeof(c.W) <: Matrix{Float64}
            embedding = c.W * Xbatch
        else
            embedding, c.back = Zygote.pullback(()->c.W(Xbatch), params(c.W))
        end
        update_embedding(s, c.id, embedding, batch)
        @printf "Client %i finish uploading embedding \n" c.id
        return batch
    elseif tag == "training"
        if typeof(c.W) <: Matrix{Float64}
            embedding = c.W * c.Xtrain
        else
            embedding = c.W(c.Xtrain)
        end
        s.train_embeddings[c.id,:,:] .= embedding
    else
        if typeof(c.W) <: Matrix{Float64}
            embedding = c.W * c.Xtest
        else
            embedding = c.W(c.Xtest)
        end
        s.test_embeddings[c.id,:,:] .= embedding
    end
end

# update embedding
function update_embedding(s::AsynServer, id::Int64, embedding::Matrix{Float64}, batch::Vector{Int64})
    s.embeddings[id,:,batch] .= embedding
end

# compute gradient
function send_gradient(s::AsynServer, id::Int64, batch::Vector{Int64})
    batch_size = length(batch)
    num_classes = s.num_classes
    sum_embeddings = reshape( sum( s.embeddings[:,:,batch], dims=1), num_classes, batch_size )
    loss = 0.0
    grads = zeros( num_classes, batch_size )
    # compute mini-batch gradient
    for i = 1:batch_size
        y = s.Ytrain[ batch[i] ]
        emb = sum_embeddings[:, i] + s.b
        pred = softmax(emb)
        loss += neg_log_loss(pred, y)
        grads[:, i] .= pred
        grads[y, i] -= 1.0
    end
    
    # send gradient information to client
    update_grads(s.clients[id], grads)

    # return mini-batch loss
    return loss / batch_size
end

# Compute training and test accuracy
function eval(s::Union{Server, AsynServer})
    train_size = length(s.Ytrain)
    test_size = length(s.Ytest)
    num_classes = s.num_classes
    sum_train_embeddings = reshape( sum( s.train_embeddings, dims=1), num_classes, train_size )
    sum_test_embeddings = reshape( sum( s.test_embeddings, dims=1), num_classes, test_size )
    train_loss = 0.0
    train_acc = 0.0
    test_loss = 0.0
    test_acc = 0.0
    for i = 1:train_size
        y = s.Ytrain[i]
        emb = sum_train_embeddings[:, i] + s.b
        pred = softmax(emb)
        train_loss += neg_log_loss(pred, y)
        if argmax(pred) == y
            train_acc += 1.0
        end
    end
    for i = 1:test_size
        y = s.Ytest[i]
        emb = sum_test_embeddings[:, i] + s.b
        pred = softmax(emb)
        test_loss += neg_log_loss(pred, y)
        if argmax(pred) == y
            test_acc += 1.0
        end
    end
    return train_loss/train_size, train_acc/train_size, test_loss/test_size, test_acc/test_size
end

