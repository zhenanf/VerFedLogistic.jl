########################################################################
# Helper Functions
########################################################################

# softmax function
function softmax(z::Vector{Float64})
    expz = exp.(z)
    s = sum(expz)
    return expz ./ s
end

# negative log-likelihood function
function neg_log_loss(z::Vector{Float64}, y::Int64)
    return -log(z[y])
end

# generate random batches
function generate_batches(num_data::Int64, num_batches::Int64)
    n,r = divrem(num_data, num_batches)
    b = collect(1:n:num_data+1)
    for i in 1:length(b)
        b[i] += i > r ? r : i-1  
    end
    p = randperm(num_data)
    return [p[r] for r in [b[i]:b[i+1]-1 for i=1:num_batches]]
end

# vertically split data
function split_data(Xtrain, Xtest, num_clients)
    num_features = size(Xtrain, 1)
    num_features_client = div(num_features, num_clients)
    Xtrain_split = Vector{ Union{Missing, Matrix{Float64}} }(missing, num_clients)
    Xtest_split = Vector{ Union{Missing, Matrix{Float64}} }(missing, num_clients)
    t = 1
    for i = 1:num_clients
        if i < num_clients
            ids = collect(t: t+num_features_client-1)
        else
            ids = collect(t: num_features)
        end
        Xtrain_split[i] = Xtrain[ids, :]
        Xtest_split[i] = Xtest[ids, :]
        t += num_features_client
    end
    return Xtrain_split, Xtest_split
end

# load data 
function load_data(filename::String)
    if filename == "mnist"
        fid = h5open("./data/MNISTdata.hdf5", "r")
        data = read(fid)
        close(fid)
        Xtrain = convert(Matrix{Float64}, data["x_train"])
        Ytrain = convert(Matrix{Int64}, data["y_train"]); Ytrain = Ytrain[:]; Ytrain .+= 1
        Xtest = convert(Matrix{Float64}, data["x_test"])
        Ytest = convert(Matrix{Int64}, data["y_test"]); Ytest = Ytest[:]; Ytest .+= 1
        return Xtrain, Ytrain, Xtest, Ytest
    else
        @printf "Unsupported filename"
    end
end