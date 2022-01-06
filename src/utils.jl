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
function split_data(Xtrain::SparseMatrixCSC{Float64, Int64}, Xtest::SparseMatrixCSC{Float64, Int64}, num_clients::Int64)
    num_features = size(Xtrain, 1)
    num_features_client = div(num_features, num_clients)
    Xtrain_split = Vector{ SparseMatrixCSC{Float64, Int64} }(undef, num_clients)
    Xtest_split = Vector{ SparseMatrixCSC{Float64, Int64} }(undef, num_clients)
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
    if filename == "adult"
        Xtrain, Ytrain = read_libsvm("./data/Adult/a8a"); Xtrain = Xtrain[1:end-1, :]
        Xtest, Ytest = read_libsvm("./data/Adult/a8a.t")
        return Xtrain, Ytrain, Xtest, Ytest
    else
        @printf "Unsupported filename"
    end
end

# read data from libsvm
function read_libsvm(filename::String)
    numLine = 0
    nnz = 0
    open(filename, "r") do f
        while !eof(f)
            line = readline(f)
            info = split(line, " ")
            numLine += 1
            nnz += ( length(info)-1 )
            if line[end] == ' '
                nnz -= 1
            end
        end
    end
    @printf("number of lines: %i\n", numLine)
    n = numLine
    m = 0
    I = zeros(Int64, nnz)
    J = zeros(Int64, nnz)
    V = zeros(Float64, nnz)
    y = zeros(Int64, n)
    numLine = 0
    cc = 1
    open(filename, "r") do f
        while !eof(f)
            numLine += 1
            line = readline(f)
            info = split(line, " ")
            value = parse(Int64, info[1] )
            if value < 0
                value = Int64(2)
            end
            y[numLine] = value
            ll = length(info)
            if line[end] == ' '
                ll -= 1
            end
            for i = 2:ll
                idx, value = split(info[i], ":")
                idx = parse(Int, idx)
                value = parse(Float64, value)
                I[cc] = numLine
                J[cc] = idx
                V[cc] = value
                cc += 1
                m = max(m, idx)
            end
        end
    end
    return sparse( J, I, V, m, n ), y
end



