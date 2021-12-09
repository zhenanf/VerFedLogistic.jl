########################################################################
# Synchronous Valuator
########################################################################

mutable struct Valuator{T1<:Int64, T2<:Vector{SparseMatrixCSC{Float64, Int64}}, T3<:Vector{Vector{Int64}}, T4<:Vector{Int64}}  
    Ytrain::T4
    num_classes::T1
    num_clients::T1
    num_epoches::T1
    batch_size::T1
    embedding_matrices::T2
    all_subsets::T3
    function Valuator(Ytrain::Vector{Int64}, config::Dict{String, Union{Int64, Float64}})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        num_epoches = config["num_epoches"]
        batch_size = config["batch_size"]
        embedding_matrices = Vector{SparseMatrixCSC{Float64, Int64}}()
        N = length(Ytrain); num_batches = div(N, batch_size); T = num_epoches * num_batches * num_classes
        for i = 1:num_clients
            push!(embedding_matrices, spzeros(T, N))
        end
        all_subsets = collect( powerset(collect(1:num_clients), 1) )
        new{Int64, Vector{SparseMatrixCSC{Float64, Int64}}, Vector{Vector{Int64}}, Vector{Int64}}(Ytrain, num_classes, num_clients, num_epoches, batch_size, embedding_matrices, all_subsets)
    end
end

# server send embeddings to valuator
function send_embedding(s::Server, v::Valuator, round::Int64)
    num_clients = s.num_clients
    num_classes = s.num_classes
    batch = s.batch
    batch_size = s.batch_size
    for i = 1:num_clients
        v.embedding_matrices[i][ (round-1)*num_classes+1 : round*num_classes, s.batch] = reshape(s.embeddings[i,:,:], num_classes, batch_size)
    end
end

# complete the embedding matrices
function complete_embedding_matrices(v::Valuator, r::Int64)
    num_clients = v.num_clients
    Factors = Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}(undef, num_clients)
    Threads.@threads for i=1:num_clients
        @printf "start completing embedding matrix for client %i \n" i
        X, Y = complete_matrix(v.embedding_matrices[i], r)
        Factors[i] = (X,Y)
    end
    return Factors
end

# roundly utility function
function Uₜ(v::Valuator, S::Vector{Int64}, t::Int64, Factors::Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}, b::Vector{Float64})
    train_size = length(v.Ytrain)
    num_classes = v.num_classes
    Sc = setdiff( collect(1:v.num_clients), S )
    last = ((t-2)*num_classes+1) : (t-1)*num_classes
    current = ((t-1)*num_classes+1) : t*num_classes
    sum_embeddings_last = zeros(num_classes, train_size)
    sum_embeddings_current = zeros(num_classes, train_size)
    for i in Sc
        (X, Y) = Factors[i]
        sum_embeddings_last .+= X[last, :] * Y
        sum_embeddings_current .+= X[last, :] * Y
    end
    for i in S
        (X, Y) = Factors[i]
        sum_embeddings_last .+= X[last, :] * Y
        sum_embeddings_current .+= X[current, :] * Y
    end
    train_loss_last = 0.0
    train_loss_current = 0.0
    for i = 1:train_size
        y = v.Ytrain[i]
        emb_last = sum_embeddings_last[:, i] + b
        emb_current = sum_embeddings_current[:, i] + b
        pred_last = softmax(emb_last)
        pred_current = softmax(emb_current)
        train_loss_last += neg_log_loss(pred_last, y)
        train_loss_current += neg_log_loss(pred_current, y)
    end
    return (train_loss_last - train_loss_current) / train_size
end

# utility function
function utility(v::Valuator, S::Vector{Int64}, Factors::Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}, b::Vector{Float64})
    num_epoches = v.num_epoches
    batch_size = v.batch_size
    train_size = length(v.Ytrain)
    num_batches = div(train_size, batch_size)
    T = num_epoches * num_batches
    u = 0.0
    for t = 2:T
        u += Uₜ(v, S, t, Factors, b)
    end
    return u
end

# compute shapley value
function compute_shapley_value(v::Valuator, Factors::Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}, b::Vector{Float64})
    num_clients = v.num_clients 
    @printf "start computing all utilities \n"
    all_utilities = Dict{Vector{Int64},Float64}()
    Threads.@threads for S in v.all_subsets
        all_utilities[S] = utility(v, S, Factors, b)
    end
    shapley_values = Vector{Float64}()
    for i in 1:num_clients
        @printf "start computing shapley value for client %i \n" i
        # power set of [M] \ {i}
        all_subsets_i = collect( powerset(setdiff(collect(1:num_clients), [i]), 1) )
        si = 0.0
        for S in all_subsets_i
            c = 1 / binomial(num_clients - 1, length(S))
            # U(S ∪ i)
            u1 = all_utilities[ sort(vcat(S, [i])) ]
            # U(S)
            u2 = all_utilities[ S ]
            si += c*(u1 - u2)
        end
        push!(shapley_values, si)
    end
    return shapley_values
end

########################################################################
# Asynchronous Valuator
########################################################################
mutable struct AsynValuator{T1<:Int64, T2<:Float64, T3<:Vector{T1}, T4<:Array{T2, 4}, T5<:Vector{T3}}  
    Ytrain::T3
    num_classes::T1
    num_clients::T1
    embeddings_all_rounds::T4
    all_subsets::T5
    Δt::T2
    current_round::T1
    function AsynValuator(Ytrain::Vector{Int64}, Δt::Float64, config::Dict{String, Union{Int64, Float64}})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        time_limit = config["time_limit"]
        num_rounds = Int64(time_limit / Δt)
        embeddings_all_rounds = zeros(Float64, num_rounds, num_clients, num_classes, length(Ytrain))
        all_subsets = collect( powerset(collect(1:num_clients), 1) )
        current_round = 1
        new{Int64, Float64, Vector{Int64}, Array{Float64, 4}, Vector{Vector{Int64}}}(Ytrain, num_classes, num_clients, embeddings_all_rounds, all_subsets, Δt, current_round)
    end
end

# server send embeddings to valuator
function send_embedding(s::AsynServer, v::AsynValuator)
    v.embeddings_all_rounds[v.current_round, :, :, :] .= s.embeddings
    v.current_round += 1
end

# roundly utility function
function Uₜ(v::AsynValuator, S::Vector{Int64}, t::Int64, b::Vector{Float64})
    train_size = length(v.Ytrain)
    num_classes = v.num_classes
    Sc = setdiff( collect(1:v.num_clients), S )
    last = ((t-2)*num_classes+1) : (t-1)*num_classes
    current = ((t-1)*num_classes+1) : t*num_classes
    sum_embeddings_last = reshape( sum( v.embeddings_all_rounds[t-1, :, :, :], dims=1), num_classes, train_size)
    sum_embeddings_current = reshape( sum( v.embeddings_all_rounds[t-1, Sc, :, :], dims=1), num_classes, train_size) + reshape( sum( v.embeddings_all_rounds[t, S, :, :], dims=1), num_classes, train_size)
    train_loss_last = 0.0
    train_loss_current = 0.0
    for i = 1:train_size
        y = v.Ytrain[i]
        emb_last = sum_embeddings_last[:, i] + b
        emb_current = sum_embeddings_current[:, i] + b
        pred_last = softmax(emb_last)
        pred_current = softmax(emb_current)
        train_loss_last += neg_log_loss(pred_last, y)
        train_loss_current += neg_log_loss(pred_current, y)
    end
    return (train_loss_last - train_loss_current) / train_size
end

# utility function
function utility(v::AsynValuator, S::Vector{Int64}, b::Vector{Float64})
    T = size(v.embeddings_all_rounds, 1)
    u = 0.0
    for t = 2:T
        u += Uₜ(v, S, t, b)
    end
    return u
end

# compute shapley value
function compute_shapley_value(v::AsynValuator, b::Vector{Float64})
    num_clients = v.num_clients 
    @printf "start computing all utilities \n"
    all_utilities = Dict{Vector{Int64},Float64}()
    Threads.@threads for S in v.all_subsets
        all_utilities[S] = utility(v, S, b)
    end
    shapley_values = Vector{Float64}()
    for i in 1:num_clients
        @printf "start computing shapley value for client %i \n" i
        # power set of [M] \ {i}
        all_subsets_i = collect( powerset(setdiff(collect(1:num_clients), [i]), 1) )
        si = 0.0
        for S in all_subsets_i
            c = 1 / binomial(num_clients - 1, length(S))
            # U(S ∪ i)
            u1 = all_utilities[ sort(vcat(S, [i])) ]
            # U(S)
            u2 = all_utilities[ S ]
            si += c*(u1 - u2)
        end
        push!(shapley_values, si)
    end
    return shapley_values
end