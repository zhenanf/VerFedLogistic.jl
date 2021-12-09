########################################################################
# Asynchronous Vertical Federated Logistic Regression for Adult
########################################################################
using Printf
using SparseArrays

# load data
filename = "adult"
Xtrain, Ytrain, Xtest, Ytest = load_data(filename)

# config
config = Dict{String, Union{Int64, Float64}}()
config["num_classes"] = 2
config["num_clients"] = 3
config["batch_size"] = 2837
config["learning_rate"] = 0.2
config["time_limit"] = 3.0

# vertically split data
Xtrain_split, Xtest_split = split_data(Xtrain, Xtest, config["num_clients"])

# initialize server 
server = AsynServer(Ytrain, Ytest, config)

# initialize valuator
Δt = 0.1
valuator = AsynValuator(Ytrain, Δt, config)
# valuator = missing

# initialize clients
clients = Vector{AsynClient}(undef, config["num_clients"])
for id = 1:config["num_clients"]
    ts = (4 - id) * 0.01
    c = AsynClient(id, Xtrain_split[id], Xtest_split[id], ts, config)
    clients[id] = c
    # connect with server
    connect(server, c)
end

# training
vertical_lr_train(server, clients, config["time_limit"], valuator=valuator)

# evaluation
evaluation(server, clients, valuator=valuator)



