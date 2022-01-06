########################################################################
# Asynchronous Vertical Federated Logistic Regression for Adult dataset
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
config["learning_rate"] = 0.05
config["time_limit"] = 5.0


# vertically split data
Xtrain_split, Xtest_split = split_data(Xtrain, Xtest, config["num_clients"])

# initialize server 
server = AsynServer(Ytrain, Ytest, config)


# initialize clients
clients = Vector{AsynClient}(undef, config["num_clients"])
for id = 1:config["num_clients"]
    ts = 0.01
    # ts = 0.01
    c = AsynClient(id, Xtrain_split[id], Xtest_split[id], ts, config)
    clients[id] = c
    # connect with server
    connect(server, c)
end

# training
vertical_lr_train(server, clients, config["time_limit"])

# evaluation
evaluation(server, clients)



