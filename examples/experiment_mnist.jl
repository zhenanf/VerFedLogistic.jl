########################################################################
# Vertical Federated Logistic Regression for MNIST
########################################################################
using Printf
# load data
filename = "mnist"
Xtrain, Ytrain, Xtest, Ytest = load_data(filename)

# config
config = Dict{String, Union{Int64, Float64}}()
config["num_classes"] = 10
config["num_clients"] = 10
config["num_epoches"] = 25
config["batch_size"] = 6000
config["learning_rate"] = 0.8

# vertically split data
Xtrain_split, Xtest_split = split_data(Xtrain, Xtest, config["num_clients"])

# initialize server 
server = Server(Ytrain, Ytest, config)

# initialize clients
clients = Vector{Client}(undef, config["num_clients"])
for id = 1:config["num_clients"]
    c = Client(id, Xtrain_split[id], Xtest_split[id], config)
    clients[id] = c
    # connect with server
    connect(server, c)
end

# training
startT = time()
vertical_lr_train(server, clients)
endT = time()
@printf "training time: %.2f secs \n" endT - startT

# evaluation
evaluation(server, clients)

