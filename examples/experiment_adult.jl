########################################################################
# Vertical Federated Logistic Regression for Adult
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
config["num_epoches"] = 20
config["batch_size"] = 2837
config["learning_rate"] = 0.2

# vertically split data
Xtrain_split, Xtest_split = split_data(Xtrain, Xtest, config["num_clients"])

# create fake feature for an additional client
m, n = size(Xtrain_split[end]); Xtrain_fake = sprand(m, n, 0.1)
m, n = size(Xtest_split[end]); Xtest_fake = sprand(m, n, 0.1)
push!(Xtrain_split, Xtrain_fake); push!(Xtest_split, Xtest_fake); config["num_clients"] += 1

# create for an additional client with same feature as some client
Xtrain_fake = Xtrain_split[1]
Xtest_fake = Xtest_split[1]
push!(Xtrain_split, Xtrain_fake); push!(Xtest_split, Xtest_fake); config["num_clients"] += 1

# initialize server 
server = Server(Ytrain, Ytest, config)

# initialize valuator
valuator = Valuator(Ytrain, config)
# valuator = missing

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
vertical_lr_train(server, clients, valuator=valuator)
endT = time()
@printf "training time: %.2f secs \n" endT - startT

# evaluation
evaluation(server, clients, valuator=valuator)

