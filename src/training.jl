########################################################################
# Synchronous Vertical Federated Logistic Regression
########################################################################

function vertical_lr_train(server::Server, clients::Vector{Client})
    # number of epoches
    num_epoches = server.num_epoches 
    # number of training data
    num_train_data = length(server.Ytrain)
    # batch size
    batch_size = server.batch_size
    # number of batches
    num_batches = div(num_train_data, batch_size)
    # round
    round = 1
    # start training
    @inbounds for epoch = 1:num_epoches
        # generate mini-batches
        batches = generate_batches(num_train_data, num_batches)
        @inbounds for i = 1:num_batches
            batch = batches[i]
            # server updates batch information
            update_batch(server, batch)
            Threads.@threads for c in clients
                # client updates batch information
                update_batch(c, batch)
                # client compute and upload embeddings
                send_embedding(c, server)
            end
            # server compute the loss and the gradient
            batch_loss = compute_mini_batch_gradient(server)
            if i % 1 == 0
                @printf "Epoch %d, Batch %d, Loss %.2f\n" epoch i batch_loss
            end
            # server and clients update model
            update_model(server)
            Threads.@threads for c in clients
                update_model(c)
            end
            # update round
            round += 1
        end
    end
end

function evaluation(server::Server, clients::Vector{Client})
    # test and train accuracy
    for c in clients
        # client compute and upload training embeddings
        send_embedding(c, server, tag = "training")
        # client compute and upload test embeddings
        send_embedding(c, server, tag = "test")
    end
    train_loss, train_acc, test_loss, test_acc = eval(server)
    @printf "Train Loss %.2f, Train Accuracy %.2f, Test Loss %.2f, Test Accuracy %.2f\n" train_loss train_acc test_loss test_acc
end


########################################################################
# Asynchronous Vertical Federated Logistic Regression
########################################################################

function vertical_lr_train(server::AsynServer, clients::Vector{AsynClient}, time_limit::Float64)
    tag = true
    # set time limit
    @async begin
        sleep(time_limit)
        tag = false
    end
    # start training
    Threads.@threads for c in clients
        while tag
            # client compute and send embedding to server
            batch = send_embedding(c, server)
            # server compute and send back the gradient
            send_gradient(server, c.id, batch)
            # client update model
            update_model(c, batch)
            # time break
            sleep(c.ts)
        end
    end
    @printf "Finish training after %.2f seconds\n" time_limit
    # print number of communication rounds
    for c in clients
        @printf "Client %i communicate %i times with server \n" c.id c.num_commu
    end
end

function evaluation(server::AsynServer, clients::Vector{AsynClient})
    # test and train accuracy
    for c in clients
        # client compute and upload training embeddings
        send_embedding(c, server, tag = "training")
        # client compute and upload test embeddings
        send_embedding(c, server, tag = "test")
    end
    train_loss, train_acc, test_loss, test_acc = eval(server)
    @printf "Train Loss %.2f, Train Accuracy %.2f, Test Loss %.2f, Test Accuracy %.2f\n" train_loss train_acc test_loss test_acc
end