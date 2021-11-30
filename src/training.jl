########################################################################
# Vertical Federated Logistic Regression
########################################################################

function vertical_lr_train(server::Server, clients::Vector{Union{Missing, Client}})
    num_epoches = server.num_epoches
    num_train_data = length(server.Ytrain)
    batch_size = server.batch_size
    num_batches = div(num_train_data, batch_size)
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
            if i % 100 == 0
                @printf "Epoch %d, Batch %d, Loss %.2f\n" epoch i batch_loss
            end
            # server and clients update model
            update_model(server)
            Threads.@threads for c in clients
                update_model(c)
            end
        end
    end
end

function evaluation(server::Server, clients::Vector{Union{Missing, Client}})
    for c in clients
        # client compute and upload training embeddings
        send_embedding(c, server, tag = "training")
        # client compute and upload test embeddings
        send_embedding(c, server, tag = "test")
    end
    train_loss, train_acc, test_loss, test_acc = eval(server)
    @printf "Train Loss %.2f, Train Accuracy %.2f, Test Loss %.2f, Test Accuracy %.2f\n" train_loss train_acc test_loss test_acc
end