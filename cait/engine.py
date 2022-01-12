"""
Train and eval functions used in main.py
"""
import paddle


def train_one_epoch(model, criterion, data_loader, optimizer, epoch):

    model.train()
    for batch_id, data in enumerate(data_loader):
        samples, targets = data[0], data[1]
        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        acc = paddle.metric.accuracy(outputs, targets)
        if batch_id % 10 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
