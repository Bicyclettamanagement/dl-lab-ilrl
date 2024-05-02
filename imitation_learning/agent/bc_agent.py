import torch
import sys

from utils import id_to_action

sys.path.append("..")
from agent.networks import CNN


class BCAgent:

    def __init__(self, network=CNN(), config=None, checkpoint=''):
        # TODO: Define network, loss function, optimizer
        if network is None:
            network = CNN()
        else:
            self.net = network
        if config is None:
            config = {'lr': 0.0001, 'optimizer': 'adam', 'loss': 'crossentropy'}

        if config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['lr'])
        elif config['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=config['lr'])
        if config['loss'] == 'crossentropy':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        if checkpoint != '':
            self.load(checkpoint)

        pass

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize
        self.optimizer.zero_grad()
        outputs = self.net(X_batch)
        loss = self.loss_fn(outputs, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, X):
        # TODO: forward pass
        outputs = self.net(X)
        outputs = torch.argmax(outputs, dim=1)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
        return file_name
