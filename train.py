import torch
import numpy as np
import click
import torch.utils.data as data
import torchvision.transforms as transforms
from discriminator import get_discriminator_model, get_ADM_model, WVEtoLVP
from keras.datasets import cifar10
import dnnlib

# TODO: Create training run according to the paper


def BCE_loss_fun(temporal_weight=None):
    """

    Args:
        temporal_weight: Function that takes t as input as returns the corresponding weight

    Returns: Weighted BCE loss function

    """
    def loss_fun(pred, labels, t=None):
        """

        Args:
            pred: B vector of model predictions
            labels: B vector of true labels
            t: B vector of diffusion times used to calculate the weight

        Returns: The BCE loss weighted with the temporal weight

        """
        if temporal_weight is None or t is None:
            loss = (torch.matmul(labels, torch.log(pred)) + torch.matmul(1 - labels, torch.log(1 - pred)))
        else:
            loss = torch.mul(temporal_weight(t), (torch.matmul(labels, torch.log(pred)) +
                                                  torch.matmul(1 - labels, torch.log(1 - pred))))
        return -torch.mean(loss)
    return loss_fun


class CustomDataset(data.Dataset):
    def __init__(self, data, labels, cond=None):
        self.data = data
        self.labels = labels
        self.cond = cond

    def __getitem__(self, item):
        if self.cond is None:
            return self.data[item], self.labels[item]
        else:
            return self.data[item], self.labels[item], self.cond[item]

    def __len__(self):
        return len(self.data)


@click.command()
@click.option('--sample_dir',                  help='Save directory',         metavar='PATH',    type=str, required=True,     default="training_data/conditional_edm_samples/edm_cond_samples.npz")
@click.option('--cond',                        help='Is it conditional?',     metavar='BOOL',    type=click.IntRange(min=0),  default=0)
@click.option('--batch_size',                  help='Batch size',             metavar='INT',     type=click.IntRange(min=1),  default=128)
@click.option('--n_epochs',                    help='Num epochs',             metavar='INT',     type=click.IntRange(min=1),  default=50)
@click.option('--lr',                          help='Learning rate',          metavar='FLOAT',   type=click.FloatRange(min=0),default=3e-4)
@click.option('--wd',                          help='Weight decay',           metavar='FLOAT',   type=click.FloatRange(min=0),default=0)
def main(**kwargs):
    # Load the arguments
    opts = dnnlib.EasyDict(kwargs)

    # Load images and discriminator labels
    true_data, true_cond = cifar10.load_data()[0]
    true_data = torch.from_numpy(true_data)

    fake_data = torch.from_numpy(np.load(opts.sample_dir)['images'])
    training_data = torch.concatenate((true_data, fake_data))
    training_lbl = torch.concatenate((torch.ones(true_data.shape[0]), torch.zeros(fake_data.shape[0])))

    # If conditioned on class, load those aswell
    if opts.cond:
        true_cond = np.eye(10)[true_cond.squeeze()]
        fake_cond = np.load(opts.sample_dir)['labels']
        training_cond = torch.from_numpy(np.concatenate((true_cond, fake_cond)))
        # Create a data set that returns each item
        dataset = CustomDataset(training_data, training_lbl, training_cond)
    else:
        dataset = CustomDataset(training_data, training_lbl)

    # Set up training and custom loss function.
    loss_fun = BCE_loss_fun()
    adm_classifier = get_ADM_model()
    discriminator = get_discriminator_model(opts.cond)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=opts.lr, weight_decay=opts.wd)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(opts.n_epochs):
        for samples in data_loader:
            # Reset grads
            optimizer.zero_grad()

            # Load samples into device memory
            if opts.cond:
                image, label, cond = samples
                cond = cond.to(device)
            else:
                image, label = samples
            image = image.to(device)
            label = label.to(device)

            # TODO: Implement Loss calculations here!
            importance_sampling = WVEtoLVP()
            # We can do uniform sampling as well, see page 24 and 15 it took me a while to understand
            # approximately what's going on in the derivations.

if __name__ == "__main__":
    main()
