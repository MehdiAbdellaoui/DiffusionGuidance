import torch
import numpy as np
import click
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from discriminator import get_discriminator_model, get_ADM_model, WVEtoLVP, load_discriminator
from keras.datasets import cifar10
import dnnlib


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
@click.option('--sample_dir',                  help='Sample directory',       metavar='PATH',    type=str, required=True,     default="training_data/conditional_edm_samples/edm_cond_samples.npz")
@click.option('--cond',                        help='Is it conditional?',     metavar='BOOL',    type=click.IntRange(min=0),  default=1)
@click.option('--batch_size',                  help='Batch size',             metavar='INT',     type=click.IntRange(min=1),  default=128)
@click.option('--n_epochs',                    help='Num epochs',             metavar='INT',     type=click.IntRange(min=1),  default=60)
@click.option('--lr',                          help='Learning rate',          metavar='FLOAT',   type=click.FloatRange(min=0),default=3e-4)
@click.option('--wd',                          help='Weight decay',           metavar='FLOAT',   type=click.FloatRange(min=0),default=1e-7)
def main(**kwargs):
    # Load the arguments
    opts = dnnlib.EasyDict(kwargs)

    # Load images and discriminator labels
    true_data, true_cond = cifar10.load_data()[0]
    true_data = torch.from_numpy(true_data)

    fake_data = torch.from_numpy(np.load(opts.sample_dir)['images'])
    training_data = torch.concatenate((true_data, fake_data)).permute(0, 3, 1, 2)
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set up training and custom loss function.
    loss_fun = torch.nn.BCELoss()
    scaler = lambda x: 2. * x - 1
    adm_feature_extraction = get_ADM_model().to(device)
    discriminator = get_discriminator_model(opts.cond).to(device)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=opts.lr, weight_decay=opts.wd)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)

    importance_sampling = WVEtoLVP()

    plot_accuracy = []
    plot_loss = []
    plot_epochs = []
    for i in range(opts.n_epochs):
        batch_loss = []
        batch_accuracy = []
        for j, samples in enumerate(data_loader):
            # Reset grads
            optimizer.zero_grad()

            # Load samples into device memory
            if opts.cond:
                image, labels, cond = samples
                cond = cond.to(device)
            else:
                cond = None
                image, labels = samples
            image = image.to(device).float()
            image = scaler(image)
            labels = labels.to(device)
            n_samples = labels.shape[0]

            # We can do uniform sampling as well, see page 24 and 15 for importance sampling
            # Get times via importance sampling
            t = importance_sampling.generate_diffusion_times(n_samples, device)
            mean, std = importance_sampling.marginal_prob(t)

            # Generate noise
            e = torch.randn_like(image)
            # Apply noise by expanding the mean and std values to fit with the image and noise
            noisy_images = mean[:, None, None, None] * image + std[:, None, None, None] * e

            with torch.no_grad():
                features = adm_feature_extraction(noisy_images, t, cond, features=True, sigmoid=False)
            discriminator_output = discriminator(features, t, cond, features=False, sigmoid=True)[:, 0]

            loss = loss_fun(discriminator_output, labels)
            loss.backward()
            optimizer.step()
            accuracy = ((discriminator_output >= 0.5).float() == labels).float().mean()

            batch_loss.append(loss.item())
            batch_accuracy.append(accuracy.item())
            # Print every 20 batches
            if j % 20 == 0:
                print('                                                                                    ', end='\r')
                print(f'Epoch {i}: Loss: {np.mean(batch_loss)}, Accuracy: {np.mean(batch_accuracy)}', end='\r')
        # Save every epoch if anything happens
        torch.save(discriminator.state_dict(), f"models/discrim_uncond_epoch{i}.pt")

        # For plotting the training
        plot_loss.append(np.mean(batch_loss))
        plot_accuracy.append(np.mean(batch_accuracy))
        plot_epochs.append(i)

    # Plotting training stats
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Mean Loss (BCE)')
    ax1.plot(plot_epochs, plot_loss, label='Mean Loss (BCE)')
    ax1.legend()

    color = 'tab:orange'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean Accuracy')
    ax2.plot(plot_epochs, plot_accuracy, label='Mean Accuracy', color=color)
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    main()
