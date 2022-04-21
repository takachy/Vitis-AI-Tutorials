import torch
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# To monitor the server's GPU installation and usage: log in the server and run `nvidia-smi`.
# It shows the list of GPUs online and their utilization.

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print("device = ", device)

epochs = 10
batch_size_train = 128
batch_size_test = 128
num_shown_images = 8

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)