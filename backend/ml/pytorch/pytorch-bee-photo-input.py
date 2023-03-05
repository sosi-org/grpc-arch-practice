
MODEL_PATH = './my-cifar_net.pth'


def imshow(img):
    import matplotlib.pyplot as plt
    import numpy as np
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_torch_image(images):
    img_rgb = torchvision.utils.make_grid(images)
    imshow(img_rgb)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_classes():
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # classes = load_dict(...)
    return classes


def load_and_classify(images):

    net = Net()
    net.load_state_dict(torch.load(MODEL_PATH))
    classes = load_classes()

    num_images = len(images)  # note tested
    outputs = net(images)

    # ?

    print(outputs)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
          for j in range(num_images)))

    # In fact this is the input
    show_torch_images(images)


def process_loaded_images(image_list):
    """ todo: image_list is a list of loaded images in format of ..."""
    pass


def process_files(image_file_list):
    """
    The file_interface
    interface: one or a list of filenames
    """
    load_and_classify(image_file_list)


def demo():
    images = []
    load_and_classify(images)


if __name__ == '__main__':
    demo()

"""
References:
[1] Wonderful tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
