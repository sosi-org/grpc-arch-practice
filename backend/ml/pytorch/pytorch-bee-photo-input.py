
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


def load_file(full_file_path):
    """
    See https://www.tutorialspoint.com/how-to-read-a-jpeg-or-png-image-in-pytorch
    """
    # import torchvision
    from torchvision.io import read_image
    import torchvision.transforms as T

    img = read_image(full_file_path)
    img = T.ToPILImage()(img)
    img.show()


def process_files(image_file_list):
    """
    API service will call this

    The file_interface
    interface: one or a list of filenames
    """
    for full_filename in image_file_list:
        pt_img = load_file([full_filename])
        # load_and_classify(image_file_list)
        load_and_classify([pt_img])


def demo_fixed_files():
    apisave_base = '../../../../..'
    # todo: move sosi-practice-files to a test folder
    images_path = apisave_base + '/' + 'sosi-practice-files'
    fn_list = [
        'photo-sosi-2023-01-21-T-21-12-53.763.jpg',
        'photo-sosi-2023-01-21-T-21-41-05.593.jpg',
    ]
    #
    image_file_list = fn_list.map(lambda fn: images_path + '/' + fn)
    process_files(image_file_list)

    # for i in range(len(fn_list)):
    #    fn = fn_list[i]
    #    filename = images_path + '/' + fn
    #
    #    load_file(filename)


def demo():
    images = []
    load_and_classify(images)


if __name__ == '__main__':
    # demo()

    demo_fixed_files()


"""
References:
[1] Wonderful tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
