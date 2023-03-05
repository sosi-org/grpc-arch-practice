import torch
import torchvision

# shortcuts:
import torch.nn.functional as F
import torch.nn as nn
#

MODEL_PATH = './my-cifar_net.pth'


# move near load_image_file()
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


# rename to `model`?
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
        print('fwd1')
        x = self.pool(F.relu(self.conv1(x)))
        print('fwd2')
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # mat1 and mat2 shapes cannot be multiplied (1x892672 and 400x120)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def mini_tain(dataset_imagearray):
    net = Net()
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train:
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in enumerate(dataset_imagearray, 0):

            print(i, 'data:', data)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            if i > 4000:
                break
        print('.')
    print(':')

    print('Finished Training')

    # torch.save(net.state_dict(), SAVE_PATH)
    # print('Saved state')
    print('bye')

    exit()

    pass


def load_classes():
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # classes = load_dict(...)
    return classes


def load_and_classify(images):

    import torch

    net = Net()

    # pre training: zill training:
    if (False):
        net.load_state_dict(torch.load(MODEL_PATH))

    classes = load_classes()

    # num_images = len(images)  # note tested
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


def load_image_file(full_file_path):
    """
    Low-level (doe snot post-process)
    See https://www.tutorialspoint.com/how-to-read-a-jpeg-or-png-image-in-pytorch
    """
    # import torchvision
    from torchvision.io import read_image
    import torchvision.transforms as T
    """
    impl-docs:
    read_image() -> output (Tensor[image_channels, image_height, image_width])
    """

    img = read_image(full_file_path)
    img = T.ToPILImage()(img)
    # for manual verificaiton
    # img.show()

    import torch
    print(
        f"Image of size {img.size}, type {type(img)}, is_tensor: { torch.is_tensor(img)} in path {full_file_path}")
    # example output: (1280, 719)

    return img


def post_process(ptimg):
    """ Closely related to (fed by) load_image_file()

    Also see: (for future)
    https://pytorch.org/vision/stable/transforms.html

    inputs from raw loaded image file

    """
    # import pdb
    # pdb.set_trace()

    # no: ptimg[:5,:5]
    # ptimg.crop()

    # based on torch.nn.Conv2d .args: (in_channels, out_channels, kernel_size, stride=1)
    return ptimg


def process_files(image_file_list):
    """
    API service will call this

    The file_interface
    interface: one or a list of filenames


    interesting: Can run run videos too: read_video()
    Also can write `write_png()`, good for highlighting.
    """
    # todo: no, create a batch first, dont call for each input individually
    for full_filename in image_file_list:
        pt_img = load_image_file(full_filename)

        pt_img1 = post_process(pt_img)

        """
        `.unsqueeze(0)`:
        GPT:
        Note that we need to unsqueeze the image tensor along the first dimension (using image.unsqueeze(0)) in order to add an additional batch dimension to the tensor. This is because PyTorch's convolutional layers expect input tensors with four dimensions: a batch dimension (which specifies the number of input images in the batch), a channel dimension (which specifies the number of input channels), and two spatial dimensions (which specify the height and width of the input images). By unsqueezing the image tensor along the first dimension, we add a batch dimension with a size of 1, so that the input tensor has the required four dimensions.
        """
        # train
        # mini_tain([[pt_img1.unsqueeze(0), 1]])

        # #load_and_classify(pt_img1)
        # #load_and_classify(image_file_list)
        # test

        # import pdb;pdb.set_trace()

        # print("1", pt_img1.shape)
        ti = torchvision.transforms.ToTensor()(pt_img1)
        #  argument 'input' (position 1) must be Tensor, not Image
        # (C x H x W)
        print("2", ti.shape)  # torch.Size([3, 719, 1280])
        uns = torch.unsqueeze(ti, 0)
        # (1, C x H x W)
        print("3", uns.shape)  # torch.Size([1, 3, 719, 1280])
        # load_and_classify([uns])
        load_and_classify(uns)
        return

        # Also this will work:
        import torchvision.transforms as T

        # See [4]
        # def get_transform(train):
        t = T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
            # T.RandomHorizontalFlip(0.5)
        ])
        t2 = t(pt_img1)
        print("4", t2.shape)  # torch.Size([3, 719, 1280])

        # return t2
        # load_and_classify(t2)


def demo_fixed_files():
    apisave_base = '../../../../..'
    # todo: move sosi-practice-files to a test folder
    images_path = apisave_base + '/' + 'sosi-practice-files'
    fn_list = [
        'photo-sosi-2023-01-21-T-21-12-53.763.jpg',
        'photo-sosi-2023-01-21-T-21-41-05.593.jpg',
    ]
    #
    image_file_list = map(
        lambda fn: images_path + '/' + fn,
        fn_list)
    process_files(image_file_list)

    # for i in range(len(fn_list)):
    #    fn = fn_list[i]
    #    filename = images_path + '/' + fn
    #
    #    load_image_file(filename)


def demo():
    images = []
    load_and_classify(images)


if __name__ == '__main__':
    # demo()

    demo_fixed_files()


"""
References:
[1] Wonderful tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

[2] read_image() https://pytorch.org/vision/main/generated/torchvision.io.read_image.html#torchvision.io.read_image

[3] Nice: torch.jit.script: https://pytorch.org/vision/main/auto_examples/plot_scripted_tensor_transforms.html#sphx-glr-auto-examples-plot-scripted-tensor-transforms-py

[4] https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

"""
