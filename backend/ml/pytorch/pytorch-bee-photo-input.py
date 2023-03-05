import torch
import torchvision

# shortcuts:
import torch.nn.functional as F
import torch.nn as nn
#

MODEL_PATH = './my-beecomb-model.pth'


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


# networks configuration
inp1_nch = 3  # inp_nchannels
out1_nch = 6
kern1_sz = 5
# previous size: torch.Size([4, 3, 32, 32])
# Conv2d:
#   torch.nn.Conv2d
#   .args: (in_channels, out_channels, kernel_size, stride=1)
#
# MaxPool2d:
#    torch.nn.MaxPool2d
#    .args: (kernel_size, stride)
#    Example: MaxPool2d(2, 2) -> (2x2) stride=(2,2)
#
#  32x32 --->  (32-5+1)x(32-5-1) = 28x28
#  conv1 --> 28x28 --> 14,14 but -2+1 -->13
#  28x28 ---> (28-2+1)^2 but (/2)^2 -> 13x13 ??

# second conv:
#    6dim input, 16dim output?! cool.
#    5x5 ==>
#      13x13 -->... -> (13-5+1)^2 = 9x9
#    9x9 x (16 channels)
#  But why is it 5?
# .... (?)
# flatten 16 * 5 * 5 --> 400
# linear: (400x120)

# rename to `model`?


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(inp1_nch, out1_nch, kern1_sz)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print('fwd1', x.shape)  # torch.Size([1, 3, 719, 1280])
        x = self.pool(F.relu(self.conv1(x)))
        print('fwd2', x.shape)  # torch.Size([1, 6, 357, 638])
        x = self.pool(F.relu(self.conv2(x)))
        print('fwd3', x.shape)  # torch.Size([1, 16, 176, 317])
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        print('fwd4', x.shape)  # torch.Size([1, 892672])
        # mat1 and mat2 shapes cannot be multiplied (1x892672 and 400x120)
        # 400x120 = mult(16 * 5 * 5, 120)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_data_generator(dataset_pairs):

    # train:
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0

        # enumerate(values, start=1)
        for i, data_pair in enumerate(dataset_pairs, 0):
            # get the inputs_image_batch
            # data_pair is a list of [inputs_image_batch, labels]
            inputs_image_batch, labels = data_pair

            assert len(inputs_image_batch) == len(
                labels), 'Number of abels should match number of images in a batch'
            for ii in range(len(inputs_image_batch)):
                # print(f'   batch item {i}',inputs_image_batch[ii].shape)  # (3,95,95)
                pass

            # the yield
            #################
            single_loss = (yield i, inputs_image_batch, labels)
            print('   generator:receive:', single_loss)
            #################

            # print statistics
            # running_loss += single_loss
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            if i > 4000:
                break
        print('.')
    print(':')
    print('Finished Training')
    # return('oh') don't return. It causes `StopIteration(value)` (if used by next())


def mini_tain(dataset_pairs):
    # list of tuples (pairs), each is a bach and a label_batch
    # todo: generator
    # array of batches (batches paired with labels)
    # rename: inputs_image_batch -> input_batch

    net = Net()
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # coroutine
    genr = train_data_generator(dataset_pairs)

    # can't send non-None value to a just-started generator
    # last_loss, last_loss_to_send, send_object
    send_object = None
    """
    for batch_index, inputs_image_batch, labels in genr:
    """
    try:
        while (True):
            batch_index, inputs_image_batch, labels = \
                genr.send(send_object)
            # next(genr)

            #
            print('from generator:', batch_index)
            # exit()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs_image_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            send_object = loss.item()

            # import random
            # lss = random.randint(1,10)
            # print('sending', lss)
            # genr.send(lss)
            # See [6]

            last_loss_to_send = lss
    except StopIteration as ss:
        print('closingg', ss)
        oo = genr.close()
        print('oooo', oo)

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

    print(len(images))  # 3
    print(images.shape)  # ([3, 95, 95])

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


def standard_loader(folder_path):
    # see https://stackoverflow.com/questions/58941007/how-do-i-load-multiple-grayscale-images-as-a-single-tensor-in-pytorch
    ds = torchvision.datasets.ImageFolder(
        folder_path, transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=3)
    print(len(dl))


def load_image_file(full_file_path):
    """
    Low-level (doe snot post-process)
    See https://www.tutorialspoint.com/how-to-read-a-jpeg-or-png-image-in-pytorch


    todo: much better:

    """
    # import torchvision
    from torchvision.io import read_image
    import torchvision.transforms as T
    """
    impl-docs:
    read_image() -> output (Tensor[image_channels, image_height, image_width])

    See img.convert('RGB') from https://stackoverflow.com/questions/58941007/how-do-i-load-multiple-grayscale-images-as-a-single-tensor-in-pytorch
    """

    img = read_image(full_file_path)
    if False:
        img = T.ToPILImage()(img)
        # for manual verificaiton
        # img.show()

        import torch
        print(
            f"Image of size {img.size}, type {type(img)}, is_tensor: { torch.is_tensor(img)} in path {full_file_path}")
        # example output: (1280, 719)

    print("type", type(img))  # <class 'torch.Tensor'>
    return img


def post_process(ptimg, mult_factor=10):
    """ Closely related to (fed by) load_image_file()

    # rename: post_process_and_multipy

    todo: make this return a transformation (in fact two)
    i.e. make this a transformation
    (return Compose, or even the array)

    Also see: (for future)
    https://pytorch.org/vision/stable/transforms.html

    inputs from raw loaded image file

    """
    # import pdb
    # pdb.set_trace()

    # no: ptimg[:5,:5]
    # ptimg.crop()

    # based on torch.nn.Conv2d .args: (in_channels, out_channels, kernel_size, stride=1)

    if False:
        # img = ... T.ToPILImage()(img) ...
       # pt_img_batch = post_process(pt_img)
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

        # return ... to be used by load_and_classify
        # load_and_classify([uns])
        load_and_classify(uns)
        return

    # RandomResizedCrop(size[, scale, ratio, ...]) #Crop a random portion of image and resize it to a given size.
    # RandomPerspective([distortion_scale, p, ...]) # Performs a random perspective transformation of the given image with a given probability.
    # RandomCrop(size[, padding, pad_if_needed, ...])  # Crop the given image at a random location.

    # Also this will work:
    import torchvision.transforms as T

    # See [4]
    # def get_transform(train):
    t = T.Compose([

        T.ToPILImage(),  # from what? Shall we input this? no.
        T.PILToTensor(),

        T.ConvertImageDtype(torch.float),
        # T.RandomHorizontalFlip(0.5)
        # T.RandomResizedCrop( (32,32) ), # error:  (16x25 and 400x120)
        # T.RandomResizedCrop((95, 95)),
    ])
    t2 = t(ptimg)
    print("4", t2.shape)  # torch.Size([3, 719, 1280])
    # return ptimg

    # single, non batched up to here:
    t2_batch = torch.unsqueeze(t2, 0)
    # rename: single_image_batch

    # to do: load multiple preloaded up to here (as a python list, not batch. or maybe batch is fine)
    # eithe rdraw randimly, or permutate (equal number: draw with substitution)

    #  a tranform that is repeated multipe times on a single-image batch of size (1, 3, w,h)
    multi_transorm = T.Compose([
        T.RandomResizedCrop((95, 95)),
    ])

    # modify below code if we want to multiply bigger batches (and shuffle them, or interlace them)
    assert t2_batch.shape[0] == 1
    # how to append multiple images in a batch pytorch
    # See [5]
    multiplied_batch = torch.stack(
        [multi_transorm(t2_batch[0]) for i in range(mult_factor)])

    print(multiplied_batch.shape)
    return multiplied_batch


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
        # todo: pre load all imates befor  this, and then retrain

        print('d1', pt_img.shape)

        pt_img_batch = post_process(pt_img)
        print('**d2', pt_img_batch.shape)
        print('**d2l', len(pt_img_batch.shape))
        assert len(pt_img_batch.shape) == 4
        print('lennnnn', len(pt_img_batch.shape))

        # exit()

        train_mode = True

        print(pt_img_batch.shape)  # torch.Size([3, 95, 95]) not a batch
        # exit()
        if train_mode:
            mini_tain([
                # pair 1
                (pt_img_batch, list(range(10)))
            ])
        else:
            load_and_classify(pt_img_batch)


def demo_fixed_files():
    apisave_base = '../../../../..'
    # todo: move sosi-practice-files to a test folder
    images_path = apisave_base + '/' + 'sosi-practice-files'

    # todo: use standard_loader
    # ibatch = standard_loader(images_path)
    # process_files(image_file_list)

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
    images = []  # wrong: not an array, it should be a batch
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

[5] how to append multiple images in a batch pytorch    https://stackoverflow.com/a/68102266/4374258

[6] Coroutines https://book.pythontips.com/en/latest/coroutines.html
"""
