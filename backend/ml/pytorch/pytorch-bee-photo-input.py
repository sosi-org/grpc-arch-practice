import torch
import torchvision
import numpy

# shortcuts:
import torch.nn.functional as F
import torch.nn as nn
#

# todo: not use current directory
MODEL_PATH = './my-beecomb-model.pth'

# todo: remove global variable
# global cli_mode
cli_mode = True

# move near load_image_file()


def imshow(img):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    import os
    matplotlib.rcParams["savefig.directory"] = os.getcwd()

    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_torch_imagebatch(images):

    # pad to almost square
    import math
    n = len(images)
    ϵ = 0.00000000001
    nrows = math.ceil(math.sqrt(n) + ϵ)
    ncols = math.ceil(float(n) / float(nrows) + ϵ)

    reminder = nrows * ncols - n
    print(f'Padding: {nrows} * {ncols} - {n} = {reminder}')
    assert reminder >= 0, 'negative padding?'

    pad1 = images[0].unsqueeze(0) * 0.0 + 1
    # print(pad1.shape) # torch.Size([1, 3, 32, 32])
    images = torch.cat([images] + [pad1]*reminder, dim=0)
    img_rgb = torchvision.utils.make_grid(images, nrow=nrows)
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

        self.accum_count = 0

    def forward(self, x):
        print('fwd1', x.shape)  # ([1, 3, 719, 1280])    ([5000, 3, 32, 32])
        x = self.pool(F.relu(self.conv1(x)))
        print('fwd2', x.shape)  # ([1, 6, 357, 638])     ([5000, 6, 14, 14])
        x = self.pool(F.relu(self.conv2(x)))
        print('fwd3', x.shape)  # ([1, 16, 176, 317])    ([5000, 16, 5, 5])
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        print('fwd4', x.shape)  # ([1, 892672])          ([5000, 400])
        # mat1 and mat2 shapes cannot be multiplied (1x892672 and 400x120)
        # 400x120 = mult(16 * 5 * 5, 120)
        x = F.relu(self.fc1(x))
        print('fwd5', x.shape)  # ([5000, 120])
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        self.accum_count += len(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print('Saved τ₁ state')
        print('accum_count:', self.accum_count)
        # 100'000

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        print('Loaded τ₁ state')

    def load_if_file_exists(self, filename):
        """ or: load_if_exists() """
        try:
            self.load(filename)
        except FileNotFoundError as e:
            print("loading saved τ₁ state failed. "
                  "Continuing without loading. Zero initiaisaitng τ₁=0")


def train_data_generator(dataset_pairs):

    # train:

    repeats_same_batch_training_insisting = 2*10

    # loop over the same batch multiple times
    for epoch in range(repeats_same_batch_training_insisting):
        running_loss = 0.0

        # enumerate(values, start=1)
        for i, data_pair in enumerate(dataset_pairs, 0):
            # get the inputs_image_batch
            # data_pair is a list of [inputs_image_batch, labels]
            inputs_image_batch, labels = data_pair

            # print(type(inputs_image_batch))
            # print(type(labels))
            assert type(labels) == torch.Tensor, type(labels)
            assert type(inputs_image_batch) == torch.Tensor, type(
                inputs_image_batch)

            print(len(inputs_image_batch))
            print(len(labels))
            # print((labels))
            # assert len(inputs_image_batch) == len(labels), \
            #    'Number of labels should match number of images in a batch'

            for ii in range(len(inputs_image_batch)):
                # print(f'   batch item {ii}', inputs_image_batch[ii].shape)
                # (3,95,95)
                pass

            # the yield
            #################
            single_loss = (yield i, inputs_image_batch, labels)
            print('   generator received: loss:', single_loss)
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


def mini_train(dataset_pairs):
    # list of tuples (pairs), each is a bach and a label_batch
    # todo: generator
    # array of batches (batches paired with labels)
    # rename: inputs_image_batch -> input_batch

    net = Net()

    # load previous training: Accumulate learning across runs
    net.load_if_file_exists(MODEL_PATH)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # coroutine
    genr = train_data_generator(dataset_pairs)
    # rename: genr -> training_data_stream
    # training_data_stream: ( (image,label) -> STREAM -> loss)
    try:
        # no "loss" before first instance of training
        # can't send non-None value to a just-started generator
        # last_loss, last_loss_to_send, send_object, _coroutine_send_object
        _coroutine_send_object = None
        while (True):

            batch_index, inputs_image_batch, labels = \
                genr.send(_coroutine_send_object)
            # next(genr), genr.send(), See [6]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs_image_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss1 = loss.item()

            _coroutine_send_object = loss1
    except StopIteration as ss:
        print('closingg', ss)
        oo = genr.close()
        print('oooo', oo)

    net.save(MODEL_PATH)

    print('bye')

    exit()

    pass


def load_classes():
    classes = ('d1', 's2')
    # classes = load_dict(...)
    return classes


def load_and_classify(images):

    import torch

    net = Net()

    # pre training: zill training:
    # if (False):
    net.load(MODEL_PATH)

    classes = load_classes()

    print(len(images))  # 3
    print(images.shape)  # ([3, 95, 95])

    num_images = len(images)
    # 5000
    outputs = net(images)

    # ?

    print(outputs)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
          for j in range(num_images)))

    for i in range(images.shape[0]):
        if predicted[i] == 0:
            # images[i] = images[i] / 2.0 + 0.5
            images[i] = images[i] / 2.0 + 0.5
            # images[i][1:3] = 0 # red
            images[i][0] = 1.0  # red
    # global cli_mode
    if not cli_mode:
        # In fact this is the input
        show_torch_imagebatch(images)

    return predicted


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


def post_process(ptimg_a, labels_a, mult_factor):
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
        # mini_train([[pt_img1.unsqueeze(0), 1]])

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

    def pre_proc_each_image(ptimg):
        t2 = t(ptimg)
        # print("4", t2.shape)  # torch.Size([3, 719, 1280])
        # return ptimg

        # single, non batched up to here:
        t2_batch1 = torch.unsqueeze(t2, 0)
        # rename: single_image_batch

        return t2_batch1

    t2_batch_a = map(pre_proc_each_image, ptimg_a)
    # load them all in advance
    t2_batch_a = list(t2_batch_a)

    # to do: load multiple preloaded up to here (as a python list, not batch. or maybe batch is fine)
    # eithe rdraw randimly, or permutate (equal number: draw with substitution)

    #  a tranform that is repeated multipe times on a single-image batch of size (1, 3, w,h)
    multi_transorm = T.Compose([
        #         T.RandomResizedCrop((95, 95)),
        T.RandomResizedCrop((32, 32)),
    ])

    # modify below code if we want to multiply bigger batches (and shuffle them, or interlace them)
    def combinations(t2_batch_a_len, mult_count, method):
        """
        Generate n times indices of m images: n * m

        @return (j, i) : in (image_id, trial)
        @param `method`:
            * np.random.permutation(m)
            * np.random.permutation(m * n)  -- slightly faster because we can, also can cover more diverse sequences, inclusing repeats
            * m * n times draw from n -- more diverse, may sample a non-equal number from differnt images
            * Also two differnt orders of `for` are possible, but one is trivial

        Example:
        (2, 3, 'SHUFFLE'):
            0 0
            1 0
            0 1
            1 1
            0 2
            1 2
        """
        # only one methos is implemented
        assert method == 'SHUFFLE'
        # m = len(t2_batch_a)
        m = t2_batch_a_len
        n = mult_count
        for i in range(n):
            for j in numpy.random.permutation(m):
                yield j, i  # j in range(m), i in range(n)
    # manual unit test

    def manual_test():
        for rep in range(3):
            numpy.random.seed(2)
            for j, i in combinations(2, 3, 'SHUFFLE'):
                print(j, i)
            print()
    # manual_test()
    # exit()

    # stimuli, num_trials: Ns, Ntr
    #  Ns = same as Nr
    # Ns = len(pt_img_a)
    # Ntr = mult_factor_count
    # Ntr * Ns

    Ns = len(t2_batch_a)
    Ntr = mult_factor
    for t2_batch in t2_batch_a:
        assert t2_batch.shape[0] == 1
    # how to append multiple images in a batch pytorch
    # See [5]
    numpy.random.seed(2)
    # todo: only if training_mode
    multiplied_batch = torch.stack(
        [multi_transorm(t2_batch_a[j][0]) for j, i in combinations(Ns, Ntr, 'SHUFFLE')])

    npa = numpy.array(
        [labels_a[j] for j, i in combinations(Ns, Ntr, 'SHUFFLE')],
        dtype=int)
    # not tested

    # todo: contents not validated/confirmed

    labels_batch = torch.Tensor(npa).long()
    assert len(labels_batch) == Ns * Ntr

    # todo: I need to do this twice, second time for labels
    #  solution 1: fix the random seed (from fetched one)
    #  solution 2: `yield from` (then, unzip?), (or multiply), (is it possible for two loops to use the same? yes: using a for loop)
    #   But will need to yeild out to two output channels
    # For now, I sed the seed solution

    assert multiplied_batch.shape[0] == Ns * Ntr

    return multiplied_batch, labels_batch


def process_files(image_file_list, labels_a):
    """
    API service will call this

    The file_interface
    interface: one or a list of filenames


    interesting: Can run run videos too: read_video()
    Also can write `write_png()`, good for highlighting.
    """

    train_mode = False

    pt_img_a = []
    # todo: create a batch first, dont call for each input individually (done)
    for full_filename in image_file_list:
        pt_img = load_image_file(full_filename)
        # todo: pre-load all imates befor  this, and then retrain (done)

        print('d1', pt_img.shape)
        pt_img_a += [pt_img]

    # todo: many more repeats, but not geenratng all at the same time in memory
    # also, read them async/delayed/yielded/re-chunk (re-vectorise)

    if train_mode:
        mult_factor_count = 50*50
        assert not labels_a == None
    else:
        mult_factor_count = 1  # 40
        # assert labels_a == None

    # Nr labels = len(labels_a)
    # labels_a = [0, 1]

    pt_img_batch, labels_batch2 = post_process(
        pt_img_a, labels_a, mult_factor_count)
    # print('shape', pt_img_batch.shape)  # torch.Size([100, 3, 32, 32])
    assert len(pt_img_batch.shape) == 4, f' ndim {len(pt_img_batch.shape)}'

    print(pt_img_batch.shape)  # torch.Size([3, 95, 95]) not a batch
    # exit()

    if False:  # decommissioning outdated labels logic
        def listint_to_tensor(list_of_int):
            labels_np = numpy.array(list_of_int, dtype=int)
            # labels = torch.stack([torch.Tensor(labels_np).int()]) # Expected input batch_size (10) to match target batch_size (1).
            # labels = torch.stack([torch.Tensor(labels_np).int()]).T # 0D or 1D target tensor expected, multi-target not supported
            # labels = torch.Tensor(labels_np).int() # Expectged Long
            labels = torch.Tensor(labels_np).long()  # Expectged Long
            return labels
        # listint_to_tensor([1,2,3])
        # input_tensor = torch.tensor([[1.0, 2.0, 3.0], [0.5, 2.5, 1.0], [0.8, 1.2, 3.0]])
        # target_tensor = torch.tensor([2, 1, 0])

        def label_batch(count, value):
            return torch.Tensor(numpy.full((count,), value, dtype=int)).long()

        # exit()

        # stimuli, num_trials

        # same as Nr
        Ns = len(pt_img_a)
        # Ntr = mult_factor_count
        labels_batch1 = label_batch(mult_factor_count * Ns, 1)

    # global cli_mode
    visualise = not (not train_mode) and (not cli_mode)
    if visualise:
        show_torch_imagebatch(pt_img_batch)
    # todo: (maybe): send this out so that the next ones are done after this

    # todo: give `mini_train()` a generator, not array
    if train_mode:
        mini_train([
            # pair 1
            (pt_img_batch, labels_batch2)
        ])
        return None
    else:
        return load_and_classify(pt_img_batch)


def demo_fixed_files(file_list=None):
    if file_list == None:
        print('demo mode')
        # global cli_mode
        # no, dont set it to False here
        # cli_mode = False
        # assert cli_mode == False

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
        # tarining only
        labels_a = [0, 1]

        #
        image_file_list = map(
            lambda fn: images_path + '/' + fn,
            fn_list)
    else:
        print('CLI mode: with specified files:', repr(file_list))
        # global cli_mode
        assert cli_mode == True
        image_file_list = file_list
        labels_a = list(map(lambda x: 0, file_list))
        # todo:
        # labels_a = None

    pass

    predicted = process_files(image_file_list, labels_a)

    print('predicted', predicted)

    # global cli_mode
    if cli_mode:
        results = []
        for i in range(len(predicted)):
            label = predicted[i].item()
            filename = fn_list[i]
            print(filename, ':', label)

            jsonobj = {}
            jsonobj['filename'] = filename
            jsonobj['diagnosis'] = label

            results.append(jsonobj)
        print(results)

        import json
        result_filename = 'results.json'
        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print('results json saved to file: ' + result_filename)

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
    import sys
    if len(sys.argv) == 1:
        demo_fixed_files()
    elif len(sys.argv) > 1:
        # files specified
        demo_fixed_files(sys.argv[1:])


"""
References:
[1] Wonderful tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

[2] read_image() https://pytorch.org/vision/main/generated/torchvision.io.read_image.html#torchvision.io.read_image

[3] Nice: torch.jit.script: https://pytorch.org/vision/main/auto_examples/plot_scripted_tensor_transforms.html#sphx-glr-auto-examples-plot-scripted-tensor-transforms-py

[4] https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

[5] how to append multiple images in a batch pytorch    https://stackoverflow.com/a/68102266/4374258

[6] Coroutines https://book.pythontips.com/en/latest/coroutines.html

[7] Useful: image.resize((imageSize, imageSize), Image.ANTIALIAS)

[8] interesting:  torch.cat(matrices).cpu().detach().numpy() https://stackoverflow.com/questions/65940793/how-to-stack-matrices-with-different-size


"""
