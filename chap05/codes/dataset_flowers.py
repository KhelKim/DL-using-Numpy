from dataset import *


class FlowersDataset(Dataset):
    pass


def flowers_init(self, resolution=[100, 100], input_shape=[-1]):
    super(FlowersDataset, self).__init__('flowers', 'select')

    path = "../../data/chap05/flowers"
    self.target_names = list_dir(path)

    images = []
    idxs = []

    for dx, dname in enumerate(self.target_names):
        subpath = f"{path}/{dname}"
        filenames = list_dir(subpath)
        for fname in filenames:
            if fname[-4:] != ".jpg":
                continue
            imagepath = f"{subpath}/{fname}"
            pixels = load_image_pixels(imagepath, resolution, input_shape)
            images.append(pixels)
            idxs.append(dx)

    self.images_shape = resolution + [3]

    xs = np.asarray(images, dtype="float32")
    ys = onehot(idxs, len(self.target_names))

    self.shuffle_data(xs, ys, 0.8)


FlowersDataset.__init__ = flowers_init


def flowers_visualize(self, xs, estimates, answers):
    draw_images_horz(xs, self.images_shape)
    show_select_results(estimates, answers, self.target_names)


FlowersDataset.visualize = flowers_visualize
