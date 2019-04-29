import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders import utils

class KittiesSegmentation(data.Dataset):
    NUM_CLASSES = 12

    def __init__(self, args, root=Path.db_root_dir('kitti'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, 'image_2')
        self.annotations_base = os.path.join(self.root, 'semantic_rgb')

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [11]
        self.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.class_names = ['Sky', 'Building', 'Road', 'Sidewalk', \
                    'Fence', 'Vegetation', 'Pole', 'Car', \
                    'Sign', 'Pedestrian', 'Cyclist', 'void']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        self.mapping = {
            torch.tensor([128, 128, 128], dtype=torch.uint8):0,
            torch.tensor([128, 0, 0], dtype=torch.uint8):1,
            torch.tensor([128, 64, 128], dtype=torch.uint8):2,
            torch.tensor([0, 0, 192], dtype=torch.uint8):3,
            torch.tensor([64, 64, 128], dtype=torch.uint8):4,
            torch.tensor([128, 128, 0], dtype=torch.uint8):5,
            torch.tensor([192, 192, 128], dtype=torch.uint8):6,
            torch.tensor([64, 0, 128], dtype=torch.uint8):7,
            torch.tensor([192, 128, 128], dtype=torch.uint8):8,
            torch.tensor([64, 64, 0], dtype=torch.uint8):9,
            torch.tensor([0, 128, 192], dtype=torch.uint8):10,
            torch.tensor([0, 0, 0], dtype=torch.uint8):11
            }

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def mask_to_class(self, mask):
        for k in self.mapping:
#             print(k.dtype)
#             print(mask.dtype)
            mask[mask==k] = self.mapping[k]
        return mask

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        print(img_path.split(os.sep)[-1])
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-1])

        _img = Image.open(img_path).convert('RGB')
        mask_arr = np.array(Image.open(lbl_path), dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_arr)
        mask_tensor = mask_tensor.permute(2,0,1)
        mask_tensor = mask_tensor.view(mask_tensor.size(0), -1).permute(1,0)
        mask_tensor = self.encode_segmap(mask_tensor)
        _target = Image.fromarray(mask_tensor)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        results = np.zeros(mask.shape[:-1])
        for _voidc in self.void_classes:
            results[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    kitties_train = KittiesSegmentation(args, split='train')

    dataloader = DataLoader(kitties_train, batch_size=2, shuffle=True, num_workers=2, drop_last = True)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            # segmap = decode_segmap(tmp, dataset='kitti')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(tmp)

        if ii == 1:
            break

    plt.show(block=True)
