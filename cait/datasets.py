import os

from paddle.vision import datasets, transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def transforms_imagenet_train(img_size=224):
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(img_size),
         transforms.RandomHorizontalFlip(prob=0.5),
         transforms.RandomVerticalFlip(prob=0.2),
         transforms.ToTensor(),
         transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
         ]
    )
    return transform


def transforms_imagenet_eval(img_size=224):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    return transform


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    dataset, nb_classes = None, None
    if args.data_set == 'CIFAR':
        dataset = datasets.Cifar100(data_file=args.data_path, mode='train' if is_train else 'test',
                                    transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    return dataset, nb_classes


def build_transform(is_train, args):
    if is_train:
        return transforms_imagenet_train()
    else:
        return transforms_imagenet_eval()
