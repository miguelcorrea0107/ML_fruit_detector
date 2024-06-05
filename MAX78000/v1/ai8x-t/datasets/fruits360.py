import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import ai8x

def fruits360_get_datasets(data, load_train=True, load_test=True):
    """
    Load the Fruits-360 dataset with additional augmentation for the training set.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-128/128, +127/128].

    Data augmentation for training data includes random resizing, cropping, horizontal flipping, and rotation.
    """
    (data_dir, args) = data

    # Define transformations for training data
    if load_train:
        print("Loading training dataset...")
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),  # Rotate by ±20 degrees
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        # Load the training dataset
        train_dataset = torchvision.datasets.ImageFolder(root=f'{data_dir}/Training',
                                                         transform=train_transform)
    else:
        train_dataset = None

    # Define transformations for test data
    if load_test:
        print("Loading test dataset...")
        test_transform = transforms.Compose([
            transforms.Resize(100),
            transforms.CenterCrop(100),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        # Load the test dataset
        test_dataset = torchvision.datasets.ImageFolder(root=f'{data_dir}/Test',
                                                        transform=test_transform)

        if args.truncate_testset:
            # Optionally truncate the test dataset for quick testing
            test_dataset.samples = test_dataset.samples[:1]  # Adjust number as needed
            test_dataset.targets = test_dataset.targets[:1]  # Adjust number as needed
        print(f"Number of test images: {len(test_dataset)}")
        print(f"Number of classes: {len(test_dataset.classes)}")
    else:
        test_dataset = None
    return train_dataset, test_dataset

datasets = [
{
        'name': 'fruits360',
        'input': (3, 100, 100),  # Adjust the dimensions based on the resizing/cropping you apply in your transforms
        'output': ('Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 
                   'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 
                   'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 
                   'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 
                   'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 
                   'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 
                   'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 
                   'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 
                   'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 
                   'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 
                   'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 
                   'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 
                   'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 
                   'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry', 
                   'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 
                   'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow', 'Tomato not Ripened', 'Walnut', 'Watermelon'),  # You can specify the classes directly if needed, or use 'auto' if the loader dynamically determines them
        'loader': fruits360_get_datasets,
    },
]

# Plot a few images from the dataset
def plot_images(dataset, num_images=16):
    dataloader = DataLoader(dataset, batch_size=num_images, shuffle=True)
    images, labels = next(iter(dataloader))

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    classes = dataset.classes  # Class names from dataset

    for i, (img, label) in enumerate(zip(images, labels)):
        ax = axes[i // 4, i % 4]
        img = img.permute(1, 2, 0)  # Reorder dimensions for displaying
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Class: {classes[label]}')
    plt.suptitle('Sample Images from Dataset')
    plt.show()

# Example usage
if __name__ == '__main__':
    dataset_path = "./ai8x-training/data"  # Path to dataset
    args = {'truncate_testset': False}  # Example args, define as needed
    train_dataset, test_dataset = fruits360_get_datasets((dataset_path, args))
    print(f'Number of training images: {len(train_dataset)}')
    print(f'Number of classes: {len(train_dataset.classes)}')
    print(f'Classes: {train_dataset.classes}')
    plot_images(train_dataset)