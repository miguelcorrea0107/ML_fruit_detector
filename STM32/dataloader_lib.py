import os 
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
from torch.utils.data import random_split
import multiprocessing




# def get_data(data_loader, batch_size):
#     images_return = []
#     labels_return = []

#     i=0
#     for batch in data_loader:
#         images, labels = batch
#         images_array = images.numpy()
#         if images_array.shape[0]!=batch_size*2:
#             print(f"Batch {i} has {images_array.shape[0]} images")
#             continue
#         labels_array = labels.numpy()

#         images_return.append(images_array)
#         labels_return.append(labels_array)
#         i+=1
    

#     images_return = np.array(images_return) 
#     # print(images_return.min(), images_return.max())
#     images_return = np.reshape(images_return, (images_return.shape[0]*images_return.shape[1], images_return.shape[2], images_return.shape[3], images_return.shape[4]))
#     labels_return = np.array(labels_return) # / 255.0
#     labels_return = np.reshape(labels_return, (labels_return.shape[0]*labels_return.shape[1]))
        
#     return images_return, labels_return


def load_images_by_torch(folder, batch_size = 40, image_size=(100,100), validation_set=False, verbose=True):

    data_tfms = tt.Compose([tt.Resize((100, 100)), 
                        tt.ToTensor(),
                        tt.RandomRotation(30),
                        tt.RandomCrop(100, padding=4, padding_mode='reflect'), 
                        #tt.RandomHorizontalFlip(), 
                        #tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                        tt.Normalize(mean=[0.582, 0.485, 0.415], std=[0.332, 0.359, 0.366] )
                        
                        
                        ]

                        
        )
    
    # classes = os.listdir(folder)
    images_return_final = []
    labels_return_final = []
    dataset = ImageFolder(folder, transform=data_tfms)

    # if validation_set == False:
    data_loader = DataLoader(dataset, batch_size, shuffle=False, pin_memory=True)
    images_return = []
    labels_return = []

    i=0
    for batch in data_loader:
        images, labels = batch
        images_array = images.numpy()
        if verbose and i%100==0:
            print("Loaded batch ", i)
        if images_array.shape[0]!=batch_size:
            print(f"Batch {i} has {images_array.shape[0]} images, skipping...")
            continue
        labels_array = labels.numpy()

        images_return.append(images_array)
        labels_return.append(labels_array)
        i+=1
    print("Loaded a total of ", i * batch_size, " images.")
    

    images_return = np.array(images_return) 
    # print(images_return.min(), images_return.max())
    images_return = images_return.reshape(-1, images_return.shape[2], images_return.shape[3], images_return.shape[4])
    images_return = images_return.transpose(0, 2, 3, 1)
    labels_return = np.array(labels_return) # / 255.0
    labels_return = labels_return.reshape(-1) ####
        
    return images_return, labels_return, dataset.classes
        
    # else:
    #     val_size = round(len(dataset) * 0.2)
    #     train_size = round(len(dataset) - val_size)
    #     train_ds, val_ds = random_split(dataset, [train_size, val_size])
    #     train_loader = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)
    #     val_loader = DataLoader(val_ds, batch_size*2, pin_memory=True)
    #     tr_im, tr_lb = get_data(train_loader, batch_size)
    #     vl_im, vl_lb = get_data(val_loader, batch_size)
    #     return tr_im, tr_lb, vl_im, vl_lb
        
        



def load_images_from_folder(folder, image_size=(100, 100), data_type=np.float32):
    images = []
    labels = []
    class_labels = os.listdir(folder)
    for label in class_labels:
        class_folder = os.path.join(folder, label)
        if not os.path.isdir(class_folder):
            continue
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            try:
                img = Image.open(img_path)
                img = img.resize(image_size)
                img_array = np.array(img).astype(data_type) #// 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    # images = np.transpose(images, (0, 3, 1, 2))  # Change shape to (N, C, H, W)

    return np.array(images), np.array(labels)

# Specify the dataset path and desired image size
# dataset_path = '/Users/TomasPacheco/Documents/MA2/MLuC/Project/ML_Model/fruits/fruits-360/Test'
# image_size = (100, 100)  # Resize images to 100x100 (or any desired size)

# # Load images and labels
# images, labels = load_images_from_folder(dataset_path, image_size)


# # Encode labels to numeric values
# label_encoder = LabelEncoder()
# numeric_labels = label_encoder.fit_transform(labels)

# print(f"Images shape: {images.shape}, Images type:{type(images)}, Labels shape: {numeric_labels.shape}, Labels type:{type(numeric_labels)}")

def print_tflite_model_details(interpreter):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    all_tensor_details = interpreter.get_tensor_details()

    # Print input details
    print("Input Details:")
    for detail in input_details:
        print(f"  Name: {detail['name']}")
        print(f"  Index: {detail['index']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Data Type: {detail['dtype']}")
        print()

    # Print output details
    print("Output Details:")
    for detail in output_details:
        print(f"  Name: {detail['name']}")
        print(f"  Index: {detail['index']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Data Type: {detail['dtype']}")
        print()

    # Print all tensor details
    print("All Tensor Details:")
    for detail in all_tensor_details:
        print(f"  Name: {detail['name']}")
        print(f"  Index: {detail['index']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Data Type: {detail['dtype']}")
        print(f"  Quantization Parameters: {detail['quantization']}")
        print()


def hex_to_c_array(hex_data, var_name):

    c_str = ''

    # Create header guard
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'

    # Add array length at top of file
    c_str += '\nstatic const unsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

    # Declare C variable
    c_str += 'static const unsigned char ' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data) :

        # Construct string from hex
        hex_str = format(val, '#04x')

        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '
        hex_array.append(hex_str)

    # Add closing brace
    c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

    # Close out header guard
    c_str += '#endif //' + var_name.upper() + '_H'

    return c_str