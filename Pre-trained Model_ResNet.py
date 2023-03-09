import os
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
import torch.optim as optim
import time

class CustomImageDataset(Dataset):
    def read_data_set(self):
        all_img_files = [] 
        all_labels = []
        
        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]} 

    def __len__(self):
        return self.length

resnet18_pretrained = models.resnet18(pretrained=True)        

hyper_param_epoch = 10
hyper_param_batch = 8
hyper_param_learning_rate = 0.001

transforms_train = transforms.Compose([transforms.Resize((150, 150)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor()])

transforms_test = transforms.Compose([transforms.Resize((150, 150)),
                                      transforms.ToTensor()])

train_data_set = CustomImageDataset(data_set_path="D:/bongkj/Dataset/Intel_Image_Classification/seg_train/", transforms=transforms_train)
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

test_data_set = CustomImageDataset(data_set_path="D:/bongkj/Dataset/Intel_Image_Classification/seg_test/", transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

if not (train_data_set.num_classes == test_data_set.num_classes):
    print("error: Numbers of class in training set and test set are not equal")
    exit()
    
num_classes = train_data_set.num_classes
# num_ftrs = resnet18_pretrained.fc.in_features
# resnet18_pretrained.fc = nn.Linear(num_ftrs, num_classes)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
resnet18_pretrained.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet18_pretrained.parameters(), lr=hyper_param_learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

for e in range(hyper_param_epoch):
    for i_batch, item in enumerate(train_loader):
        images = item['image'].to(device)
        labels = item['label'].to(device)

        outputs = resnet18_pretrained(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    if e % 1 == 0:
        print(f'Epoch {e+1}/{hyper_param_epoch}, LR: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {loss.item():.4f}')
            
start = time.time()
resnet18_pretrained.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for item in test_loader:
        images = item['image'].to(device)
        labels = item['label'].to(device)
        outputs = resnet18_pretrained(images)
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()

    print(f'total : {total}, correct : {correct}')
    print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
    
end = time.time()
print(f'time: {end-start}')