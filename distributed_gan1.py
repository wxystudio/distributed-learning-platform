import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import copy
import random
import os
from torchvision.utils import save_image
import torch.nn.functional as F
import pdb
import copy
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyper parameters
num_epochs =100
num_classes = 10
batch_size = 100
learning_rate = 0.001
sample_dir = 'samples1'
latent_size = 64
hidden_size = 256
image_size = 784
# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
# MNIST dataset
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                 std=(0.5, 0.5, 0.5))])
    
train_dataset_1 = torchvision.datasets.MNIST(root='C:/Users/SmartSpace/Desktop/mnist/mnist_data',
                                           train=True, 
                                           transform=transform,
                                           download=False)

train_dataset_2 = torchvision.datasets.MNIST(root='C:/Users/SmartSpace/Desktop/mnist/mnist_data',
                                           train=True, 
                                           transform=transform,
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root='C:/Users/SmartSpace/Desktop/mnist/mnist_data',
                                          train=False, 
                                          transform=transform)

# Data loader
train_loader_1 = torch.utils.data.DataLoader(dataset=train_dataset_1,
                                           batch_size=batch_size, 
                                           shuffle=True)

# Data loader
train_loader_2 = torch.utils.data.DataLoader(dataset=train_dataset_2,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

server=Net()
server.to(device)

user_1=Net()
user_1.to(device)

user_2=Net()
user_2.to(device)

temp_1=Net()
temp_1.to(device)

temp_2=Net()
temp_2.to(device)

temp_1.load_state_dict(user_1.state_dict())
temp_2.load_state_dict(user_2.state_dict())
server.load_state_dict(user_1.state_dict())

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(server.parameters(), lr=learning_rate)

criterion_1 = nn.CrossEntropyLoss()
optimizer_1 = torch.optim.Adam(user_1.parameters(), lr=learning_rate)

criterion_2 = nn.CrossEntropyLoss()
optimizer_2 = torch.optim.Adam(user_2.parameters(), lr=learning_rate)
# Train the server
total_step_1 = len(train_loader_1)
total_step_2 = len(train_loader_2)

for epoch in range(num_epochs):
    for i, ((images_1, labels_1), (images_2,labels_2)) in enumerate(zip(train_loader_1, train_loader_2)):
        #print(server.conv1.weight.data.size())
        images_1 = images_1.to(device)
        labels_1 = labels_1.to(device)
        images_2 = images_2.to(device)
        labels_2 = labels_2.to(device)
        
        #Forward pass  
        outputs_1 = user_1(images_1)
        loss_1 = criterion_1(outputs_1, labels_1)
        
        outputs_2 = user_2(images_2)
        loss_2 = criterion_2(outputs_2, labels_2)
        
        # Backward and optimize
        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()
        
        optimizer_2.zero_grad()
        loss_2.backward()
        optimizer_2.step()
        
#====================================================================================================
        
        conv1weight_1 = user_1.conv1.weight.data - temp_1.conv1.weight.data
        conv1weight_2 = user_2.conv1.weight.data - temp_2.conv1.weight.data
        F.dropout(conv1weight_1, p = 0.01, inplace = True)
        serverconv1weight = torch.where(conv1weight_1 != 0, conv1weight_1, conv1weight_2)
        server.conv1.weight.data.add_(serverconv1weight)
        #pdb.set_trace()
          
        conv1bias_1 = user_1.conv1.bias.data - temp_1.conv1.bias.data
        conv1bias_2 = user_2.conv1.bias.data - temp_2.conv1.bias.data
        F.dropout(conv1bias_1, p = 0.01, inplace = True)
        serverconv1bias = torch.where(conv1bias_1 != 0, conv1bias_1, conv1bias_2)
        server.conv1.bias.data.add_(serverconv1bias)
            
        conv2weight_1 = user_1.conv2.weight.data - temp_1.conv2.weight.data
        conv2weight_2 = user_2.conv2.weight.data - temp_2.conv2.weight.data
        F.dropout(conv2weight_1, p = 0.01, inplace = True)
        serverconv2weight = torch.where(conv2weight_1 != 0, conv2weight_1, conv2weight_2)
        server.conv2.weight.data.add_(serverconv2weight)
        
        conv2bias_1 = user_1.conv2.bias.data - temp_1.conv2.bias.data
        conv2bias_2 = user_2.conv2.bias.data - temp_2.conv2.bias.data
        F.dropout(conv2bias_1, p = 0.01, inplace = True)
        serverconv2bias = torch.where(conv2bias_1 != 0, conv2bias_1, conv2bias_2)
        server.conv2.bias.data.add_(serverconv2bias)
          
        fc1weight_1 = user_1.fc1.weight.data - temp_1.fc1.weight.data
        fc1weight_2 = user_2.fc1.weight.data - temp_2.fc1.weight.data
        F.dropout(fc1weight_1, p = 0.01, inplace = True)
        serverfc1weight = torch.where(fc1weight_1 != 0, fc1weight_1, fc1weight_2)
        server.fc1.weight.data.add_(serverfc1weight)
        
        fc1bias_1 = user_1.fc1.bias.data - temp_1.fc1.bias.data
        fc1bias_2 = user_2.fc1.bias.data - temp_2.fc1.bias.data
        F.dropout(fc1bias_1, p = 0.01, inplace = True)
        serverfc1bias = torch.where(fc1bias_1 != 0, fc1bias_1, fc1bias_2)
        server.fc1.bias.data.add_(serverfc1bias)
        
        fc2weight_1 = user_1.fc2.weight.data - temp_1.fc2.weight.data
        fc2weight_2 = user_2.fc2.weight.data - temp_2.fc2.weight.data
        F.dropout(fc2weight_1, p = 0.01, inplace = True)
        serverfc2weight = torch.where(fc2weight_1 != 0, fc2weight_1, fc2weight_2)
        server.fc2.weight.data.add_(serverfc2weight)
        
        fc2bias_1 = user_1.fc2.bias.data - temp_1.fc2.bias.data
        fc2bias_2 = user_2.fc2.bias.data - temp_2.fc2.bias.data
        F.dropout(fc2bias_1, p = 0.01, inplace = True)
        serverfc2bias = torch.where(fc2bias_1 != 0, fc2bias_1, fc2bias_2)
        server.fc2.bias.data.add_(serverfc2bias)
        
        temp_1.load_state_dict(user_1.state_dict())
        temp_2.load_state_dict(user_2.state_dict())
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step_1, loss_1.item(), loss_2.item()))   
            
    server.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = server(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print('Test Accuracy of the server on the 10000 test images: {} %'.format(100 * correct / total))
   
    user_1.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = user_1(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print('Test Accuracy of the user_1 on the 10000 test images: {} %'.format(100 * correct / total))
        
    user_2.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = user_2(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print('Test Accuracy of the user_2 on the 10000 test images: {} %'.format(100 * correct / total))
   