import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import pdb 
import random
from collections import OrderedDict
import torch.nn.functional as F
f = open('test.txt', 'w')
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 100
batch_size = 100
sample_dir = 'samples'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Image processing
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                     std=(0.5, 0.5, 0.5))])

# MNIST dataset
dataset_1 = torchvision.datasets.MNIST(root='C:/Users/SmartSpace/Desktop/mnist/mnist_data/',
                                   train=True,
                                   transform=transform,
                                   download=False)


idx_0 = dataset_1.train_labels==0 
idx_1 = dataset_1.train_labels==1
idx_2 = dataset_1.train_labels==2
idx_3 = dataset_1.train_labels==3
idx_4 = dataset_1.train_labels==4
idx_5 = dataset_1.train_labels==5
idx_6 = dataset_1.train_labels==6
idx_7 = dataset_1.train_labels==7
idx_8 = dataset_1.train_labels==8
idx_9 = dataset_1.train_labels==9
idx = idx_0
#print(idx.size())
dataset_1.train_labels = dataset_1.train_labels[idx]
#print(dataset_1.train_labels.size())
dataset_1.train_data = dataset_1.train_data[idx]
#print(dataset_1.train_data.size())

dataset_2 = torchvision.datasets.MNIST(root='C:/Users/SmartSpace/Desktop/mnist/mnist_data/',
                                   train=True,
                                   transform=transform,
                                   download=False)

idx_0 = dataset_2.train_labels==0 
idx_1 = dataset_2.train_labels==1
idx_2 = dataset_2.train_labels==2
idx_3 = dataset_2.train_labels==3
idx_4 = dataset_2.train_labels==4
idx_5 = dataset_2.train_labels==5
idx_6 = dataset_2.train_labels==6
idx_7 = dataset_2.train_labels==7
idx_8 = dataset_2.train_labels==8
idx_9 = dataset_2.train_labels==9
idx = idx_6
#print(idx.size())
dataset_2.train_labels = dataset_2.train_labels[idx]
#print(dataset_2.train_labels.size())
dataset_2.train_data = dataset_2.train_data[idx]
#print(dataset_2.train_data.size())
#pdb.set_trace()

# Data loader
data_loader_1 = torch.utils.data.DataLoader(dataset=dataset_1,
                                          batch_size=batch_size, 
                                          shuffle=True)
data_loader_2 = torch.utils.data.DataLoader(dataset=dataset_2,
                                          batch_size=batch_size, 
                                          shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='C:/Users/SmartSpace/Desktop/mnist/mnist_data',
                                          train=False, 
                                          transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Discriminator
D = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(image_size, hidden_size)),
    ('relu1', nn.LeakyReLU(0.2)),
    ('fc2', nn.Linear(hidden_size, hidden_size)),
    ('relu2', nn.LeakyReLU(0.2)),
    ('fc3', nn.Linear(hidden_size, 1)),
    ('out', nn.Sigmoid())
]))

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

# Device setting
D1 = D.to(device)
G = G.to(device)
D2 = D.to(device)
server = D.to(device)
temp_1 = D.to(device)
temp_2 = D.to(device)

temp_1.load_state_dict(D1.state_dict())
temp_2.load_state_dict(D2.state_dict())
server.load_state_dict(D1.state_dict())

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d1_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
d2_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
server_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d1_optimizer.zero_grad()
    g_optimizer.zero_grad()
    d2_optimizer.zero_grad()
    server_optimizer.zero_grad()
    
# Start training
total_step_1= len(data_loader_1)
total_step_2= len(data_loader_2)
print(total_step_1)
print(total_step_2)


for epoch in range(num_epochs):
    for i, ((images_1, labels_1), (images_2,labels_2)) in enumerate(zip(data_loader_1, data_loader_2)):
        
        #print(i)
        #print(labels_1)
        #print(labels_2)
        
        #save_image(denorm(images_1), os.path.join(sample_dir, 'real_images_1.png'))
        #save_image(denorm(images_2), os.path.join(sample_dir, 'real_images_2.png'))
        #pdb.set_trace()
        if(images_1.size()[0] == 100 and images_2.size()[0] == 100):
            images_1 = images_1.reshape(batch_size, -1).to(device)
            images_2 = images_2.reshape(batch_size, -1).to(device)
        else:
            break
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # ================================================================== #
        #                      Train the discriminator    1                   #
        # ================================================================== #
        
        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs_1 = D1(images_1)
        d1_loss_real = criterion(outputs_1, real_labels)
        real_score_1 = outputs_1
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs_1 = D1(fake_images)
        d1_loss_fake = criterion(outputs_1, fake_labels)
        fake_score_1 = outputs_1
        #print(outputs_1)
        # Backprop and optimize
        d1_loss = d1_loss_real + d1_loss_fake
        reset_grad()
        d1_loss.backward()
        d1_optimizer.step()
        
        
        
        # ================================================================== #
        #                      Train the discriminator   2                    #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs_2 = D2(images_2)
        d2_loss_real = criterion(outputs_2, real_labels)
        real_score_2 = outputs_2
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs_2 = D2(fake_images)
        d2_loss_fake = criterion(outputs_2, fake_labels)
        fake_score_2 = outputs_2
        
        # Backprop and optimize
        d2_loss = d2_loss_real + d2_loss_fake
        reset_grad()
        d2_loss.backward()
        d2_optimizer.step()
        
        #=======================================================================
        fc1weight_1 = D1.fc1.weight.data - temp_1.fc1.weight.data
        fc1weight_2 = D2.fc1.weight.data - temp_2.fc1.weight.data
        F.dropout(fc1weight_1, p = 0.01, inplace = True)
        serverfc1weight = torch.where(fc1weight_1 != 0, fc1weight_1, fc1weight_2)
        server.fc1.weight.data.add_(serverfc1weight)
        
        fc1bias_1 = D1.fc1.bias.data - temp_1.fc1.bias.data
        fc1bias_2 = D2.fc1.bias.data - temp_2.fc1.bias.data
        F.dropout(fc1bias_1, p = 0.01, inplace = True)
        serverfc1bias = torch.where(fc1bias_1 != 0, fc1bias_1, fc1bias_2)
        server.fc1.bias.data.add_(serverfc1bias)
        
        fc2weight_1 = D1.fc2.weight.data - temp_1.fc2.weight.data
        fc2weight_2 = D2.fc2.weight.data - temp_2.fc2.weight.data
        F.dropout(fc2weight_1, p = 0.01, inplace = True)
        serverfc2weight = torch.where(fc2weight_1 != 0, fc2weight_1, fc2weight_2)
        server.fc2.weight.data.add_(serverfc2weight)
        
        fc2bias_1 = D1.fc2.bias.data - temp_1.fc2.bias.data
        fc2bias_2 = D2.fc2.bias.data - temp_2.fc2.bias.data
        F.dropout(fc2bias_1, p = 0.01, inplace = True)
        serverfc2bias = torch.where(fc2bias_1 != 0, fc2bias_1, fc2bias_2)
        server.fc2.bias.data.add_(serverfc2bias)
        
        fc3weight_1 = D1.fc3.weight.data - temp_1.fc3.weight.data
        fc3weight_2 = D2.fc3.weight.data - temp_2.fc3.weight.data
        F.dropout(fc2weight_1, p = 0.01, inplace = True)
        serverfc3weight = torch.where(fc3weight_1 != 0, fc3weight_1, fc3weight_2)
        server.fc3.weight.data.add_(serverfc3weight)
        
        fc3bias_1 = D1.fc3.bias.data - temp_1.fc3.bias.data
        fc3bias_2 = D2.fc3.bias.data - temp_2.fc3.bias.data
        F.dropout(fc3bias_1, p = 0.01, inplace = True)
        serverfc3bias = torch.where(fc3bias_1 != 0, fc3bias_1, fc3bias_2)
        server.fc3.bias.data.add_(serverfc3bias)
        
        
        temp_1.load_state_dict(D1.state_dict())
        temp_2.load_state_dict(D2.state_dict())
        #=======================================================================
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        
        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        #outputs = (D1(fake_images) + D2(fake_images))/2
        outputs = server(fake_images) 
        #print(outputs)
        #pdb.set_trace()
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d1_loss: {:.4f}, g_loss: {:.4f}, D1(x): {:.2f}, D1(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step_1, d1_loss.item(), g_loss.item(), 
                          real_score_1.mean().item(), fake_score_1.mean().item()))
            print('Epoch [{}/{}], Step [{}/{}], d2_loss: {:.4f}, g_loss: {:.4f}, D2(x): {:.2f}, D2(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step_2, d2_loss.item(), g_loss.item(), 
                          real_score_2.mean().item(), fake_score_2.mean().item()))
    
    # Save real images
    if (epoch+1) == 1:
        images_1 = images_1.reshape(images_1.size(0), 1, 28, 28)
        save_image(denorm(images_1), os.path.join(sample_dir, 'real_images_1.png'))
        images_2 = images_2.reshape(images_2.size(0), 1, 28, 28)
        save_image(denorm(images_2), os.path.join(sample_dir, 'real_images_2.png'))
    
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

    f.write(str())
    #f.write(',')
    #f.write(str(d2_loss.item()))
    f.write('\n')
f.close()