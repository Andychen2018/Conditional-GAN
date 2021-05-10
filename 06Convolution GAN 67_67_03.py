
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas
import pandas, numpy, random
import matplotlib.pyplot as plt
from pandas import Series


# In[2]:


# functions to generate random data
def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data

# size here must only be an integer
def generate_random_one_hot(size):
    label_tensor = torch.zeros((size))
    random_idx = random.randint(0,size-1)
    label_tensor[random_idx] = 1.0
    return label_tensor


# In[3]:


class View(nn.Module):
    def __init__(self,shape):
        super().__init__()
        self.shape = shape,
        
    def forward(self,x):
        return x.view(*self.shape)


# # Dataset Class

# In[4]:


class MnistDataset(Dataset):
    def __init__(self,csv_file):
        self.data_df = pandas.read_csv(csv_file,header = 0)
        pass
    
    def  __len__(self):
        return len(self.data_df)
    
    def __getitem__(self,index):
        label = self.data_df.iloc[index,0]

        target = torch.zeros((2))
        target[label] = 1.0
        #image_values = torch.FloatTensor(self.data_df.iloc[index,1:].values)/255.0
        image_values = torch.FloatTensor(self.data_df.iloc[index,1:4490].values)/6108.0
        #image_values = torch.cuda.FloatTensor(image_values).view(1,1,67,67)
        
        return label,image_values,target
    
    def plot_image(self,index):
        img = self.data_df.iloc[index,1:4490].values.reshape(67,67)
        plt.title("User" + str(self.data_df.index[index])+ "  Label="+ str(self.data_df.iloc[index,0]))
        #plt.title("label = " + str(self.data_df.iloc[index,0]))
        plt.imshow(img,interpolation='none',cmap = 'Blues')
        pass
    pass


# In[5]:


mnist_dataset = MnistDataset('./FullDataSet1.csv')
mnist_dataset.plot_image(4230)


# # Discriminator Network 构建鉴别器

# In[6]:


class Discriminator(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        #define neural network layers
        self.model = nn.Sequential(
             
            nn.Linear(4489+2, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
           
        
        )
        
        self.loss_function = nn.BCELoss()
        
        self.optimiser = torch.optim.Adam(self.parameters(),lr = 0.0001)
        
        self.counter = 0;
        self.progress = []
        pass
    
    def forward(self, image_tensor, label_tensor):
        # combine seed and label
        inputs = torch.cat((image_tensor, label_tensor))
        return self.model(inputs)
    
    
    def train(self, inputs, label_tensor, targets):
        # calculate the output of the network
        outputs = self.forward(inputs, label_tensor)
        
        # calculate loss
        loss = self.loss_function(outputs, targets)

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 4320 == 0):
            print("counter = ", self.counter)
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass
    
    
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        pass
    
    pass  


# # Test Discriminator

# In[7]:


get_ipython().run_cell_magic('time', '', '# test discriminator can separate real data from random noise\n\nD = Discriminator()\n#D.to(device)\n\nfor label, image_data_tensor, label_tensor in mnist_dataset:\n    # real data\n    D.train(image_data_tensor, label_tensor, torch.FloatTensor([1.0]))\n    # fake data\n    D.train(generate_random_image(4489), generate_random_one_hot(2), torch.FloatTensor([0.0]))\n    pass')


# In[8]:


D.plot_progress()


# In[9]:


# manually run discriminator to check it can tell real data from fake

for i in range(4):
  label, image_data_tensor, label_tensor = mnist_dataset[random.randint(0,4320)]
  print( D.forward( image_data_tensor, label_tensor ).item() )
  pass

for i in range(4):
  print( D.forward( generate_random_image(4489), generate_random_one_hot(2) ).item() )
  pass


# # Generator Network

# In[10]:


# generator class

class Generator(nn.Module):
    
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(100+2, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 4489),
            nn.Sigmoid()
        )
        
        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        # counter and accumulator for progress
        self.counter = 0;
        self.progress = []
        
        pass
    
    
    def forward(self, seed_tensor, label_tensor):        
        # combine seed and label
        inputs = torch.cat((seed_tensor, label_tensor))
        return self.model(inputs)


    def train(self, D, inputs, label_tensor, targets):
        # calculate the output of the network
        g_output = self.forward(inputs, label_tensor)
        
        # pass onto Discriminator
        d_output = D.forward(g_output, label_tensor)
        
        # calculate error
        loss = D.loss_function(d_output, targets)

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass
    
    def plot_images(self, label):
        label_tensor = torch.zeros((2))
        label_tensor[label] = 1.0
        # plot a 3 column, 2 row array of sample images
        f, axarr = plt.subplots(2,3, figsize=(16,8))
        for i in range(2):
            for j in range(3):
                axarr[i,j].imshow(G.forward(generate_random_seed(100), label_tensor).detach().cpu().numpy().reshape(67,67), interpolation='none', cmap='Blues')
                pass
            pass
        pass
    
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        pass
    
    pass


# # Test Generator Output

# In[11]:


# check the generator output is of the right type and shape

G = Generator()

output = G.forward(generate_random_seed(100), generate_random_one_hot(2))

img = output.detach().numpy().reshape(67,67)

plt.imshow(img, interpolation='none', cmap='Blues')


# # Train GAN

# In[12]:


# train Discriminator and Generator

D = Discriminator()
G = Generator()
# D.to(device)
# G.to(device)


# In[13]:


get_ipython().run_cell_magic('time', '', '\n# train Discriminator and Generator\n\nepochs = 100\n\nfor epoch in range(epochs):\n  print ("epoch = ", epoch + 1)\n\n  # train Discriminator and Generator\n\n  for label, image_data_tensor, label_tensor in mnist_dataset:\n    # train discriminator on true\n    D.train(image_data_tensor, label_tensor, torch.FloatTensor([1.0]))\n\n    # random 1-hot label for generator\n    random_label = generate_random_one_hot(2)\n    \n    # train discriminator on false\n    # use detach() so gradients in G are not calculated\n    D.train(G.forward(generate_random_seed(100), random_label).detach(), random_label, torch.FloatTensor([0.0]))\n    \n    # different random 1-hot label for generator\n    random_label = generate_random_one_hot(2)\n\n    # train generator\n    G.train(D, generate_random_seed(100), random_label, torch.FloatTensor([1.0]))\n\n    pass\n    \n  pass')


# In[14]:


# plot discriminator error

D.plot_progress()


# In[15]:


# plot generator error

G.plot_progress()


# In[16]:


# plot several outputs from the trained generator

G.plot_images(1)


# In[17]:


# plot several outputs from the trained generator

G.plot_images(0)


# In[38]:


output1 = G.forward(generate_random_seed(100),torch.tensor([1., 0.]))


# In[39]:


output0 = G.forward(generate_random_seed(100),torch.tensor([0., 1.]))


# In[35]:


output0.reshape(67,67)


# In[41]:


x = (output1-output0).reshape(67,67)


# In[42]:


x

