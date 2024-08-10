import sys
import os
import pathlib
from pathlib import Path
import numpy as np
import time
import datetime
import ast
import csv
from torch import nn 
import torch
import beam_model
import pickle
from PIL import Image
from beam_model import Model
import random

device = torch.device("cuda")

model = Model().to(device)

#data parameters
Timeall = 499
Epochs = 2000
Trajectories = 20
epoch_loss = []
save_loss = []
rollout = []
result = []
Pad = 1

#train or eval
Train = True
Eval = False


#data file names
file_path_training = 'train.pickle'
file_path_valid = 'valid.pickle'


with open(file_path_training, 'rb') as file:
    dataset_main = pickle.load(file)

with open(file_path_valid, 'rb') as file:
    validation = pickle.load(file)

validation = dataset_main[0]
validation = [validation]

xpos = np.array(dataset_main[0][0]['world_pos'][:,0])
nodenum = len(xpos)
t_max = len(dataset_main[0])

def create_accel(dataset,trajectory):

    xpos_all = []
    ypos_all = []

    for t in range(t_max):
        xpos_t = np.array(dataset[trajectory][t]['world_pos'][:, 0])
        ypos_t = np.array(dataset[trajectory][t]['world_pos'][:, 1])

        xpos_all.append(xpos_t)
        ypos_all.append(ypos_t)

    xpos_array = np.column_stack(xpos_all)
    ypos_array = np.column_stack(ypos_all)

    dt = np.linspace(0,t_max-1,t_max)

    vx = np.zeros((nodenum,t_max))
    ax = np.zeros((nodenum,t_max))

    vy = np.zeros((nodenum,t_max))
    ay = np.zeros((nodenum,t_max))

    for i in range(nodenum):
        vx[i, :] = np.gradient(xpos_array[i, :], dt)
        ax[i, :] = np.gradient(vx[i, :], dt)

        vy[i, :] = np.gradient(ypos_array[i, :], dt)
        ay[i, :] = np.gradient(vy[i, :], dt)
    
    return ax, ay

ax,ay = create_accel(dataset_main,19)
axmax = np.max(ax)
axmin = np.min(ax)
aymax = np.max(ay)
aymin = np.min(ay)

ayn = np.zeros((nodenum,t_max))
axn = np.zeros((nodenum,t_max))

for n in range(nodenum):
    axn[n,:] = (ax[n,:] - axmin) / (axmax - axmin)
    ayn[n,:] = (ay[n,:] - aymin) / (aymax - aymin)

ymax = np.max(np.array(dataset_main[19][498]['world_pos'][:,1]))
youngmax = 10000000


#takes take a dataset and creates an image from the data at the inputed trajectory and timestep
def make_img(trajectory,timestep,dataset):

    xpos = np.array(dataset[trajectory][timestep]['world_pos'][:,0])
    ypos = np.array(dataset[trajectory][timestep]['world_pos'][:,1])
    youngs = np.array(dataset[trajectory][timestep]['youngs'][:])

    ayn = np.zeros((nodenum,t_max))
    axn = np.zeros((nodenum,t_max))

    for n in range(nodenum):
        axn[n,:] = (ax[n,:] - axmin) / (axmax - axmin)
        ayn[n,:] = (ay[n,:] - aymin) / (aymax - aymin)


    def map_rgb(val):
        return (val*255)

    
    yposn = ypos/ymax
    youngsn = youngs/youngmax

    rp = map_rgb(xpos)
    gp = map_rgb(yposn)

    ra = map_rgb(axn[:,timestep])
    ga = map_rgb(ayn[:,timestep])

    b = map_rgb(youngsn)
    b = np.squeeze(b.T)

    r = np.append(rp,ra)
    g = np.append(gp,ga)
    b = np.append(b,b)


    rgb = np.stack((r, g, b), axis=-1)

    length = len(r)
    side_length = int(np.ceil(np.sqrt(length)))

    # Padding the array to make it square if necessary
    padded_rgb = np.zeros((side_length * side_length, 3), dtype=np.uint8)
    padded_rgb[:length, :] = rgb
    Pad = len(padded_rgb) - len(r)
    

    # Reshape the padded array into a square image
    image_array = padded_rgb.reshape((side_length, side_length, 3))

    # Create an image using PIL
    image = Image.fromarray(image_array)


    return image


#takes an input of an image in tensor form and returns the data to the origional format
def rev_img(output):

    #print(output.size())
    #sys.exit()
    rollout = torch.squeeze(output)
    rollout = rollout.detach().cpu().numpy()
    rollout = rollout.reshape(3,-1)


    r = rollout[0]/255
    g = rollout[1]/255
    b = rollout[2]/255

    pad = 8 

    r = r[:-pad]
    g = g[:-pad]
    b = b[:-pad]

    nodenum = int(len(r)/2)

    axnr = r[nodenum :]
    aynr = g[nodenum :]


    axr = (axnr * (axmax - axmin)) + axmin
    ayr = (aynr * (aymax - aymin)) + aymin


    return axr,ayr
    


#converts a img object to a rgb tensor
def img_to_tensor(img):
    nparray = np.array(img)
    tensor = torch.from_numpy(nparray)
    tensor = tensor.float()
    return tensor


#loss function takes input of network outputed prediction as well as the ground truth and previous tensor used to create the prediction
def loss_fn(dataset,trajectory,timestep,output):
    
    ax,ay = create_accel(dataset,trajectory)
    ax = torch.tensor(ax[:,timestep], requires_grad=True)
    ay = torch.tensor(ay[:,timestep], requires_grad=True)
    at = torch.cat((ax,ay))

    axr,ayr = rev_img(output)
    ar = np.concatenate((axr,ayr))
    ar = torch.tensor(ar, requires_grad=True)

    #print('real')
    #print(len(ax))
    #print('output')
    #print(len(axr))

    error1 = torch.sum((ar - at) ** 2)

    error = error1

    return error


                        
optimzer = torch.optim.Adam(model.parameters(), lr = 0.1, weight_decay = 1e-4)

#learner input number of epochs, time steps in each trajectory and number of trajectories 
def learner(epochs,timeall,trajectories):

    #print(trajectories.shape)
    #sys.exit()
    for epoch in range(epochs):
        print("Epoch:", epoch)
        #if epoch > 1:
            #print("Loss:", loss,"at epoch:", epoch)
        epoch_loss = []
        for trajectory in range(trajectories):
            optimzer.zero_grad()

            #print("Trajectory:", trajectory)
            nsample = 40
            itersample = random.sample(range(timeall),nsample)
            save_error= 0
            for iter in itersample:

                old = make_img(trajectory,int(iter),dataset_main)
                old = img_to_tensor(old)
                old = old.unsqueeze(0)
                old = old.permute(0,3,1,2)
                output = model(old.to(device))

                error = loss_fn(dataset_main,trajectory,int(iter),output)
                save_error = error + save_error
            loss = save_error*1000
            epoch_loss.append(loss)
            #print(save_error)
            loss.backward()
            #print('Trajectory loss', loss)
            optimzer.step()
        epoch_loss = sum(epoch_loss)
        print('Epoch Loss', epoch_loss)

#evaluator takes input of validation dataset and total time steps 
def evaluator(Validation,timeall):

    #access the output file
    path = os.path.join(os.getcwd(), 'output')
    os.chdir(path)

    #loads trained model
    trained_model = Model().to(device)
    trained_model.load_state_dict(torch.load('model_updated.pth')) #CHANGE TRAINED MODEL NAME HERE
    trained_model.eval()

    #saves the intial time step to the rollout data (I think I may have made a mistake with the formating here as the plot of the first data point is not showing correctly)
    start_img = make_img(0,0,Validation)
    old = img_to_tensor(start_img)
    save = rev_img(old)

    old = old.unsqueeze(0)
    old = old.permute(0,3,1,2)

    rollout.append(save)

    #loops over timesteps inputing the network output into itself for timeall 
    for time in range(timeall):
        new = trained_model(old.to(device))
        save = rev_img(new.cpu())
        rollout.append(save)
        old = new

    return rollout

    
#Code for training the model
if Train:
       
    learner(Epochs,Timeall,Trajectories)


    path = os.path.join(os.getcwd(), 'output')
    os.chdir(path)

    torch.save(model.state_dict(), 'model_updated.pth')

    with open('epoch_loss.pickle', 'wb') as f:
        pickle.dump(epoch_loss,f) 
    print("Finished Training")

    
#Code for evaluating the model 
if Eval:

    rollout = evaluator(validation,Timeall)
    

    with open('rollout_updated.pickle', 'wb') as f:
        pickle.dump(rollout,f)    
    print('Finished Eval')  

