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


device = torch.device("cuda")

model = Model().to(device)

#data parameters
Timeall = 400
Epochs = 200
Trajectories = 20
epoch_loss = []
save_loss = []
rollout = []
result = []

#train or eval
Train = False
Eval = True


#data file names
file_path_training = 'train.pickle'
file_path_valid = 'valid.pickle'


with open(file_path_training, 'rb') as file:
    dataset_main = pickle.load(file)

with open(file_path_valid, 'rb') as file:
    validation = pickle.load(file)




#takes take a dataset and creates an image from the data at the inputed trajectory and timestep
def make_img(trajectory,timestep,dataset):

    #normalizing
    ymax = np.max(np.array(dataset_main[19][498]['world_pos'][:,1]))
    youngmax = 10000000

    xpos = np.array(dataset[trajectory][timestep]['world_pos'][:,0])
    ypos = np.array(dataset[trajectory][timestep]['world_pos'][:,1])
    youngs = np.array(dataset[trajectory][timestep]['youngs'][:])

    #covnerting from 0-1 scale to 0-255 rgb scale
    def map_rgb(val):
        return (val*255).astype(int)


    yposn = ypos/ymax
    youngsn = youngs/youngmax

    r = map_rgb(xpos)
    g = map_rgb(yposn)
    b = map_rgb(youngsn)
    b = np.squeeze(b.T)

    #padding vectors to make 100 data points for 10x10 image
    red = np.pad(r,(0,100-len(r)))
    green = np.pad(g,(0,100-len(g)))
    blue = np.pad(b,(0,100-len(b)))

    red = red.reshape(10,10)
    green = green.reshape(10,10)
    blue = blue.reshape(10,10)
    rgb_image = np.stack([red, green, blue], axis=-1)

    #creating image
    image = Image.fromarray(rgb_image.astype('uint8'), 'RGB')
    return image 


#takes an input of an image in tensor form and returns the data to the origional format
def rev_img(output):

    #normalization values
    ymax = np.max(np.array(dataset_main[19][498]['world_pos'][:,1]))
    youngmax = 10000000

    #formating image tensor data
    rollout = torch.squeeze(output)
    rollout = rollout.detach().numpy()
    rollout = rollout.reshape(3,-1)
    
    #extracting rgb values
    r = rollout[0]
    g = rollout[1]
    b = rollout[2]

    #removing paded values
    r = r[:-6]
    g = g[:-6]
    b = b[:-6]

    #undoing normalization
    x = r / 255
    y = g / 255 * ymax
    youngs = b * 255 * youngmax 

    #storing results
    result = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), youngs.reshape(-1, 1)), axis=1)


    return result
    


#converts a img object to a rgb tensor
def img_to_tensor(img):
    nparray = np.array(img)
    tensor = torch.from_numpy(nparray)
    tensor = tensor.float()
    return tensor


#loss function takes input of network outputed prediction as well as the ground truth and previous tensor used to create the prediction
def loss_fn(prev_tensor,network_tensor,true_img,time,trajectory,save_loss):
    
    #converting true data to same format as network output
    true_tensor = img_to_tensor(true_img)
    true_tensor = true_tensor.unsqueeze(0)
    true_tensor = true_tensor.permute(0,3,1,2)

    #print(network_tensor.shape)
    #print(true_tensor)

    #velocity values
    vel_true = (true_tensor - prev_tensor)
    vel_pred = (network_tensor - prev_tensor)

    #difference in position and velocity
    error1 = torch.sum((true_tensor/255 - network_tensor/255) ** 2, dim=1)
    error2 = torch.sum((vel_true/255 - vel_pred/255) ** 2, dim=1)

    true_tensor = true_tensor.squeeze() 
    print(true_tensor.shape)


    print(torch.mean(true_tensor[:,:,0]))
    print(torch.mean(true_tensor[:,:,1]))
    print(torch.mean(true_tensor[:,:,2]))

    print(torch.std(true_tensor[:,:,0]))
    print(torch.std(true_tensor[:,:,1]))
    print(torch.std(true_tensor[:,:,2]))
   
    #sum or errors weights can be added later
    error = error1 + error2 

    #mean of the error as loss
    loss = torch.mean(error)
    
    #saveing loss for all timesteps and all trajectories in an epoch
    save_loss.append(loss)
    
    #prints sum of loos over an epoch at end of each training epoch
    if trajectory > Trajectories - 2:
        if time > Timeall - 2:
            print("Epoch Loss: ",sum(save_loss))
            epoch_loss.append(sum(save_loss))
            #restting stored loss
            save_loss = []
    return loss


                        
optimzer = torch.optim.Adam(model.parameters(), lr = 0.0001)

#learner input number of epochs, time steps in each trajectory and number of trajectories 
def learner(epochs,timeall,trajectories):

    for epoch in range(epochs):
        print("Epoch:", epoch)
        #if epoch > 1:
            #print("Loss:", loss,"at epoch:", epoch)
        for trajectory in range(trajectories):
            #print("Trajectory:", trajectory)
            for time in range(timeall):
                optimzer.zero_grad()
                #if time is even take current timestep as prev and next at target if odd take current as target and the previous time step as previous
                if time % 2 == 0: 
                    prev_img = make_img(trajectory,time,dataset_main)
                    targ_img = make_img(trajectory,time+1,dataset_main)
                else: 
                    targ_img = make_img(trajectory,time,dataset_main)
                    prev_img = make_img(trajectory,time-1,dataset_main)

                #formating data for network input and running through the network
                prev_tensor = img_to_tensor(prev_img)
                prev_tensor = prev_tensor.unsqueeze(0)
                prev_tensor = prev_tensor.permute(0,3,1,2)
                pred_tensor = model(prev_tensor.to(device))
                pred_tensor = pred_tensor.cpu()
                loss = loss_fn(prev_tensor,pred_tensor,targ_img,time,trajectory,save_loss)
                loss.backward()
                optimzer.step()

#evaluator takes input of validation dataset and total time steps 
def evaluator(Validation,timeall):

    #access the output file
    path = os.path.join(os.getcwd(), 'output')
    os.chdir(path)

    #loads trained model
    trained_model = Model().to(device)
    trained_model.load_state_dict(torch.load('model_pos_loss_only.pth'))
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

    torch.save(model.state_dict(), 'model.pth')

    with open('epoch_loss.pickle', 'wb') as f:
        pickle.dump(epoch_loss,f) 

    
#Code for evaluating the model 
if Eval:

    rollout = evaluator(validation,Timeall)
    

    with open('rollout.pickle', 'wb') as f:
        pickle.dump(rollout,f)      

