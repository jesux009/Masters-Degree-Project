import torch
import numpy as np
import os
from data_handling import img2tensor
from BasicConvNet import BasicConvNet
from Nets.UNetV2 import UNet
from data_handling import ContrailsDataset
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn


def generate_examples(example, net, threshold):
     print(f'Generating example images...', end='\n')

     fig, axs = plt.subplots(example, 3, figsize=(30, 8))

     # Take a set of 5 random images with contrails to visualize the network performance
     images_path = r'D:\GOES-16 Dataset\SingleFrame_PNG\validation\images'
     positives_fnames = [filename.split("\\")[3] for filename in np.load(r'C:\Users\USER\Desktop\UNIVERSIDAD\MÁSTER AERONÁUTICA - UC3M\SEGUNDO\TRABAJO DE FIN DE MÁSTER\Development\Inspection\positive_validation.npy')]
     positives_fnames = np.random.choice(positives_fnames, size=example, replace=False)
     # Locate their corresponding labels
     labels_path = r'D:\GOES-16 Dataset\SingleFrame_PNG\validation\ground_truth'

     for i, fname in enumerate(positives_fnames):
          img_path = os.path.join(images_path, fname + '.png')
          lbl_path = os.path.join(labels_path, fname + '.npy')
          image = np.array(Image.open(img_path))
          input_tensor = img2tensor(image/255)
          input_tensor = input_tensor.unsqueeze(0)
          input_tensor = F.pad(input_tensor,(11,11,11,11), mode='reflect')
          outputs = torch.sigmoid(net(input_tensor)).view(1,256,256)
          outputs = (outputs > threshold).float()
          labels = img2tensor(np.load(lbl_path))
          
          axs[i][0].imshow(image)          
          axs[i][1].imshow(labels[0,:,:], cmap='gray_r')
          axs[i][1].set_title(fname) 
          axs[i][2].imshow(outputs[0,:,:], cmap='gray_r')

     plt.tight_layout()  
     plt.show() 



def validate(net, usage, device, test_batch_size=50, threshold=0.5, example=4):    

     testset = ContrailsDataset(path=r'D:\GOES-16 Dataset\SingleFrame_PNG', use=usage)
     testloader = torch.utils.data.DataLoader(testset,batch_size=test_batch_size, shuffle=True, num_workers=0)

     criterion = nn.BCEWithLogitsLoss()

     # Positive pixels labelled as positive
     TP = 0
     # Negative pixels labelled as negative
     TN = 0
     # Positive pixels labelled as negative
     FN = 0
     # Negative pixels labelled as positive
     FP = 0

     running_loss = 0

     with torch.no_grad():
          for i, data in enumerate(testloader):
               images, labels = data
               images, labels = images.to(device), labels.to(device)
               images = F.pad(images,(23,23,23,23), mode='reflect')
               outputs = torch.sigmoid(net(images)).view(-1,1,256,256)
               binary_outputs = (outputs > threshold).float()

               loss = criterion(outputs, labels)
               running_loss += loss.item()

               TP += torch.sum((binary_outputs == 1) & (labels == 1)).item()
               TN += torch.sum((binary_outputs == 0) & (labels == 0)).item()
               FN += torch.sum((binary_outputs == 0) & (labels == 1)).item()
               FP += torch.sum((binary_outputs == 1) & (labels == 0)).item()

               print(f'Processing batch {i+1}/{len(testloader)}', end='\r')

     # Pixel Accuracy
     PA = TP/(TP+TN+FP+FN)
     # Jaccard Coefficient
     IoU = TP/(TP+FP+FN)
     # Precision
     precision = TP/(TP+FP)
     # Recall
     recall = TP/(TP+FN)
     # F1 Score
     F1 = 2*(precision*recall/(precision+recall))
     # Dice Coefficient
     dice = 2*TP/(2*TP+FP+FN)

     if example != False:
          generate_examples(example, net, threshold)

     return running_loss/len(testset), PA, IoU, precision, recall, F1, dice
          

def generate_loss_plots(loss_evolution, val_evolution, train_evolution, PA_evolution, IoU_evolution, recall_evolution, precision_evolution, F1_evolution, dice_evolution):
     plt.plot(range(len(loss_evolution)), loss_evolution)
     plt.xlabel('Mini-batch')
     plt.ylabel('Batch loss')
     plt.title('Running loss for each training mini-batch')
     plt.grid(True)
     plt.show()

     plt.plot(range(len(PA_evolution)), PA_evolution)
     plt.xlabel('Epoch')
     plt.ylabel('Precision Accuracy [-]')
     plt.title('Precision accuracy')
     plt.grid(True)
     plt.show()
     
     plt.plot(range(len(IoU_evolution)), IoU_evolution)
     plt.xlabel('Epoch')
     plt.ylabel('Jaccard Index [-]')
     plt.title('Jaccard Index')
     plt.grid(True)
     plt.show()

     plt.plot(range(len(recall_evolution)), recall_evolution)
     plt.xlabel('Epoch')
     plt.ylabel('Recall [-]')
     plt.title('Recall')
     plt.grid(True)
     plt.show()

     plt.plot(range(len(precision_evolution)), precision_evolution)
     plt.xlabel('Epoch')
     plt.ylabel('Precision [-]')
     plt.title('Precision')
     plt.grid(True)
     plt.show()

     plt.plot(range(len(F1_evolution)), F1_evolution)
     plt.xlabel('Epoch')
     plt.ylabel('F1 Score [-]')
     plt.title('F1 Score')
     plt.grid(True)
     plt.show()

     plt.plot(range(len(dice_evolution)), dice_evolution)
     plt.xlabel('Epoch')
     plt.ylabel('Dice Coefficient [-]')
     plt.title('Dice Coefficient')
     plt.grid(True)
     plt.show()

     plt.plot(range(len(val_evolution)), val_evolution, label='Validation')  
     plt.plot(range(len(train_evolution)), train_evolution, label='Training')  
     plt.xlabel('Epoch')
     plt.ylabel('Loss per mini-batch')
     plt.legend()
     plt.title('Loss across epochs')
     plt.grid(True)
     plt.show()