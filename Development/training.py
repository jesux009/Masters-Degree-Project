import torch
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from BasicConvNet import BasicConvNet
from Nets.UNetV2 import UNet
from data_handling import ContrailsDataset
from validation import validate, generate_loss_plots
import torch.autograd.profiler as profiler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main(reduceLR=False, pretrained=False, select_device=False, soft_label=True, train_batch_size=100, learning_rate=0.001, epoch=5, accum_iter=1):

    loss_evolution = []
    val_evolution = []
    train_evolution = []
    PA_evolution = []
    IoU_evolution = []
    precision_evolution = []
    recall_evolution = []
    F1_evolution = []
    dice_evolution = []

    # Move to GPU if needed 
    if select_device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    print(f"Using device: {device.upper()}")

    # Definition of the network
    net = UNet(in_channels=3, first_out_channels=64, exit_channels=1, padding=0, levels=3).to(device)

    # Load weights from pre-trained model of one epoch
    if pretrained:
        pretrained_model_path = r"C:\Users\USER\Desktop\UNIVERSIDAD\MÁSTER AERONÁUTICA - UC3M\SEGUNDO\TRABAJO DE FIN DE MÁSTER\Development\Nets\Trained\UNET_SLPositives_2203_3epoch.pth"
        weigths = torch.load(pretrained_model_path)
        net.load_state_dict(weigths)

    # Load the dataset      
    trainset = ContrailsDataset(path=r'D:\GOES-16 Dataset\SingleFrame_PNG', use='train', soft_labels=soft_label)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)

    # Optimize with SGD and BCE
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
    # Learning rate scheduler
    if reduceLR:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Initialize statistics
    start_time = time.time()
    running_loss = 0

    # Training loop
    print('Beginning training...', end='\n')
    for e in range(epoch):
        for i, data in enumerate(trainloader):
            # Obtaining the inputs
            inputs, labels = data
            if inputs is None or labels is None:
                continue
            else:
                # Move inputs and labels to CUDA
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    # Pad the inputs for the network to match the output shape
                    inputs = F.pad(inputs,(23,23,23,23), mode='reflect')
                    # Apply forward propagation
                    outputs = net(inputs)
                    # Calculate the loss function
                    loss = criterion(outputs, labels)
                    loss = loss / accum_iter
                    # Apply backpropagation
                    loss.backward()
                    running_loss += loss.item()
                    # scheduler.step(loss.item())

                    # Apply a step of SGD after several gradient accumulation steps
                    if ((i + 1) % accum_iter == 0) or (i + 1 == len(trainloader)):
                        optimizer.step()

            loss_evolution.append(loss.item())
            # Calculate statistics
            time_spent = time.time() - start_time
            remaining_batches = len(trainloader)*epoch - len(trainloader)*e - i - 1
            time_remaining = remaining_batches*(time_spent/(len(trainloader)*e+i+1))
            time_remaining_hours = int(np.floor(time_remaining / 3600))
            time_remaining_minutes = int(np.floor((time_remaining - time_remaining_hours*3600)/60))

            print(f'Epoch {e+1}/{epoch}, batch {i+1}/{len(trainloader)} with loss {loss_evolution[-1]/train_batch_size}. Remaining training time: {time_remaining_hours}hr {time_remaining_minutes}min.  ', end='\r')

        name = f'UNET_SLPositives_0104_{e+1}epoch.pth'
        torch.save(net.state_dict(), name)

        print(f'\n Average running loss per example from training in epoch {e+1}: {running_loss/len(trainset)}', end='\n')
        train_evolution.append(running_loss/len(trainloader))
        running_loss = 0

        print(f'\n Beginning cross-validation in epoch {e+1}', end='\n')
        val_loss, PA, IoU, precision, recall, F1, dice = validate(net, device=device, example=False, usage='cross-validate', test_batch_size=train_batch_size)

        print(f'\n Average running loss per example from cross validation in epoch {e+1}: {val_loss}', end='\n')
        print(f'\n Pixel accuracy: {PA}', end='\n')
        print(f'\n Jaccard Index: {IoU}', end='\n')
        print(f'\n Precision: {precision}', end='\n')
        print(f'\n Recall: {recall}', end='\n')
        print(f'\n F1 Score: {F1}', end='\n')
        print(f'\n Dice coefficient: {dice}', end='\n')
        PA_evolution.append(PA)
        IoU_evolution.append(IoU)
        precision_evolution.append(precision)
        recall_evolution.append(recall)
        F1_evolution.append(F1)
        dice_evolution.append(dice)
        val_evolution.append(val_loss)

        # Perform a step in the learning rate scheduler
        if reduceLR:
            scheduler.step(val_loss)
    
    print('\n Finished training.', end='\n')

    generate_loss_plots(loss_evolution, val_evolution, train_evolution, PA_evolution, IoU_evolution, recall_evolution, precision_evolution, F1_evolution, dice_evolution)

    print('\n Performing final cross-validation...', end='\n')
    validate(net, device=device, example=4, usage='cross-validation', test_batch_size=train_batch_size)
    

if __name__ == '__main__':
    main(epoch=1, learning_rate=0.01, accum_iter=5, train_batch_size=25, select_device=True)