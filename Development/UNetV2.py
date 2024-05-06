import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as T
import numpy as np

# Generalized UNet architecture from scratch for learning purposes. See https://arxiv.org/abs/1505.04597

class CNNBlock(nn.Module):

     # In the UNet architecture, this is the smallest possible block.
     # Each of these applies a 2D convolution, normalizes the output and passes it through a ReLU activation function.
     # The result of passing an input of size [N, in, H_in, W_in] through this layer is an output [N, out, H_out, W_out]
     #    where H_out = (H_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1 
     #          W_out = (W_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1 
     # This output corresponds to the element_wise result of the ReLU activation function of the normalized convolution output.

     # The block is used several times at each "level", which is why we later define a CNNSet, which is composed of several 
     # usages of the CNNBlock sequentially 

     """
     Parameters:
     in_channels (int): Number of channels in the input to the block.
     out_channels (int): Number of channels produced by the convolution of the input within the block.
     kernel_size (int): Size of the convolving kernel. Default = 3.
     stride (int) : Stride of the convolution. Default = 1.
     padding (int) : Padding added to all four sides of the input. Default = 0.
     """

     def __init__(self, in_channels: int, out_channels:int, kernel_size:int=3, stride:int=1, padding:int=0):
          super(CNNBlock, self).__init__()

          self.sequential_block = nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True)
          )

     def forward(self, x):
          x = self.sequential_block(x)
          return x
     


class Encoder(nn.Module):

     # The encoder part of the UNet consists of the downsampling of the initial input through the application of 
     # a sequential series of convolution sets followed by a max pooling layer. The complete operation's objective 
     # is to capture the context and spatial information of the input image at different scales.

     """
     Parameters:
     in_channels (int): Number of input channels of the first CNNSet.
     out_channels (int): Number of output channels of the first CNNSet.
     padding (int): Padding applied in each convolution.
     levels (int): Number times a CNNSet + MaxPool2D layer is applied.
     """

     def __init__(self, input_channels:int, output_channels:int, pool_kernelsize: int, parameters: list):
          super(Encoder, self).__init__()
          self.encoder_layers = nn.ModuleList()
          levels = len(parameters)
          for level in range(levels-1):
               for conv in range(len(parameters[level])):
                    conv_kernelsize = parameters[level][conv][0]
                    conv_stride = parameters[level][conv][1]
                    self.encoder_layers.append(CNNBlock(in_channels=input_channels, out_channels=output_channels, kernel_size=conv_kernelsize, stride=conv_stride))
                    input_channels = output_channels
               output_channels *= 2
               self.encoder_layers.append(nn.MaxPool2d(pool_kernelsize))
          # A final convolution set is applied after all the levels, commonly referred to as the bottleneck
          for conv in range(len(parameters[-1])):
               conv_kernelsize = parameters[-1][conv][0]
               conv_stride = parameters[-1][conv][1]
               self.encoder_layers.append(CNNBlock(in_channels=input_channels, out_channels=output_channels, kernel_size=conv_kernelsize, stride=conv_stride))
               input_channels = output_channels

     def forward(self,x):
          residual_connection = []
          for i, layer in enumerate(self.encoder_layers):
               x = layer(x)
               # After the set CNN is processed, the result is logged to be sent in a connection to the decoder
               if i<len(self.encoder_layers)-1 and isinstance(self.encoder_layers[i+1], nn.MaxPool2d):
                    residual_connection.append(x)
               # If the processed layer is a pooling operation, the result is not logged
          return x, residual_connection



class Decoder(nn.Module):

     """
     Parameters:
     in_channels (int): Number of input channels of the first up-convolution layer.
     out_channels (int): Number of output channels of the first up-convolution layer.
     padding (int): Padding applied in each convolution.
     levels (int): number times an up-convolution + CNNSet is applied.
     """

     # After the encoder has downsampled the information, the decoder now applies an upsampling to match to original features.
     # This is achieved combining up-convolutions followed by convolution sets sequentially, achieving a recovery of the 
     # fine-grained spatial information lost during the downsampling in the encoder.

     def __init__(self, input_channels:int, exit_channels:int, uppool_kernelsize:int, parameters: list):
          super(Decoder, self).__init__()
          self.exit_channels = exit_channels
          self.decoder_layers = nn.ModuleList()

          levels = len(parameters)
          for level in range(levels-1):
               output_channels = int(input_channels/2)
               self.decoder_layers.append(nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=uppool_kernelsize, stride=uppool_kernelsize))
               for conv in range(len(parameters[level])):
                    conv_kernelsize = parameters[level][conv][0]
                    conv_stride = parameters[level][conv][1]
                    self.decoder_layers.append(CNNBlock(in_channels=input_channels, out_channels=output_channels, kernel_size=conv_kernelsize, stride=conv_stride))
                    input_channels = output_channels
          # A final convolution set without the ReLU activation function since the output will be passed through a BCELoss 
          self.decoder_layers.append(nn.Conv2d(in_channels=input_channels, out_channels=exit_channels, kernel_size=1))

     def forward(self, x, residual_connection):
          for i, layer in enumerate(self.decoder_layers):
               # After the previous output is up-sampled, the connection from the equivalent level is concatenated
               if i>0 and isinstance(self.decoder_layers[i-1], nn.ConvTranspose2d):
                    # First we center-crop the route tensor to make the size match
                    residual_connection[-1] = T.center_crop(residual_connection[-1], x.shape[2])
                    # Then we concatenate the tensors in the dimensions of the channels
                    x = torch.cat([x, residual_connection.pop(-1)], dim=1)
                    x = layer(x)
               # If the processed layer is an up-convolution operation, the connection is not performed
               else:
                    x = layer(x)
          return x

class UNetV2(nn.Module):

     """
     Parameters:
     in_channels (int): Number of input channels.
     first_out_channels (int): Number of output channels of the first convolution set.
     exit_channels (int): Number of output channels.
     levels (int): Number of levels for the encoder-decoder architecture.
     padding (int): Padding applied in each convolution operation.
     """

     # After the encoder has downsampled the information, the decoder now applies an upsampling to match to original features.
     # This is achieved combining up-convolutions followed by convolution sets sequentially, achieving a recovery of the 
     # fine-grained spatial information lost during the downsampling in the encoder.

     def __init__(self, in_channels, first_out_channels, exit_channels, pool_kernelsize, down_parameters, up_parameters):
          super(UNetV2, self).__init__()
          levels = len(down_parameters)
          self.encoder = Encoder(input_channels=in_channels, output_channels=first_out_channels, pool_kernelsize=pool_kernelsize, parameters=down_parameters)
          self.decoder = Decoder(input_channels=first_out_channels*(2**(levels-1)), exit_channels=exit_channels, uppool_kernelsize=pool_kernelsize, parameters=up_parameters)

     def forward(self, x):
          encoder_out, residuals = self.encoder(x)
          decoder_out = self.decoder(encoder_out, residuals)
          return T.center_crop(decoder_out, (256,256))