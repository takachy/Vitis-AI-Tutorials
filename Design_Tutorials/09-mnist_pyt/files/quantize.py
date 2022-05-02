'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''


'''
Simple PyTorch MNIST example - quantization
'''

'''
Author: Mark Harvey, Xilinx inc
'''

import os
import sys
import argparse
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer
from common import *


DIVIDER = '-----------------------------------------'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_picture(): #return trainsets, trainsets_small, testsets, testsets_small
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainsets = list()
    trainsets_small = list()
    # for i in range(1, 6):
    #     d = unpickle("./data/cifar-10-batches-py/data_batch_{0}".format(i))
    #     for j in range(len(d[b"data"])):
    #         np_image = d[b"data"][j].reshape(3, 32, 32)
    #         np_image = np_image.transpose([1, 2, 0])
    #         image = Image.fromarray(np_image, mode="RGB")
    #         resized_image = np.array(image.resize((224, 224)))
    #         resized_image = resized_image.transpose([2, 0, 1])
    #         # d[b"labels"][j] = change_label[d[b"labels"][j]]
    #         trainsets.append([resized_image, d[b"labels"][j]])
    #         if d[b"labels"][j] not in set(range(8, 10)):
    #             trainsets_small.append([resized_image, d[b"labels"][j]])
    # print("Finish make train-data")
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    testsets = list()
    testsets_small = list()
    d = unpickle("./data/cifar-10-batches-py/test_batch")
    for j in range(len(d[b"data"])):
        np_image = d[b"data"][j].reshape(3, 32, 32)
        np_image = np_image.transpose([1, 2, 0])
        image = Image.fromarray(np_image, mode="RGB")
        resized_image = np.array(image.resize((224, 224)))
        resized_image = resized_image.transpose([2, 0, 1])
        # d[b"labels"][j] = change_label[d[b"labels"][j]]
        # testsets.append([resized_image, d[b"labels"][j]])
        if d[b"labels"][j] not in set(range(8, 10)):
            testsets_small.append([resized_image, d[b"labels"][j]])
    print("Finish make test-data")
    return trainsets, trainsets_small, testsets, testsets_small




def quantize(build_dir,quant_mode,batchsize):

  dset_dir = build_dir + '/dataset'
  float_model = build_dir + '/float_model'
  quant_model = build_dir + '/quant_model'


  # use GPU if available   
  if (torch.cuda.device_count() > 0):
    print('You have',torch.cuda.device_count(),'CUDA devices available')
    for i in range(torch.cuda.device_count()):
      print(' Device',str(i),': ',torch.cuda.get_device_name(i))
    print('Selecting device 0..')
    device = torch.device('cuda:0')
  else:
    print('No CUDA devices available..selecting CPU')
    device = torch.device('cpu')

  # load trained model
  model = torchvision.models.resnet50()
  model.fc = torch.nn.Identity()
  # model = CNN()
  model = model.to(device)
  model.load_state_dict(torch.load(os.path.join(float_model,'my_f_model.pth')))

  # force to merge BN with CONV for better quantization accuracy
  optimize = 1

  # override batchsize if in test mode
  if (quant_mode=='test'):
    batchsize = 1
  
  rand_in = torch.randn([batchsize, 3, 224, 224])
  # rand_in = torch.randn([batchsize, 1, 28, 28])
  quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
  quantized_model = quantizer.quant_model


  # # # data loader
  # _, _, _, testsets_s = load_picture()
  # test_dataset = testsets_s
  # # test_dataset = torchvision.datasets.MNIST(dset_dir,
  # #                                           train=False, 
  # #                                           download=True,
  # #                                           transform=test_transform)

  # test_loader = torch.utils.data.DataLoader(test_dataset,
  #                                           batch_size=batchsize, 
  #                                           shuffle=False)

  # # evaluate 
  # test(quantized_model, device, test_loader)


  # export config
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if quant_mode == 'test':
    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  
  return



def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-b',  '--batchsize',  type=int, default=100,        help='Testing batchsize - must be an integer. Default is 100')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir    : ',args.build_dir)
  print ('--quant_mode   : ',args.quant_mode)
  print ('--batchsize    : ',args.batchsize)
  print(DIVIDER)

  quantize(args.build_dir,args.quant_mode,args.batchsize)

  return



if __name__ == '__main__':
    run_main()

