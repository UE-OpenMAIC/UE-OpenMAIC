import torch

import numpy

import math

import numpy as np

import torch

def hanning_tensor(X):

    length = X.size(2)

    weight = (1-np.cos(2*math.pi*np.arange(length)/length))/2

    weight = torch.tensor(weight, dtype=X.dtype, device=X.device)

    return weight*X

                        

                         

                              

                                 

                                                                          

                                 

                                                        

                                                          

                                   

                            

class LSELoss(torch.nn.modules.loss._Loss):

    

       

    def __init__(self, win_size, M, N, win_type):

        super(LSELoss, self).__init__()

        self.win_size = win_size

        self.win_type = win_type

        self.M = M

        self.N = N

                               

                      

                          

    def forward(self, batch, encoder, save_memory=False):

        M = self.M

        N = self.N

        length_pos_neg=self.win_size

        

        total_length = batch.size(2)

        center_list = []

        loss1 = 0

        for i in range(M):

            random_pos = numpy.random.randint(0, high=total_length - length_pos_neg*2 + 1, size=1)

            rand_samples = [batch[0,:, i: i+length_pos_neg] for i in range(random_pos[0],random_pos[0]+N)]

                               

            if self.win_type == 'hanning':

                embeddings = encoder(hanning_tensor(torch.stack(rand_samples)))

            else:

                embeddings = encoder(torch.stack(rand_samples))

                                     

            size_representation = embeddings.size(1)

            for i in range(N):

                for j in range(N):

                    if j<=i:

                        continue

                    else:

                                                                                        

                                                                            

                                                                                       

                        loss1 += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(

                            embeddings[i].view(1, 1, size_representation),

                            embeddings[j].view(1, size_representation, 1))))

            center = torch.mean(embeddings, dim=0)

            center_list.append(center)

        

        loss2=0

        for i in range(M):

            for j in range(M):

                if j<=i:

                    continue

                                                                                 

                                                                     

                                                                                

                loss2 += -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(

                    center_list[i].view(1, 1, size_representation),

                    center_list[j].view(1, size_representation, 1))))

        loss = loss1/(M*N*(N-1)/2) + loss2/(M*(M-1)/2)

                                  

        return loss
