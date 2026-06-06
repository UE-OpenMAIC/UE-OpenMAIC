import torch

import numpy

import math

import numpy as np

import torch

def hanning_tensor(X):

    length = X.size(2)

    weight = (1-np.cos(2*math.pi*np.arange(length)/length))/2

    weight = torch.tensor(weight)

    return weight.cuda()*X

                        

                         

                              

                                 

                                                                          

                                 

                                                        

                                                          

                                   

                            

class LSELoss(torch.nn.modules.loss._Loss):

    

       

    def __init__(self, compared_length, nb_random_samples, negative_penalty, M, N):

        super(LSELoss, self).__init__()

        self.compared_length = compared_length

        if self.compared_length is None:

            self.compared_length = numpy.inf

        self.nb_random_samples = nb_random_samples

        self.negative_penalty = negative_penalty

        self.M = M

        self.N = N

                               

        self.tau = 1

                          

    def forward(self, batch, encoder, train, save_memory=False):

                                                                   

        M = self.M

        N = self.N

        length_pos_neg=self.compared_length

        

        total_length = batch.size(2)

                                                          

        center_list = []

        loss1 = 0

        for i in range(M):

            random_pos = numpy.random.randint(0, high=total_length - length_pos_neg*2 + 1, size=self.nb_random_samples)

            rand_samples = [batch[0,:, i: i+length_pos_neg] for i in range(random_pos[0],random_pos[0]+N)]

                               

            embeddings = encoder(hanning_tensor(torch.stack(rand_samples)))

                                                             

                                     

            size_representation = embeddings.size(1)

            for i in range(N):

                for j in range(N):

                    if j<=i:

                        continue

                    else:

                        loss1 += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(

                            embeddings[i].view(1, 1, size_representation),

                            embeddings[j].view(1, size_representation, 1))/self.tau))

            center = torch.mean(embeddings, dim=0)

            center_list.append(center)

        

        loss2=0

        for i in range(M):

            for j in range(M):

                if j<=i:

                    continue

                loss2 += -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(

                    center_list[i].view(1, 1, size_representation),

                    center_list[j].view(1, size_representation, 1))/self.tau))

        loss = loss1/(M*N*(N-1)/2) + loss2/(M*(M-1)/2)

                                  

        return loss
