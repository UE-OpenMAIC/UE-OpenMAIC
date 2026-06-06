                                                            

                                                              

                                                       

                                                            

                                                   

                                                            

                                                            

                                              

                                                            

                                                             

                                                        

                                                           

                                                         

                    

                                                              

                                                                            

                             

             

                                  

                                                                              

                                                                               

                                                                              

                                                                           

                                                                       

                                                          

                                                                                

                                                 

                                                                            

                                                                          

                                                                             

                                                                        

                                                                               

                                                                               

           

import torch

class Chomp1d(torch.nn.Module):

    

       

    def __init__(self, chomp_size):

        super(Chomp1d, self).__init__()

        self.chomp_size = chomp_size

    def forward(self, x):

        return x[:, :, :-self.chomp_size]

class SqueezeChannels(torch.nn.Module):

    

       

    def __init__(self):

        super(SqueezeChannels, self).__init__()

    def forward(self, x):

        return x.squeeze(2)

class CausalConvolutionBlock(torch.nn.Module):

    

       

    def __init__(self, in_channels, out_channels, kernel_size, dilation,

                 final=False):

        super(CausalConvolutionBlock, self).__init__()

                                                                           

        padding = (kernel_size - 1) * dilation

                                  

        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(

            in_channels, out_channels, kernel_size,

            padding=padding, dilation=dilation

        ))

                                                     

        chomp1 = Chomp1d(padding)

        relu1 = torch.nn.LeakyReLU()

                                   

        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(

            out_channels, out_channels, kernel_size,

            padding=padding, dilation=dilation

        ))

        chomp2 = Chomp1d(padding)

        relu2 = torch.nn.LeakyReLU()

                        

        self.causal = torch.nn.Sequential(

            conv1, chomp1, relu1, conv2, chomp2, relu2

        )

                             

        self.upordownsample = torch.nn.Conv1d(

            in_channels, out_channels, 1

        ) if in_channels != out_channels else None

                                   

        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):

        out_causal = self.causal(x)

        res = x if self.upordownsample is None else self.upordownsample(x)

        if self.relu is None:

            return out_causal + res

        else:

            return self.relu(out_causal + res)

class CausalCNN(torch.nn.Module):

    

       

    def __init__(self, in_channels, channels, depth, out_channels,

                 kernel_size):

        super(CausalCNN, self).__init__()

        layers = []                                     

        dilation_size = 1                         

        for i in range(depth):

            in_channels_block = in_channels if i == 0 else channels

            layers += [CausalConvolutionBlock(

                in_channels_block, channels, kernel_size, dilation_size

            )]

            dilation_size *= 2                                          

                    

        layers += [CausalConvolutionBlock(

            channels, out_channels, kernel_size, dilation_size

        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)

class CausalCNNEncoder(torch.nn.Module):

    

       

    def __init__(self, in_channels, channels, depth, reduced_size,

                 out_channels, kernel_size):

        super(CausalCNNEncoder, self).__init__()

        causal_cnn = CausalCNN(

            in_channels, channels, depth, reduced_size, kernel_size

        )

        reduce_size = torch.nn.AdaptiveMaxPool1d(1)

        squeeze = SqueezeChannels()                                       

        linear = torch.nn.Linear(reduced_size, out_channels)

        self.network = torch.nn.Sequential(

            causal_cnn, reduce_size, squeeze, linear

        )

    def forward(self, x):

        return self.network(x)
