

   

import numpy as np

from TSpy.label import reorder_label

from TSpy.utils import calculate_scalar_velocity_list

class Time2State:

    def __init__(self, win_size, step, encoder_class, clustering_class, verbose=False):

        self.__win_size = win_size

        self.__step = step

        self.__offset = int(win_size/2)

        self.__encoder = encoder_class

        self.__clustering_component = clustering_class

    def set_step(self, step):

        self.__step = step

    def set_clustering_component(self, clustering_obj):

        self.__clustering_component = clustering_obj

        return self

    def fit_encoder(self, X):

        self.__encoder.fit(X)

        return self

                                                   

    def predict_without_encode(self, X, win_size, step):

        self.__cluster()

        self.__assign_label()

        self.__smooth()

        return self

    def predict(self, X, win_size, step):

        self.__length = X.shape[0]

        self.__encode(X, win_size, step)

        self.__cluster()

        self.__assign_label()

        self.__smooth()

        return self

    def fit(self, X, win_size, step):

        self.__length = X.shape[0]

        self.fit_encoder(X)

        self.__encode(X, win_size, step)

        self.__cluster()

        self.__assign_label()

        self.__smooth()

        self.__calculate_velocity()

        self.use_cps()

        return self

    def __encode(self, X, win_size, step):

        self.__embeddings = self.__encoder.encode(X, win_size, step)

    def __cluster(self):

        self.__embedding_label = reorder_label(self.__clustering_component.fit(self.__embeddings))

    def __assign_label(self):

        hight = len(set(self.__embedding_label))

                                                                                                         

        weight_vector = np.ones(shape=(2*self.__offset)).flatten()

        self.__state_seq = self.__embedding_label

        vote_matrix = np.zeros((self.__length,hight))

        i = 0

        for l in self.__embedding_label:

            vote_matrix[i:i+self.__win_size,l]+= weight_vector

            i+=self.__step

        self.__state_seq = np.array([np.argmax(row) for row in vote_matrix])

    def __calculate_velocity(self):

        self.__velocity = calculate_scalar_velocity_list(self.__embeddings, interval=1)

                                               

                                                                                         

                                                                                           

                             

                                                                                                   

                                          

                            

                                 

                                   

                                                    

                          

                                             

                         

                                        

                                        

                              

                                         

                                        

                        

                             

                                        

                             

                         

                      

        

                                                   

                                        

                                  

                                       

                                          

                                          

                                                                         

                                                                      

                         

                                         

              

                                

                            

                                 

                                   

                                                    

                          

                                             

                         

                                        

                                        

                                        

                                  

                                       

                                          

                                          

                                                                         

                                                                      

                         

    def use_cps(self):

        cut_list = self.find_potentional_cp()

        self.__embedding_label = self.bucket(self.__embedding_label, cut_list)

    

    def find_potentional_cp(self):

        threshold = np.mean(self.__velocity)

        idx = self.__velocity>=threshold

        pre = idx[0]

        cut_list = []

        for i, e in enumerate(idx):

            if e == pre:

                continue

            else:

                cut_list.append(i)

                pre = e

        self.__change_points = cut_list

        return cut_list

    def bucket(self, X, cut_points):

        result = np.array(X.shape, dtype=int)

        pre = cut_points[0]

        for cut in cut_points[1:]:

            sub_seq = X[pre:cut]

            label_set = list(set(sub_seq))

            vote_list = []

            for label in label_set:

                vote_list.append(len(np.argwhere(sub_seq==label)))

            max_idx = np.argmax(vote_list)

                                                       

            result[pre:cut]=label_set[max_idx]

            pre = cut

        return result

    def __smooth(self):

        return

                                

                                                          

                                                        

                                                     

                                                              

                                  

                               

                                                  

                                                                    

                                                   

                                                                                

                                                  

               

                                          

                                                                

                            

                                                                                              

    def save_encoder(self):

        pass

    def load_encoder(self):

        pass

    def save_result(self, path):

        pass

    def load_result(self, path):

        pass

    def plot(self, path):

        pass

    @property

    def embeddings(self):

        return self.__embeddings

    @property

    def state_seq(self):

        return self.__state_seq

    

    @property

    def embedding_label(self):

        return self.__embedding_label

    @property

    def velocity(self):

        return self.__velocity

    @property

    def change_points(self):

        return self.__change_points
