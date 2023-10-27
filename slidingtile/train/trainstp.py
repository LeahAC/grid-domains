import numpy as np
#import keras
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
import random
from tensorflow.keras.callbacks import ModelCheckpoint
from twodattention import AttentionAugmentation2D
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, Input, concatenate, Lambda, Multiply, Softmax, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import time
import math
import copy
import tensorflow as tf

dim = 5

class AddPositionalEncoding(Layer):
    """
    Injects positional encoding signal described in section 3.5 of the original
    paper "Attention is all you need". Also a base class for more complex
    coordinate encoding described in "Universal Transformers".
    """

    def __init__(self,dim, **kwargs):
        self.dim = dim
        self.signal = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def build(self, input_shape):
        _, height, width, channels = input_shape
        print(height,width,channels,self.dim)
        self.signal = posencode2d(
            self.dim, self.dim, channels)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.signal

def posencode2d(height,width,channels):
    """
    :param channels: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: height*width*channels position matrix
    """
    if channels % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(channels))
    pe = np.zeros((height, width, channels), dtype=K.floatx())
    channels = int(channels / 2) # 4
    div_term = K.exp(K.arange(0., channels, 2) *
                         -(math.log(10000.0) / channels))
    #print(div_term)
    pos_w = K.expand_dims(K.arange(0., width),1)
    pos_h = K.expand_dims(K.arange(0., height),1)

    pe[:, :, 0:channels:2] = K.repeat_elements(K.expand_dims(K.sin(pos_h * div_term),1),width, 1)
    pe[:, :, 1:channels:2] = K.repeat_elements(K.expand_dims(K.cos(pos_h * div_term),1),width, 1)
    pe[:, :, channels::2] = K.repeat_elements(K.expand_dims(K.sin(pos_w * div_term),0),height, 0)
    pe[:, :, channels + 1::2] =K.repeat_elements(K.expand_dims(K.cos(pos_w * div_term),0),height, 0)
    return pe

def to_categorical_tensor(x3d,dim) :#portal is a dictionary
      x1d = x3d.ravel()
      y1d = to_categorical(x1d, 25)
      y4d = y1d.reshape([dim, dim, 25])
      

      return y4d

def get_data():
    all_states=[]
    all_actions= []
    f=open("stile.txt", "r")
    g=open("stileact.txt", "r") #1 - tile up, 2 - tile down, 3 - tile left, 4 - tile right
    array_s=[]
    array_a=[]
    duplicate = []
    index = []

    i=0
    for line in f:
        array_s.append([int(x) for x in line.split()])
        #print(i)
        if array_s[i] not in duplicate:
            arr=np.asarray(array_s[i])
            temp=np.reshape(arr, (5,5))
            all_states.append(temp)
            duplicate.append(array_s[i])
            index.append(i)

        i+=1
    i = 0
    for line in g:
        if i in index:
           #print(i)
           array_a.append([int(x) for x in line.split()])
        i+=1



    return all_states , array_a
    f.close()
    g.close()

class DNN:
    def __init__(self,dim):

        self.model        = self.create_model(dim)

    def create_model(self,dim):

        inputA = Input(shape=(None,None,25))
        inputB = Input(shape=(None,None,25))

        inp =  concatenate([inputA, inputB], axis=3)

        b = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv1')(inp)
        b1 = concatenate([b, inp], axis=3)
        
        c = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv2')(b1)
        c1 = concatenate([c, inp], axis=3)
       
        d = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv3')(c1)
        d1 = concatenate([d, inp], axis=3)
       
        e = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv4')(d1)
        e1 = concatenate([e, inp], axis=3)
        
        f = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv5')(e1)
        f1 = concatenate([f, inp], axis=3)        

        g = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv6')(f1)
        g1 = concatenate([g, inp], axis=3)
        
        h = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv7')(g1)
        h1 = concatenate([h, inp], axis=3)
        
        ########################################################################################
        i=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv8-1')(inp)
        att8 = AttentionAugmentation2D(60,60,2,dim)(i)
        att8 = concatenate([att8, AddPositionalEncoding(dim)(i), inp], axis=3)

        j=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv9-1')(att8)
        att9 = AttentionAugmentation2D(60,60,2,dim)(j)
        att9 = concatenate([att9, AddPositionalEncoding(dim)(j), inp], axis=3)

        k=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv10-1')(att9)
        att10 = AttentionAugmentation2D(60,60,2,dim)(k)
        att10 = concatenate([att10, AddPositionalEncoding(dim)(k), inp], axis=3)
        
        l=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv11-1')(att10)
        att11 = AttentionAugmentation2D(60,60,2,dim)(l)
        att11 = concatenate([att11, AddPositionalEncoding(dim)(l),  inp], axis=3)

        f1=GlobalAveragePooling2D()(att11)
        d1 = Dense(256,  activation='relu', name = 'dense-1')(f1)
        op1 = Dense(4, activation='softmax', name = 'op-1')(d1)
        ###############################################################################
        p=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv8-2')(h1)
        att15 = AttentionAugmentation2D(60,60,2,dim)(p)
        att15 = concatenate([att15,AddPositionalEncoding(dim)(p), inp], axis=3)

        q=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv9-2')(att15)
        att16 = AttentionAugmentation2D(60,60,2,dim)(q)
        att16 = concatenate([att16,AddPositionalEncoding(dim)(q), inp], axis=3)

        r=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv10-2')(att16)
        att17 = AttentionAugmentation2D(60,60,2,dim)(r)
        att17 = concatenate([att17,AddPositionalEncoding(dim)(r),inp], axis=3)


        s=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv11-2')(att17)
        att18 = AttentionAugmentation2D(60,60,2,dim)(s)
        att18 = concatenate([att18,AddPositionalEncoding(dim)(s), inp], axis=3)
 
        f2=GlobalAveragePooling2D()(att18)
        d2 = Dense(256,  activation='relu', name = 'dense-2')(f2)
        op2 = Dense(1,  name = 'op-2')(d2)
        model = Model([inputA,inputB], [op1,op2])
        model.compile(optimizer='adam',
                     loss=['categorical_crossentropy','mse'])
        return model
        
def find_goal_state():
        arr = [[0 for i in range(dim)] for j in range(dim)]
        k = 1
        for row in range(0, dim):
            for col in range(0, dim):
                if row == dim - 1 and col == dim - 1:
                    arr[row][col] == 0
                    continue
                arr[row][col] = k
                k = k + 1
        return (to_categorical_tensor(np.array(arr), dim)), np.array(arr)        

def main():
    print(tf. __version__ )
    dqn = DNN(5)
    dim = 5
    #Get train data
    all_states, all_actions = get_data() 
    #print(len(all_states),len(all_actions))#,len(all_statesval), len(all_actionsval))
    goal_state, _ = find_goal_state()
    X1_Train = []
    X2_Train = []
    Y1_Train = []
    Y2_Train = []
    tot_solved = 0
    done = False
    solved=0
    print("Data Loaded...")

    for i in range(2000):
        print("step",i)
        state = all_states[i]
        full_len = len(all_actions[i])

        if len(all_actions[i])>=0:
            for stat_len in range(len(all_actions[i])):
                #print("new", state,"\n")
                find_blank = np.where(state == 0)
                blank = list(zip(find_blank[0], find_blank[1]))[0]
                copy_state = copy.deepcopy(state)
                X1_Train.append(to_categorical_tensor(state,dim))
                X2_Train.append(goal_state)
                y_temp = [0,0,0,0]
                y_temp[all_actions[i][stat_len]-1] = 1
                #print(all_actions[i][stat_len])
                Y1_Train.append(np.array(y_temp))
                Y2_Train.append(np.array(full_len))

                full_len-=1
                #print(all_actions[i][stat_len])
                if all_actions[i][stat_len] == 2:#tile up
                    copy_state[blank[0]][blank[1]] = copy_state[blank[0]+1][blank[1]]
                    copy_state[blank[0]+1][blank[1]] = 0
                    #temp[0] = 1
                    #Y1_Train.append(temp)

                if all_actions[i][stat_len] == 1:#tile down
                   copy_state[blank[0]][blank[1]] = copy_state[blank[0]-1][blank[1]]
                   copy_state[blank[0]-1][blank[1]] = 0
                   #temp[1] = 1
                   #Y1_Train.append(temp)

                if all_actions[i][stat_len] == 4:#tile left
                  copy_state[blank[0]][blank[1]] = copy_state[blank[0]][blank[1]+1]
                  copy_state[blank[0]][blank[1]+1] = 0
                  #temp[2] = 1
                  #Y1_Train.append(temp)

                if all_actions[i][stat_len] == 3:#tile right
                    copy_state[blank[0]][blank[1]] = copy_state[blank[0]][blank[1]-1]
                    copy_state[blank[0]][blank[1]-1] = 0
                    #temp[3] = 1

                state = copy_state
            #print("final_state ",state,"\n")
            


    X1_Train = np.array(X1_Train)
    X2_Train = np.array(X2_Train)
    Y1_Train = np.array(Y1_Train)
    Y2_Train = np.array(Y2_Train)
    
    filepath = 'stile'
    check = ModelCheckpoint(
    filepath=filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
    callbacks_list = [check]
    history = dqn.model.fit([X1_Train,X2_Train], [Y1_Train,Y2_Train], epochs=10, batch_size=100,callbacks=callbacks_list)
    #dqn.model.save_weights('stp1.h5')




if __name__ == "__main__":
    main()
