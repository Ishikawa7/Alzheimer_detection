# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:32:22 2021

@author: della
"""

# =============================================================================
# Import librerie
# =============================================================================

import os 

import tensorflow as tp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Dropout

import numpy             as np
import pandas            as pd
import tkinter           as tk 


from datetime                import datetime
from sklearn.model_selection import train_test_split 
from tkinter                 import filedialog
from PIL                     import Image
from sklearn.metrics import confusion_matrix


from tensorflow.keras.models import model_from_json 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

import time

########################################################################
#Data

# Tk è una libreria che gestisce delle semplici interfacce
root = tk.Tk()
root.directory =  filedialog.askdirectory(title="Choose train folder")
dir_train=root.directory
root.directory =  filedialog.askdirectory(initialdir = os.path.dirname(dir_train), title="Choose test folder")
dir_test=root.directory
root.directory =  filedialog.askdirectory(initialdir = os.path.dirname(dir_train), title="Choose working folder")
dir_work=root.directory
root.destroy()

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Aggiungere f(x) che divide tr-ts


Class_Count = 4 

start_time = time.time()

def formatta_tempo(secondi):
    h = secondi // 3600
    m = (secondi % 3600) // 60
    s = secondi % 60
    return f"{int(h)} ore {int(m)} minuti {s:.2f} secondi"

# Stampa l'orario di partenza
orario_partenza = time.strftime("%H:%M:%S", time.localtime(start_time))
print(f"Orario di partenza: {orario_partenza}")

# =============================================================================
# Create file log
# =============================================================================

def print_log(model, work_dir, time_, train_x, train_y,val_x, val_y, test_x, test_y ):
    
    _, acc = model.evaluate(test_x, test_y)
    filepath_log = '{}/CNN_log_{}({:.2f}%).txt'.format(dir_work,time_, acc*100 )
    with open(filepath_log, 'w') as file_log:
        file_log.write('Net architecture:\n')
        
        model.summary(print_fn=lambda x: file_log.write(x + '\n'))
        
        file_log.write('\n')
        file_log.write('\n')
        
        for layer  in model.layers:
            try:
                file_log.write(f'kernel and filters: {layer.kernel_size}, {layer.filters}\n')
            except:
                try:
                    file_log.write(f'pool size: {layer.pool_size}\n')
                except:
                    try:
                        file_log.write(f'units of dense layer: {layer.units}\n')
                    except:
                        try:
                            file_log.write(f'dropout rate: {layer.rate}\n')
                        except:
                            pass
        
        file_log.write('\n')
        file_log.write('Training results:\n')
        loss, acc = model.evaluate(train_x, train_y)        
        file_log.write('loss: {:.3f}\n'.format(loss))
        file_log.write('w.acc: {:.3f}\n'.format(acc*100))
        file_log.write('\n')
        
        file_log.write('Validation results:\n')
        loss, acc = model.evaluate(val_x, val_y)        
        file_log.write('loss: {:.3f}\n'.format(loss))
        file_log.write('w.acc: {:.3f}\n'.format(acc*100))
        file_log.write('\n')
        
        file_log.write('Testing results:\n')
        loss, acc = model.evaluate(test_x, test_y)        
        file_log.write('loss: {:.3f}\n'.format(loss))
        file_log.write('w.acc: {:.3f}\n'.format(acc*100))
        
        test_pred = model.predict(test_x)
        y_pred = np.argmax(test_pred, axis =1)
        y_test = np.argmax(test_y, axis =1)
        file_log.write('\n')
        file_log.write('Testing set confusion matrix\n')
        file_log.write(str(confusion_matrix(y_test, y_pred)))


########################################################################
#create dataset 

def dataset(dir_train,dir_test):
    '''

    Parameters
    ----------
    dir_train : cartella in cui si trova il training set
        
    dir_test : cartella in cui si trova il test set
        

    Returns
    -------
    train_images : files training in un unico array di numeri
    test_images : files testing in un unico array di numeri
        
    height : altezza immagini
    width : larghezza immagini
    num_channels : num canali delle immagini
    
    Questa funzione carica in memoria le immagini, le apre leggendo i valori RGB o grayscale,
    le carica in un array multidimensionale e le ritorna.

    '''
    train_image_file_names = [i for i in os.listdir(dir_train)]
    test_image_file_names  = [i for i in os.listdir(dir_test)]
    
    train_images_p = []
    test_images_p  = []
       
    os.chdir(dir_train) #cambio directory
    for i in range(len(train_image_file_names)):
        im = Image.open(train_image_file_names[i])
        train_images_p.append(im.copy())
        im.close()
        if i % 20 == 0:
            print('train image {} processed'.format(i))
        
    os.chdir(dir_test)
    for i in range(len(test_image_file_names)):
        im = Image.open(test_image_file_names[i])
        test_images_p.append(Image.open(test_image_file_names[i]))
        im.close()
        if i % 20 == 0:
            print('test image {} processed'.format(i))
            
    train_images = np.array([np.array(image_data) for image_data in train_images_p])
    height       = train_images.shape[1]
    width        = train_images.shape[2]
    num_channels = train_images.shape[3]
    #new_dim      = height*width*num_channels
    #train_images = train_images.reshape((-1, new_dim))
    
    del train_images_p
    
    test_images  = np.array([np.array(image_data) for image_data in test_images_p])
    #test_images  = test_images.reshape((-1, new_dim))
    
    del test_images_p
    
    return train_images, test_images, height, width, num_channels
    
 
    
def labels(dir_work):
    root = tk.Tk()
    root.filename =  filedialog.askopenfilename(initialdir = dir_work,
                                                title = "Select Train Target file",
                                                filetypes = (("nna files","*.nna"),
                                                             ("all files","*.*")))
    df = pd.read_csv(root.filename, sep="\t",header=None)
    t_train     = np.asarray(df)
    num_classes = np.shape(t_train)[1]
    directory = os.path.basename(root.filename)

    root.filename =  filedialog.askopenfilename(initialdir = directory,
                                                title = "Select Test Target file",
                                                filetypes = (("nna files","*.nna"),
                                                             ("all files","*.*")))
    df     = pd.read_csv(root.filename, sep="\t", header=None)
    t_test = np.asarray(df)
    
    root.destroy()
    
    return t_train, t_test, num_classes


train_x, test_x_t, img_height, img_width, num_channels = dataset(dir_train, dir_test)
train_y, test_y_t, num_classes = labels(dir_work)

#creation validation set (opzionale)
validation = 0

if validation:
    test_x, val_x, test_y, val_y = train_test_split(test_x_t, test_y_t, test_size = 0.2, random_state = Class_Count)

    del test_x_t, test_y_t
else:
    val_x = test_x_t
    val_y = test_y_t
    test_x = test_x_t
    test_y = test_y_t
    del test_x_t, test_y_t 


K.clear_session()
#create model
model = Sequential()
#add model layers - RP Architettura
model.add(Conv2D(16, kernel_size=1, activation="relu", input_shape=(img_height, img_width, num_channels)))
model.add(Conv2D(16, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=3, activation="relu")) #3
model.add(Conv2D(32, kernel_size=3, activation="relu")) #3
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation="relu")) #3
model.add(Conv2D(64, kernel_size=3, activation="relu")) #3
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, activation="relu")) #3
model.add(Conv2D((96), kernel_size=3, activation="relu")) #3
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, kernel_size=3, activation="relu")) #3
#model.add(Conv2D(128, kernel_size=3, activation="relu")) #3
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(15, kernel_size=3,  padding = 'SAME', activation="relu")) #3
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(10, kernel_size=3, activation="relu")) #3
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(15, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(Class_Count, activation="softmax"))

#compile model using accuracy to measure model performance

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'],run_eagerly=True)

os.chdir(dir_work) 
time_ = datetime.now().strftime('%Y-%m-%d_%H%M')
filepath = '{}/CNN_{}.hdf5'.format(dir_work,time_ )

saveBest = ModelCheckpoint(filepath, 
                             monitor='val_categorical_accuracy',
                             mode='max',
                             save_best_only=True,
                             verbose=0)
#RUN della rete
model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size = 256,epochs=1, callbacks=[saveBest]) # epochs=200 batch_size=32

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Tempo di esecuzione: {elapsed_time} secondi")
print(f"Tempo di esecuzione (in ore, minuti e secondi): {formatta_tempo(elapsed_time)}")



model.load_weights(filepath)
model.evaluate(test_x, test_y)




print_log(model, dir_work, time_, train_x, train_y,val_x, val_y, test_x, test_y )   


#   
# =============================================================================
# GET FIRST DENSE LAYER + NNR
# =============================================================================
# with a Sequential model
print(model.layers)
get_dense_layer = K.function([model.layers[0].input],[model.layers[-5].output])
get_last_layer = K.function([model.layers[0].input],[model.layers[-1].output])


x = train_x
layer_output = get_dense_layer([x])[0]

tr = np.concatenate([layer_output,train_y], axis = 1)
tr = pd.DataFrame(tr)
M,N = tr.shape
with open(f'training({M}x{N-num_classes}x{num_classes})_{time_}.nna','w') as file:
    tr.to_csv(file, header = False, index = False, sep = '\t', line_terminator='\n')
    
x = test_x
layer_output = get_dense_layer([x])[0]

ts = np.concatenate([layer_output,test_y], axis = 1)
ts = pd.DataFrame(ts)
M,N = ts.shape
with open(f'testing({M}x{N-num_classes}x{num_classes})_{time_}.nna','w') as file:
    ts.to_csv(file, header = False, index = False, sep = '\t', line_terminator='\n')    
    
x = test_x
layer_output = get_last_layer([x])[0]

ts = np.concatenate([test_y,layer_output], axis = 1)
ts = pd.DataFrame(ts)
M,N = ts.shape
with open(f'[NNR]testing({M}x{N-num_classes}x{num_classes})_{time_}.nnr','w') as file:
    ts.to_csv(file, header = False, index = False, sep = '\t', line_terminator='\n')    
    

minmax = tr.append(ts)
M,N = minmax.shape
with open(f'minmax({M}x{N-num_classes}x{num_classes})_{time_}.nna','w') as file:
    minmax.to_csv(file, header = False, index = False, sep = '\t', line_terminator='\n')    
   
    


# =============================================================================
# SAVE MODEL
# =============================================================================
#serialize model to JSON

model_json = model.to_json()
with open(f"CNN_model_{time_}.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5/

model.save_weights(f"CNN_weights_{time_}.h5")
print("Saved model to disk")
#later... load json and create model


# =============================================================================
# RELOAD MODEL
# =============================================================================
# Tk è una libreria che gestisce delle semplici interfacce
file_chose = tk.Tk()
file_chose.filename =  filedialog.askopenfilename(initialdir = os.getcwd(), 
                                                  title="Select JSON file",
                                                  filetypes = (("json files","*.json"),
                                                               ("all files","*.*")))
file_json_to_open=file_chose.filename
file_chose.filename =  filedialog.askopenfilename(initialdir = os.getcwd(), 
                                                  title="Select h5 file",
                                                  filetypes = (("h5 files","*.h5"),
                                                               ("all files","*.*")))
file_h5_to_open=file_chose.filename
 
file_chose.destroy()

work_dir = os.path.dirname(os.path.realpath(file_json_to_open))
os.chdir(work_dir)
json_file = open(file_json_to_open, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weights into new model
work_dir = os.path.dirname(os.path.realpath(file_h5_to_open))
os.chdir(work_dir)
loaded_model.load_weights(file_h5_to_open)
print("Loaded model from disk")

for layer  in loaded_model.layers:
    try:
        print(f'kernel and filters: {layer.kernel_size}, {layer.filters}\n')
    except:
        try:
            print(f'pool size: {layer.pool_size}\n')
        except:
            try:
                print(f'units of dense layer: {layer.units}\n')
            except:
                try:
                    print(f'dropout rate: {layer.rate}\n')
                except:
                    pass



#evaluate loaded model on test data

loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
score = loaded_model.evaluate(test_x, test_y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
   




#se si vuole rieseguire del codice precedente: 
model = loaded_model
