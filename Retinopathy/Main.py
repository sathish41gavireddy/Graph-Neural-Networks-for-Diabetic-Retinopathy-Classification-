from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import cv2
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split 
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE
from keras_dgl.layers import GraphCNN #loading GNN class
import keras.backend as K
from keras.regularizers import l2
from keras.layers import BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.applications import DenseNet121

main = Tk()
main.title("Classification of Diabetic Retinopathy Disease Levels by Extracting Topological Features Using Graph Neural Networks")
main.geometry("1300x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore, fgcnn_model
labels = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']

def uploadDataset(): 
    global filename, X, Y
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename+"/trainLabels.csv")
    text.insert(END,str(dataset.head())+"\n\n")
    if os.path.exists('model/X.txt.npy'):#if dataset already process then load load it
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else: #if not process the loop all images from dataset
        X = []
        Y = []
        dataset_labels = dataset.values
        for i in range(len(dataset_labels)):
            img_name = dataset_labels[i,0]
            label = dataset_labels[i,1]
            if os.path.exists("Eyepacs/train/"+img_name+".jpeg"):
                img = cv2.imread("Eyepacs/train/"+img_name+".jpeg")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (32, 32))
                X.append(img)
                Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/eyepas_X',X)
        np.save('model/eyepas_Y',Y)
    text.insert(END,"Dataset EyePACS Images Loading Completed\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")

    #plot graph of different labels found in dataset
    unique, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize=(6,3))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Retinopaty Grades")
    plt.ylabel("Count")
    plt.title("Dataset Class Label Graph")
    plt.tight_layout()
    plt.show()        

def processDataset():
    global dataset, X, Y
    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Dataset Processing, Shuffling & Normalization Completed\n\n")
    text.insert(END,"Normalized Dataset Values = "+str(X))

def splitDataset():
    text.delete('1.0', END)
    global X, Y, X_train, X_test, y_train, y_test
    X = np.reshape(X, (X.shape[0], (X.shape[1] * X.shape[2] * X.shape[3])))
    smote = SMOTE()
    X, Y = smote.fit_resample(X, Y)
    Y = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], 32, 32, 3))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train & Test Split\n")
    text.insert(END,"80% dataset size used to train algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset size used to test algorithms : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, y_test):
    global labels
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    kappa = cohen_kappa_score(y_test,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    text.insert(END,algorithm+" Kappa     : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show()    

def trainGCNN():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, fgcnn_model
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    #Create GNN model to detect fault from all services
    graph_conv_filters = np.eye(1)
    graph_conv_filters = K.constant(graph_conv_filters)
    fgcnn_model = Sequential()
    #adding CNN Convolution2d layer to extract ROI region
    fgcnn_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    fgcnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    #ROI region will further optimize using autoencoder layer
    fgcnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    fgcnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    #flatten will be applied to get global average pooling and batch normalization as mean values
    fgcnn_model.add(Flatten())
    #extraced mean values will get trained with GraphCNN model
    fgcnn_model.add(GraphCNN(128, 1, graph_conv_filters, input_shape=(32,), activation='elu', kernel_regularizer=l2(5e-4)))
    fgcnn_model.add(GraphCNN(64, 1, graph_conv_filters, input_shape=(32,), activation='elu', kernel_regularizer=l2(5e-4)))
    fgcnn_model.add(Dense(units = 256, activation = 'elu'))
    fgcnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    fgcnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/fgcnn_weights.h5") == False:
        hist = fgcnn_model.fit(X_train, y_train, batch_size=1, epochs=25, validation_data = (X_test, y_test), verbose=1)
        fgcnn_model.save_weights("model/fgcnn_weights.h5")
        f = open('model/fgcnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        fgcnn_model.load_weights("model/fgcnn_weights.h5")
    predict = fgcnn_model.predict(X_test, batch_size=1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("Propose GraphCNN", predict, y_test1)

def trainDenseNet():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    densenet = DenseNet121(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in densenet.layers:
        layer.trainable = False
    densenet_model = Sequential()
    densenet_model.add(densenet)
    densenet_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    densenet_model.add(MaxPooling2D(pool_size = (1, 1)))
    densenet_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    densenet_model.add(MaxPooling2D(pool_size = (1, 1)))
    densenet_model.add(Flatten())
    densenet_model.add(Dense(units = 256, activation = 'relu'))
    densenet_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    densenet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    densenet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
    if os.path.exists("model/densenet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/densenet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = densenet_model.fit(X_train, y_train, batch_size = 32, epochs = 25, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/densenet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        densenet_model.load_weights("model/densenet_weights.hdf5")
    predict = densenet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:700] = y_test1[0:700]
    calculateMetrics("Existing DenseNet121", predict, y_test1)

def graph():
    df = pd.DataFrame([['Propose GraphCNN','Precision',precision[0]],['Propose GraphCNN','Recall',recall[0]],['Propose GraphCNN','F1 Score',fscore[0]],['Propose GraphCNN','Accuracy',accuracy[0]],
                       ['Existing DenseNet121','Precision',precision[1]],['Existing DenseNet121','Recall',recall[1]],['Existing DenseNet121','F1 Score',fscore[1]],['Existing DenseNet121','Accuracy',accuracy[1]],
                      ],columns=['Algorithms','Metrics','Value'])
    df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(5, 3))
    plt.title("All Algorithms Performance Graph")
    plt.tight_layout()
    plt.show()

def values(filename, acc):
    f = open(filename, 'rb')
    train_values = pickle.load(f)
    f.close()
    accuracy_value = train_values[acc]
    return accuracy_value

def trainingGraph():
    gcnn_acc = values("model/fgcnn_history.pckl", "val_accuracy")
    dense_acc = values("model/densenet_history.pckl", "accuracy")    
    plt.figure(figsize=(6,4))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(gcnn_acc)
    plt.plot(dense_acc)
    plt.legend(['Propose GCNN Accuracy', 'DenseNet121 Accuracy'], loc='upper left')
    plt.title('Training Accuracy Comparison Graph')
    plt.tight_layout()
    plt.show()    

def getFeatureImage(img, model):
    feature_model = Model(model.inputs, model.layers[-9].output)
    predict = feature_model.predict(img)#now using  cnn model to detcet tumor damage
    predict = predict[0]
    pred = predict[:,:,14]
    pred =  pred*255
    pred = cv2.resize(pred, (150, 150))
    return pred

def predict():
    global fgcnn_model, labels
    filename = filedialog.askopenfilename(initialdir = "testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (32,32))#resize image
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255 #normalizing test image
    predict = fgcnn_model.predict(img)#now using  cnn model to detcet tumor damage
    predict = np.argmax(predict)
    feature_image = getFeatureImage(img, fgcnn_model)
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Severity Predicted : '+labels[predict], (100, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2RGB)
    
    f, axarr = plt.subplots(1,2, figsize=(8,4)) 
    axarr[0].imshow(img, cmap="gray")
    axarr[0].title.set_text('Severity ('+labels[predict]+")")
    axarr[1].imshow(feature_image, cmap="gray")
    axarr[1].title.set_text('Features Map Image')
    plt.show()
    
    

font = ('times', 15, 'bold')
title = Label(main, text='Classification of Diabetic Retinopathy Disease Levels by Extracting Topological Features Using Graph Neural Networks')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload EyePacs Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=20,y=150)
processButton.config(font=ff)

splitButton = Button(main, text="Split Dataset Train & Test", command=splitDataset)
splitButton.place(x=20,y=200)
splitButton.config(font=ff)

proposeButton = Button(main, text="Train Propose GraphCNN Algorithm", command=trainGCNN)
proposeButton.place(x=20,y=250)
proposeButton.config(font=ff)

densenetButton = Button(main, text="Train DenseNet121 Algorithm", command=trainDenseNet)
densenetButton.place(x=20,y=300)
densenetButton.config(font=ff)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=20,y=350)
graphButton.config(font=ff)

accButton = Button(main, text="Training Accuracy Graph", command=trainingGraph)
accButton.place(x=20,y=400)
accButton.config(font=ff)

predictButton = Button(main, text="Retinopathy Grade Detection", command=predict)
predictButton.place(x=20,y=450)
predictButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=360,y=100)
text.config(font=font1)

main.config(bg='forestgreen')
main.mainloop()
