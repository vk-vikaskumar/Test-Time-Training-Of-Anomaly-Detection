import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import roc_auc_score, accuracy_score

from keras.layers import Dense, Input, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D

from losses_super_dean import load_loss
from extract_ts_features import apply_time_series_transformations





#based on old main.py
seed_value = 42  # You can choose any integer as the seed
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
#scaler = StandardScaler()

#choose which loss function to use. 'base' was used for most of the paper, while for example 'theory' is our approach to a fraction independent loss function
loss2="wasserstein"

#dataset to be used. Should be in the same format as cardio.npz ('x' training data, 'tx' test data, 'ty' test labels [0:normal, 1:anomaly]. Also mean(ty)=0.5)
dataset="cardio.npz"

#wether to downsample the test set. Represents the fraction of anomalies in the test set. So 0.5 means no downsampling and is the standart case.
frac=0.5

#how many epochs to use for pretraining
epoch1=10

#how many epochs to use for the main training phase
epoch2=20

#some losses understand a weighting factor seperating training and test losses differently. This is this factor (set to 1 to ignore weighting)
weight=0.5

#wether to do feature bagging (0.5 means use random half of features, 1.0 means no feature bagging)
fbfrac=0.5

#batch size used. Increase to 500 for theory loss
batch=128

#save results (filename) or not (None)
filename=None




def normalize(x,tx):
    mn,mx=np.min(x,axis=0),np.max(x,axis=0)
    x=(x-mn)/(mx-mn+0.00001)
    tx=(tx-mn)/(mx-mn+0.00001)
    return x,tx





class FeatureBagging():
    def __init__(self,dim,count=10):
        self.dim=dim
        if type(count)==float:
            count=int(dim*count)
        self.count=count
        self.bag=np.random.choice(dim,count,replace=False)

    def __call__(self,x):
        return x[:,self.bag]
    def get_selected_features(self):
        return self.bag





def refractor(tx,ty,frac=frac):
    #cuts either normal or abnormal ones, so that the fraction of abnormal ones is frac
    np.random.seed(42)#make sure different runs are comparable
    N,A=tx[ty==0],tx[ty==1]
    if frac==0:
        return N, np.zeros(len(N))
    np.random.shuffle(N)
    np.random.shuffle(A)
    #0->only N, 1->only A
    nN=len(N)
    nA=len(A)
    goal=(1/frac-1)
    if frac<=0.5:
        nA=int(nN/goal)
    else:
        nN=int(nA*goal)

    N=N[:nN]
    A=A[:nA]
    tx=np.concatenate([N,A])
    ty=np.concatenate([np.zeros(nN),np.ones(nA)])

    idx=np.arange(len(tx))
    np.random.shuffle(idx)
    tx=tx[idx]
    ty=ty[idx]

    return tx,ty



loss0="mse"
def sigmoidp(x):#modified sigmoid function
    return 1/(1+tf.math.exp(1-x))
    #return tf.keras.activations.sigmoid(1-x)

def one_model(x0,tx0,ty,loss2, weig=weight):#train one model

    if fbfrac<1.0:#if feature bagging is used, apply it
        fb=FeatureBagging(dim=x0.shape[1],count=fbfrac)
        x=fb(x0)
        tx=fb(tx0)
    else:
        x=x0
        tx=tx0


    inp=keras.layers.Input(shape=(x.shape[1],))
    q=inp
    q=keras.layers.Dense(100,activation='relu',use_bias=False)(q)#no bias is used in the paper, similarly to DeepSVDD and DEAN
    q=keras.layers.Dense(100,activation='relu',use_bias=False)(q)
    q=keras.layers.Dense(100,activation='relu',use_bias=False)(q)
    q=keras.layers.Dense(10,activation='relu',use_bias=False)(q)

    q=keras.layers.Dense(1,activation=sigmoidp,use_bias=False)(q)


    model=keras.Model(inputs=inp,outputs=q)
    opt=keras.optimizers.Adam(learning_rate=0.001)
    #print("statred pre-training")
    model.compile(optimizer=opt,loss=load_loss(loss0))
    model.fit(x,np.ones(len(x))/2,epochs=epoch1,batch_size=batch,verbose=0)
    #print("finished pre-training")
    mn=1.0/2#expected value of the mean of the learned distribution

    alpha=10.0



    train=np.concatenate([x,tx],axis=0)#create a new dataset, that contains both training and testing data
    train_y=np.concatenate([np.zeros(len(x)),np.ones(len(tx))],axis=0)

    #shuffle the new dataset
    idx=np.arange(len(train))
    np.random.shuffle(idx)
    train=train[idx]
    train_y=train_y[idx]

    
    model.compile(optimizer=opt,loss=load_loss(loss2))

    model.fit(train,train_y,epochs=epoch2,batch_size=batch,callbacks=[keras.callbacks.TerminateOnNaN()],verbose=0)
    #print("fininshed training super dean ------")
    pred=model.predict(tx)
    px=model.predict(x)


    return px,pred

#px,pred=one_model(x,tx0,ty)
def ensemble_models_with_performance(x, tx0, ty,loss, n_models=10, dim=None):
    feature_performance = {}  # To store the performance of models using specific features
    ensemble_px = np.zeros(x.shape[0])
    ensemble_pred = np.zeros(tx0.shape[0])
    successful_models = 0

    for i in range(n_models):
        try:
            print(f"Training model {i + 1}/{n_models}")
            fb = FeatureBagging(dim=x.shape[1], count=fbfrac)
            selected_features = fb.get_selected_features()
            
            # Train the model on the selected features
            px, pred = one_model(fb(x), fb(tx0), ty, loss)
            if np.isnan(pred[0]):
                continue

            # Evaluate the submodel's performance (e.g., AUC)
            submodel_auc = roc_auc_score(ty, pred)
            print("ROC: ",submodel_auc)
            print(selected_features)
            print("feature :",len(selected_features))
            # Update feature performance scores
            for feature in selected_features:
                if feature not in feature_performance:
                    feature_performance[feature] = []
                feature_performance[feature].append(submodel_auc)

            # Accumulate predictions for ensemble
            ensemble_px += px.flatten()
            ensemble_pred += pred.flatten()
            successful_models += 1
        except Exception as e:
            print(f"Model {i + 1} failed due to: {e}")
            continue

    if successful_models > 0:
        ensemble_px /= successful_models
        ensemble_pred /= successful_models
    else:
        raise ValueError("All models failed. No valid ensemble can be created.")

    return ensemble_px, ensemble_pred, feature_performance

def super_dean(df_train, df_test, loss):
    ty= df_test.loc[:, "is_anomaly"].values
    df_train, df_test = df_train.loc[:, df_train.columns != "is_anomaly"], df_test.loc[:, df_test.columns != "is_anomaly"]
    
    df_train, df_test=normalize(df_train, df_test)
    df_train_transformed = apply_time_series_transformations(df_train, test="yes", window_size=10, wavelet='db1')
    #print(f"Transformed Data Points:\n{df_train_transformed.head()}")
    df_test_transformed = apply_time_series_transformations(df_test,test="yes", window_size=10, wavelet='db1')
    #print(f"Transformed Data Points:\n{df_test_transformed.head()}")

    x,tx=df_train_transformed.loc[:, df_train_transformed.columns != "is_anomaly"].values, df_test_transformed.loc[:, df_test_transformed.columns != "is_anomaly"].values
    
    x0,tx0,ty=x,tx,ty 
    # Train ensemble
    try:
        ensemble_px, ensemble_pred, feature_performance = ensemble_models_with_performance(x, tx0, ty,loss, n_models=10, dim=x.shape[1])
        auc = roc_auc_score(ty, ensemble_pred)

        # Aggregate performance metrics (e.g., mean AUC for each feature)
        feature_importance_by_performance = {
            df_train_transformed.columns[feature]: np.mean(perfs) for feature, perfs in feature_performance.items()
        }

    # Calculate AUC
        auc = roc_auc_score(ty, ensemble_pred)
        print("Data ROC_AUC score: ", auc)
        if filename:
            tosave = {"px": ensemble_px, "ptx": ensemble_pred, "tx": tx0, "ty": ty, "x": x, "auc": auc}
            np.savez_compressed(filename, **tosave)

        return auc, ensemble_px, ensemble_pred, feature_importance_by_performance
    except Exception as e:
        print(f"Ensemble training failed: {e}")
        
    

