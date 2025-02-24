import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from sklearn.metrics import roc_auc_score

def load_loss(nam,*args,**kwargs):
    nam=nam.lower()
    if nam=="diff_q1_q3":
        return gen_base_loss_diff_q1_q3(*args,**kwargs)
    if nam=="sorted_diff":
        return gen_base_loss_diff_sorted(*args,**kwargs)
    if nam=="wasserstein":
        return gen_base_loss_wasserstein(*args,**kwargs)
    if nam=="bhattacharya":
        return gen_base_loss_bhattacharya(*args,**kwargs)
    if nam=="mahalanobis":
        return gen_base_loss_mahalanobis(*args,**kwargs)
    if nam=="cosine":
        return gen_base_loss_cosine_distance(*args,**kwargs)
    if nam=="sorted_diff_wasserstein":
        return gen_base_loss_sorted_diff_wasserstein(*args,**kwargs)
    if nam=="sorted_diff_mahalanobis":
        return gen_base_loss_sorted_diff_mahalanobis(*args,**kwargs)
    if nam=="sorted_diff_cosine":
        return gen_base_loss_sorted_diff_cosine(*args,**kwargs)
    if nam=="cosine_mahalanobis":
        return gen_base_loss_cosine_mahalanobis(*args,**kwargs)
    if nam=="mahalanobis_wasserstein":
        print("detected_ loss mahalanobis wasserstein:--------------")
        return gen_base_loss_mahalanobis_wasserstein(*args,**kwargs)
    if nam=="cosine_wasserstein":
        return gen_base_loss_cosine_wasserstein(*args,**kwargs)
    if nam=="min_max_mean":
        return min_max_mean_loss(*args, **kwargs)
    if nam=="cos":
        return cosine_dist(*args, **kwargs)
    if nam=="wasser":
        return wasser(*args, **kwargs)
    if nam=="mse":
        return gen_mse(*args,**kwargs)
    if nam=="base":
        return gen_base_loss(*args,**kwargs)
    if nam=="l1":
        return gen_linear_loss(*args,**kwargs)
    if nam=="sigmoidal":
        return gen_sigmoid_loss(*args,**kwargs)
    if nam=="simple":
        return gen_simple_loss(*args,**kwargs)
    if nam=="divider":
        return gen_divider_loss(*args,**kwargs)
    if nam=="mse":
        return gen_mse(*args,**kwargs)
    if nam=="multi":
        return gen_multi_loss(*args,**kwargs)
    if nam=="both":
        return gen_both_loss(*args,**kwargs)
    if nam=="half":
        return gen_half_loss(*args,**kwargs)
    if nam=="max":
        return gen_maximum_loss(*args,**kwargs)
    if nam=="min":
        return gen_minimum_loss(*args,**kwargs)
    if nam=="twister":
        return gen_twister_loss(*args,**kwargs)
    if nam=="theory":
        return gen_theory_loss(*args,**kwargs)
    if nam=="base_new":
        return gen_base_loss_new(*args,**kwargs)
    if nam=="lag_loss":
        return gen_lagg_loss(*args,**kwargs)
    if nam=="l1_std": 
        return gen_linear_loss_std(*args,**kwargs)
    if nam=="l2_std": 
        return gen_l2_loss_std(*args,**kwargs)
    if nam == "l1_without_median":
        return gen_linear_loss_without_median(*args,**kwargs)
    if nam == "l2_without_median":
        return gen_l2_loss_without_median(*args,**kwargs)
    if nam == "l1_min_max":
        return gen_linear_loss_min_max(*args,**kwargs)
    if nam == "l2_min_max":
        return gen_l2_loss_min_max(*args,**kwargs)
    if nam == "l1_min_max_part1":
        return gen_l1_loss_min_max_part1(*args,**kwargs)
    if nam == "l2_min_max_part1":
        return gen_l2_loss_min_max_part1(*args,**kwargs)
    if nam == "l1_std_inverted":
        return gen_l1_loss_std_inverted(*args,**kwargs)
    if nam == "l2_std_inverted":
        return gen_l2_loss_std_inverted(*args,**kwargs)
    if nam == "l1_90_10":
        return gen_linear_loss_90_10(*args,**kwargs)
    if nam == "l2_90_10":
        return gen_l2_loss_90_10(*args,**kwargs)
    if nam == "l1_10_25_50_75_90":
        return gen_linear_loss_without_10_25_50_75_90(*args,**kwargs)
    if nam == "l2_10_25_50_75_90":
        return gen_l2_loss_without_10_25_50_75_90(*args,**kwargs)
    if nam=="l1_sorted_diff":
        return gen_l1_sorted_diff(*args,**kwargs)
    if nam=="l2_sorted_diff":
        return gen_l2_sorted_diff(*args,**kwargs)
    if nam=="l1_20_50_80":
        return gen_linear_loss_without_20_50_80(*args,**kwargs)
    if nam=="l1_25_50_75":
        return gen_linear_loss_without_25_50_75(*args,**kwargs)
    if nam=="l2_sorted_diff":
        return gen_l2_sorted_diff(*args,**kwargs)
    if nam == "l1_extreme_low_high":
        return gen_linear_loss_without_extreme_low_high(*args,**kwargs)
    if nam=="l1_IQR_90_10_75_25":
        return gen_linear_loss_without_IQR_90_10_75_25(*args,**kwargs)
    if nam=="l1_25_75":
        return gen_linear_loss_without_25_75(*args,**kwargs)
    if nam == "l2_min_max_inverted":
        return gen_l2_loss_min_max_inverted(*args,**kwargs)
    if nam == "l1_min_max_inverted":
        return gen_l2_loss_min_max_inverted(*args,**kwargs)
    if nam=="gen_base_new_wasserstein_smooth_panelty":
        return gen_anomaly_loss(*args,**kwargs)    
    return nam

def gen_mse(*args,**kwargs):
    return keras.losses.MeanSquaredError()
    


def gen_base_loss(weig=4.0):
    def base_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        loss1=tf.reduce_mean(tf.square(part1-tf.zeros_like(part1)))
        loss2=tf.reduce_mean(tf.square(part2-tf.ones_like(part2)))
        
        return loss1+loss2*weig
    
    return base_loss


def gen_base_loss_new(weig1=1.0, weig2=4.0):
    def base_loss(y_true, y_pred):
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))
        
        
        return loss1 + weig1 * loss2 
    
    return base_loss

def gen_linear_loss(weig=1.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        part1_mean = (part1 - tf.reduce_mean(part1))
        part2_mean = (part2 - tf.reduce_mean(part2))
        loss1=tf.reduce_mean(tf.abs(part1_mean-tf.zeros_like(part1_mean)))
        loss2=tf.reduce_mean(tf.abs(part2_mean-tf.ones_like(part2_mean)))
    
        return loss1+loss2*weig
    return linear_loss

def gen_linear_loss_std(weig=1.0):
    def linear_loss(y_true, y_pred):
        # Create masks for classes 0 and 1
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        # Check if part1 or part2 are empty
        if tf.size(part1) == 0 or tf.size(part2) == 0:
            return tf.constant(0.0)  # Return a 0 loss if no valid samples for class 0 or class 1

        # Avoid division by zero by adding a small epsilon
        epsilon = tf.keras.backend.epsilon()

        part1_std = tf.math.reduce_std(part1) + epsilon
        part2_std = tf.math.reduce_std(part2) + epsilon

        # Normalize the parts
        part1_mean = (part1 - tf.reduce_mean(part1)) / part1_std
        part2_mean = (part2 - tf.reduce_mean(part2)) / part2_std

        # Compute losses for part1 and part2
        loss1 = tf.reduce_mean(tf.abs(part1_mean - tf.zeros_like(part1_mean)))
        loss2 = tf.reduce_mean(tf.abs(part2_mean - tf.ones_like(part2_mean)))

        return loss1 + loss2 * weig

    return linear_loss

def gen_l2_loss_std(weig=1.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        part1_mean = (part1 - tf.reduce_mean(part1))/tf.math.reduce_std(part1)
        part2_mean = (part2 - tf.reduce_mean(part2))/tf.math.reduce_std(part2)
        loss1=tf.reduce_mean(tf.square(part1_mean-tf.zeros_like(part1_mean)))
        loss2=tf.reduce_mean(tf.square(part2_mean-tf.ones_like(part2_mean)))
    
        return loss1+loss2*weig
    return linear_loss


def gen_l2_loss_std_inverted(weig=1.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        part1_mean = (part1 - tf.reduce_mean(part1))/tf.math.reduce_std(part1)
        part2_mean = (part2 - tf.reduce_mean(part2))/tf.math.reduce_std(part2)
        loss1=tf.reduce_mean(tf.square(part1_mean-tf.ones_like(part1_mean)))
        loss2=tf.reduce_mean(tf.square(part2_mean-tf.zeros_like(part2_mean)))
    
        return loss1+loss2*weig
    return linear_loss

def gen_l1_loss_std_inverted(weig=1.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        part1_mean = (part1 - tf.reduce_mean(part1))/tf.math.reduce_std(part1)
        part2_mean = (part2 - tf.reduce_mean(part2))/tf.math.reduce_std(part2)
        loss1=tf.reduce_mean(tf.abs(part1_mean-tf.ones_like(part1_mean)))
        loss2=tf.reduce_mean(tf.abs(part2_mean-tf.zeros_like(part2_mean)))
    
        return loss1+loss2*weig
    return linear_loss


def gen_linear_loss_min_max(weig=1.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]
        part1_min = tf.reduce_min(part1)
        part1_max = tf.reduce_max(part1)

        part2_min = tf.reduce_min(part2)
        part2_max = tf.reduce_max(part2)
        
        
        part1_min_max = (part1 - part1_min)/(part1_max - part1_min)
        part2_min_max = (part2 - part1_min)/(part2_max - part2_min)
        loss1=tf.reduce_mean(tf.abs(part1_min_max-tf.zeros_like(part1_min_max)))
        loss2=tf.reduce_mean(tf.abs(part2_min_max-tf.ones_like(part2_min_max)))
    
        return loss1+loss2*weig
    return linear_loss

def gen_l2_loss_min_max_part1(weig=4.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]
        part1_min = tf.reduce_min(part1)
        part1_max = tf.reduce_max(part1)

        #part2_min = tf.reduce_min(part2)
        #part2_max = tf.reduce_max(part2)
        
        
        part1_min_max =(part1 - part1_min)/(part1_max - part1_min)
        part2_min_max =(part2 - part1_min)/(part1_max - part1_min)
        loss1=tf.reduce_mean(tf.square(part1_min_max-tf.zeros_like(part1_min_max)))
        loss2=tf.reduce_mean(tf.square(part2_min_max-tf.ones_like(part2_min_max)))
    
        return loss1+loss2*weig
    return linear_loss

def gen_l1_loss_min_max_part1(weig=4.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]
        part1_min = tf.reduce_min(part1)
        part1_max = tf.reduce_max(part1)

        #part2_min = tf.reduce_min(part2)
        #part2_max = tf.reduce_max(part2)
        
        
        part1_min_max =(part1 - part1_min)/(part1_max - part1_min)
        part2_min_max =(part2 - part1_min)/(part1_max - part1_min)
        loss1=tf.reduce_mean(tf.abs(part1_min_max-tf.zeros_like(part1_min_max)))
        loss2=tf.reduce_mean(tf.abs(part2_min_max-tf.ones_like(part2_min_max)))
    
        return loss1+loss2*weig
    return linear_loss


def gen_l2_loss_min_max_inverted(weig=1.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]
        part1_min = tf.reduce_min(part1)
        part1_max = tf.reduce_max(part1)

        part2_min = tf.reduce_min(part2)
        part2_max = tf.reduce_max(part2)
        
        
        part1_min_max =(part1 - part1_min)/(part1_max - part1_min)
        part2_min_max =(part2 - part2_min)/(part2_max - part2_min)
        loss1=tf.reduce_mean(tf.square(part1_min_max-tf.ones_like(part1_min_max)))
        loss2=tf.reduce_mean(tf.square(part2_min_max-tf.zeros_like(part2_min_max)))
    
        return loss1+loss2*weig
    return linear_loss

def gen_l1_loss_min_max_inverted(weig=1.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]
        part1_min = tf.reduce_min(part1)
        part1_max = tf.reduce_max(part1)

        part2_min = tf.reduce_min(part2)
        part2_max = tf.reduce_max(part2)
        
        
        part1_min_max =(part1 - part1_min)/(part1_max - part1_min)
        part2_min_max =(part2 - part2_min)/(part2_max - part2_min)
        loss1=tf.reduce_mean(tf.abs(part1_min_max-tf.ones_like(part1_min_max)))
        loss2=tf.reduce_mean(tf.abs(part2_min_max-tf.zeros_like(part2_min_max)))
    
        return loss1+loss2*weig
    return linear_loss

def gen_l2_loss_min_max_minus_1(weig=1.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]
        part1_min = tf.reduce_min(part1)
        part1_max = tf.reduce_max(part1)

        part2_min = tf.reduce_min(part2)
        part2_max = tf.reduce_max(part2)
        
        
        part1_min_max =(part1 - part1_min)/(part1_max - part1_min)
        part2_min_max =(part2 - part2_min)/(part2_max - part2_min)
        loss1=tf.reduce_mean(tf.square(part1_min_max-tf.zeros_like(part1_min_max)))
        loss2=tf.reduce_mean(tf.square(part2_min_max-tf.ones_like(part2_min_max)))

        return (loss1+loss2*weig)
    return linear_loss

def gen_l2_loss_min_max(weig=1.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]
        part1_min = tf.reduce_min(part1)
        part1_max = tf.reduce_max(part1)

        part2_min = tf.reduce_min(part2)
        part2_max = tf.reduce_max(part2)
        
        
        part1_min_max =(part1 - part1_min)/(part1_max - part1_min)
        part2_min_max =(part2 - part2_min)/(part2_max - part2_min)
        loss1=tf.reduce_mean(tf.square(part1_min_max-tf.zeros_like(part1_min_max)))
        loss2=tf.reduce_mean(tf.square(part2_min_max-tf.ones_like(part2_min_max)))
    
        return loss1+loss2*weig
    return linear_loss

def gen_l2_loss_90_10(weig=1.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]
        part1_min = tfp.stats.percentile(part1, 10.0)
        part1_max = tfp.stats.percentile(part1, 90.0)

        part2_min = tfp.stats.percentile(part2, 10.0)
        part2_max = tfp.stats.percentile(part2, 90.0)
        
        
        part1_min_max = (part1 - part1_min)/(part1_max - part1_min)
        part2_min_max = (part2 - part2_min)/(part2_max - part2_min)
        loss1=tf.reduce_mean(tf.square(part1_min_max-tf.zeros_like(part1_min_max)))
        loss2=tf.reduce_mean(tf.square(part2_min_max-tf.ones_like(part2_min_max)))
    
        return loss1+loss2*weig
    return linear_loss

def gen_linear_loss_90_10(weig=1.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]
        part1_min = tfp.stats.percentile(part1, 10.0)
        part1_max = tfp.stats.percentile(part1, 90.0)

        part2_min = tfp.stats.percentile(part2, 10.0)
        part2_max = tfp.stats.percentile(part2, 90.0)
        
        
        part1_min_max =(part1 - part1_min)/(part1_max - part1_min)
        part2_min_max =(part2 - part2_min)/(part2_max - part2_min)
        loss1=tf.reduce_mean(tf.abs(part1_min_max-tf.zeros_like(part1_min_max)))
        loss2=tf.reduce_mean(tf.square(part2_min_max-tf.ones_like(part2_min_max)))
    
        return loss1+loss2*weig
    return linear_loss

    

def gen_linear_loss_without_median(weig=1.0):
    def linear_loss_without_median(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        part1_median = part1 - tfp.stats.percentile(part1, 50.0)
        part2_median = part2 - tfp.stats.percentile(part1, 50.0)
        loss1=tf.reduce_mean(tf.abs(part1_median-tf.zeros_like(part1_median)))
        loss2=tf.reduce_mean(tf.abs(part2_median-tf.ones_like(part2_median)))
    
        return loss1+loss2*weig

    return linear_loss_without_median

def gen_l2_loss_without_median(weig=1.0):
    def l2_loss_without_median(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        part1_median = part1 - tfp.stats.percentile(part1, 50.0)
        part2_median = part2 - tfp.stats.percentile(part1, 50.0)
        loss1=tf.reduce_mean(tf.square(part1_median-tf.zeros_like(part1_median)))
        loss2=tf.reduce_mean(tf.square(part2_median-tf.ones_like(part2_median)))
    
        return loss1+loss2*weig

    return l2_loss_without_median

def gen_linear_loss_without_10_25_50_75_90(weig=1.0):
    def linear_loss_without_median(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        part1_median = part1 - tf.reduce_mean([tfp.stats.percentile(part1, 10.0),tfp.stats.percentile(part1, 25.0),tfp.stats.percentile(part1, 50.0),tfp.stats.percentile(part1, 75.0), tfp.stats.percentile(part1, 90.0)] )
        part2_median = part2 - tf.reduce_mean([tfp.stats.percentile(part2, 10.0),tfp.stats.percentile(part2, 25.0),tfp.stats.percentile(part2, 50.0)] )
        loss1=tf.reduce_mean(tf.abs(part1_median-tf.zeros_like(part1_median)))
        loss2=tf.reduce_mean(tf.abs(part2_median-tf.ones_like(part2_median)))
    
        return loss1+loss2*weig

    return linear_loss_without_median

def gen_linear_loss_without_IQR_90_10_75_25(weight=1.0):
    def linear_loss_without_median(y_true, y_pred):
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        # Adjust based on mean of IQRs
        part1_adjustment = (
            (tfp.stats.percentile(part1, 90.0) - tfp.stats.percentile(part1, 10.0)) +
            (tfp.stats.percentile(part1, 75.0) - tfp.stats.percentile(part1, 25.0)) +
            (tfp.stats.percentile(part1, 60.0) - tfp.stats.percentile(part1, 40.0))
        ) / 3.0
        part2_adjustment = (
            (tfp.stats.percentile(part2, 90.0) - tfp.stats.percentile(part2, 10.0)) +
            (tfp.stats.percentile(part2, 75.0) - tfp.stats.percentile(part2, 25.0)) +
            (tfp.stats.percentile(part2, 60.0) - tfp.stats.percentile(part2, 40.0))
        ) / 3.0
        tf.print("loss detected")

        part1_adjusted = part1 - part1_adjustment
        part2_adjusted = part2 - part2_adjustment

        # Compute the loss
        loss1 = tf.reduce_mean(tf.abs(part1_adjusted - tf.zeros_like(part1_adjusted)))
        loss2 = tf.reduce_mean(tf.abs(part2_adjusted - tf.ones_like(part2_adjusted)))

        return loss1 + loss2 * weight

    return linear_loss_without_median


def gradient_penalty(y_pred):
    """Encourage smoothness by penalizing large gradient changes (except for anomalies)."""
    diffs = y_pred[1:] - y_pred[:-1]
    return tf.reduce_mean(tf.abs(diffs))  # L1 smoothness penalty

def wasserstein_distance(part1, part2):
    part1_shape = part1.shape[0]  # Get the batch size
    part2_shape = part2.shape[0]
    
    # Ensure both tensors have the same length
    min_size = tf.minimum(part1_shape, part2_shape)
    part1, part2 = part1[:min_size], part2[:min_size]  # Slice to equal length
    
    return tf.reduce_mean(tf.abs(tf.sort(part1) - tf.sort(part2)))

def gen_anomaly_loss(weig1=1.0, weig2=1.0, weig3=0.1):
    """Loss function that pushes normal data to 0, anomalies to 1, and ensures smooth trends."""
    def anomaly_loss(y_true, y_pred):
        part1 = y_pred[y_true == 0]  # Normal data (should be close to 0)
        part2 = y_pred[y_true == 1]  # Anomalies (should be close to 1)

        # Push normal data to 0 and anomalies to 1
        loss1 = tf.reduce_mean(tf.abs(part1 - tf.zeros_like(part1)))  # MSE for normal
        loss2 = tf.reduce_mean(tf.abs(part2 - tf.ones_like(part2)))  # MSE for anomalies
        
        # Encourage clear separation between normal and anomalies
        wasserstein_dist = wasserstein_distance_approx(part1, part2)
        
        # Encourage smoothness in predictions to avoid noisy outputs
        smoothness_penalty = gradient_penalty(y_pred)

        return loss1 * weig1 + loss2 * weig1 - wasserstein_dist * weig2 + smoothness_penalty * weig3

    return anomaly_loss

def gen_linear_loss_without_25_50_75(weig=1.0):
    def linear_loss_without_median(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        part1_median = part1 - tf.reduce_mean([tfp.stats.percentile(part1, 25.0),tfp.stats.percentile(part1, 50.0),tfp.stats.percentile(part1, 75.0)] )
        part2_median = part2 - tf.reduce_mean([tfp.stats.percentile(part2, 25.0),tfp.stats.percentile(part2, 50.0),tfp.stats.percentile(part2, 75.0)] )
        loss1=tf.reduce_mean(tf.abs(part1_median-tf.zeros_like(part1_median)))
        loss2=tf.reduce_mean(tf.abs(part2_median-tf.ones_like(part2_median)))
    
        return loss1+loss2*weig

    return linear_loss_without_median

def gen_linear_loss_without_extreme_low_high(weig=1.0):
    def linear_loss_without_median(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        part1_median = part1 - tf.reduce_mean([tfp.stats.percentile(part1, 0.0),tfp.stats.percentile(part1, 10.0),tfp.stats.percentile(part1, 25.0)] )
        part2_median = part2 + tf.reduce_mean([tfp.stats.percentile(part2, 75.0),tfp.stats.percentile(part2, 90.0),tfp.stats.percentile(part2, 100.0)] )
        loss1=tf.reduce_mean(tf.abs(part1_median-tf.zeros_like(part1_median)))
        loss2=tf.reduce_mean(tf.abs(part2_median-tf.ones_like(part2_median)))
    
        return loss1+loss2*weig

    return linear_loss_without_median

def gen_linear_loss_without_25_75(weig=1.0):
    def linear_loss_without_median(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        part1_median = part1 - tfp.stats.percentile(part1, 0.0)
        part2_median = part2 + tfp.stats.percentile(part2, 100.0)
        loss1=tf.reduce_mean(tf.abs(part1_median-tf.zeros_like(part1_median)))
        loss2=tf.reduce_mean(tf.abs(part2_median-tf.ones_like(part2_median)))
    
        return loss1+loss2*weig

    return linear_loss_without_median



def gen_linear_loss_without_20_50_80(weig=1.0):
    def linear_loss_without_median(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        part1_median = part1 - tf.reduce_mean([tfp.stats.percentile(part1, 20.0),tfp.stats.percentile(part1, 50.0),tfp.stats.percentile(part1, 80.0)] )
        part2_median = part2 - tf.reduce_mean([tfp.stats.percentile(part2, 20.0),tfp.stats.percentile(part2, 50.0),tfp.stats.percentile(part2, 80.0)] )
        loss1=tf.reduce_mean(tf.abs(part1_median-tf.zeros_like(part1_median)))
        loss2=tf.reduce_mean(tf.abs(part2_median-tf.ones_like(part2_median)))
    
        return loss1+loss2*weig

    return linear_loss_without_median
def gen_l2_loss_without_10_25_50_75_90(weig=1.0):
    def linear_loss_without_median(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        part1_median = part1 - tf.reduce_mean([tfp.stats.percentile(part1, 10.0),tfp.stats.percentile(part1, 25.0),tfp.stats.percentile(part1, 50.0),tfp.stats.percentile(part1, 75.0), tfp.stats.percentile(part1, 90.0)] )
        part2_median = part2 - tf.reduce_mean([tfp.stats.percentile(part2, 10.0),tfp.stats.percentile(part2, 25.0),tfp.stats.percentile(part2, 50.0),tfp.stats.percentile(part2, 75.0), tfp.stats.percentile(part2, 90.0)] )
        loss1=tf.reduce_mean(tf.square(part1_median-tf.zeros_like(part1_median)))
        loss2=tf.reduce_mean(tf.square(part2_median-tf.ones_like(part2_median)))
    
        return loss1+loss2*weig

    return linear_loss_without_median
def gen_linear_loss_without_mean(weig=1.0):
    def linear_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        loss1=tf.reduce_mean(tf.abs(part1-tf.zeros_like(part1)))
        loss2=tf.reduce_mean(tf.abs(part2-tf.ones_like(part2)))
    
        return loss1+loss2*weig

    return linear_loss

def gen_l1_sorted_diff():
    def diff_loss(y_true, y_pred):
        part1 = y_pred[y_true==0]
        part2 = y_pred[y_true==1]

        loss1 = tf.reduce_mean(tf.abs(tf.sort(part1) - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.abs(tf.sort(part2) - tf.ones_like(part2)))
        return loss1+loss2
    return diff_loss 

def gen_l2_sorted_diff():
    def diff_loss(y_true, y_pred):
        part1 = y_pred[y_true==0]
        part2 = y_pred[y_true==1]

        loss1 = tf.reduce_mean(tf.square(tf.sort(part1) - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(tf.sort(part2) - tf.ones_like(part2)))
        return loss1+loss2
    return diff_loss 

def gen_sigmoid_loss(weig=1.0):
    def sigmoid_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]
    
        loss1=tf.reduce_mean(tf.square(part1-tf.zeros_like(part1)))
        loss2=tf.reduce_mean(tf.square(part2-tf.ones_like(part2)))
        
        return tf.nn.sigmoid(loss1)+tf.nn.sigmoid(loss2)*weig

    return sigmoid_loss
    
def gen_simple_loss(weig=1.0):
    def simple_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        return tf.reduce_mean(part2)-tf.reduce_mean(part1)*weig
    return simple_loss
    
def gen_divider_loss(weig=1.0):
    def divider_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        mn1=tf.reduce_mean(part1)
        mn2=tf.reduce_mean(part2)*weig
        return (mn1-mn2)/(mn1+mn2)
    return divider_loss
    

def gen_multi_loss(weig=1.0):
    def multi_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        loss1=tf.reduce_mean(tf.square(part1-tf.zeros_like(part1)))
        loss2=tf.reduce_mean(tf.square(part2-tf.ones_like(part2)))
        
        return loss1*loss2

    return multi_loss
    
def gen_both_loss(weig=1.0):
    def both_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        loss1=tf.reduce_mean(tf.square(part1-tf.zeros_like(part1)))
        loss2=tf.reduce_mean(tf.square(part2-tf.ones_like(part2)))
        loss3=tf.reduce_mean(tf.abs(part1-tf.zeros_like(part1)))
        loss4=tf.reduce_mean(tf.abs(part2-tf.ones_like(part2)))
        
        return loss1+loss2+loss3+loss4

    return both_loss

def gen_half_loss(weig=1.0):
    def half_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        loss1=tf.reduce_mean(tf.square(part1-0.5*tf.ones_like(part1)))
        loss2=tf.reduce_mean(tf.square(part2-tf.ones_like(part2)))
        
        return loss1+loss2

    return half_loss

def gen_maximum_loss(weig=1.0):
    def maximum_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        loss1=tf.reduce_max(tf.square(part1-tf.zeros_like(part1)))
        loss2=tf.reduce_max(tf.square(part2-tf.ones_like(part2)))
        
        return loss1+loss2

    return maximum_loss

def gen_minimum_loss(weig=1.0):
    def minimum_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        loss1=tf.reduce_min(tf.square(part1-tf.zeros_like(part1)))
        loss2=tf.reduce_min(tf.square(part2-tf.ones_like(part2)))
        
        return loss1+loss2

    return minimum_loss

def gen_twister_loss(weig=1.0):
    def twister_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        loss1=tf.reduce_max(tf.square(part1-tf.zeros_like(part1)))
        loss2=tf.reduce_mean(tf.square(part2-tf.ones_like(part2)))
        
        return loss1+loss2

    return twister_loss

def gen_theory_loss(weig=1.0):
    def theory_loss(y_true,y_pred):
        part1=y_pred[y_true==0]
        part2=y_pred[y_true==1]

        loss1=tf.reduce_max(part1)
        loss2=tf.reduce_max(part2)
        
        return loss1-loss2
    return theory_loss



def min_max_mean_loss(weig1=1.0, weig2=2.0, gamma=1.0):
    def base_loss(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))
        loss3 = tf.math.abs(0.5*(tf.math.abs(tf.reduce_min(part1)-tf.reduce_min(part2)+
                                              tf.math.abs(tf.reduce_max(part1)-tf.reduce_max(part2))))- 
                                              tf.math.abs(tf.reduce_mean(part1)-tf.reduce_mean(part2)))
        return loss1 + weig1* loss2 + weig2*(loss3)
    return base_loss

def gen_mse(*args,**kwargs):
    return keras.losses.MeanSquaredError()

def wasserstein_distance_approx(part1, part2):
    # Sort the tensors (approximate CDF calculation)
    part1_sorted = tf.sort(part1)
    part2_sorted = tf.sort(part2)

    # Ensure the tensors have the same size by truncating the longer one
    min_size = tf.minimum(tf.size(part1_sorted), tf.size(part2_sorted))

    part1_trimmed = part1_sorted[:min_size]
    part2_trimmed = part2_sorted[:min_size]

    return tf.reduce_mean(tf.square(part1_trimmed - part2_trimmed))

def mahalanobis_distance(x, y):
    # Ensure inputs are 2D: [n_samples, n_features]
    x = tf.reshape(x, [-1, 1])  # Reshape to (n_samples, 1)
    y = tf.reshape(y, [-1, 1])  # Reshape to (n_samples, 1)

    # Compute the covariance matrix of x and y
    combined = tf.concat([x, y], axis=0)
    covariance_matrix = tfp.stats.covariance(combined)

    # Compute the inverse covariance matrix
    inv_covariance_matrix = tf.linalg.inv(covariance_matrix)

    # Compute the mean difference between x and y
    mean_diff = tf.reduce_mean(x, axis=0) - tf.reduce_mean(y, axis=0)

    # Compute Mahalanobis distance
    distance = tf.sqrt(tf.matmul(tf.matmul(mean_diff[None, :], inv_covariance_matrix), mean_diff[:, None]))
    return tf.squeeze(distance)



def pad_to_match_length(x, y):
    """Pads the shorter vector to match the length of the longer vector."""
    len_x = tf.shape(x)[0]
    len_y = tf.shape(y)[0]

    # Calculate the padding required
    pad_x = tf.maximum(0, len_y - len_x)  # Padding for x if y is longer
    pad_y = tf.maximum(0, len_x - len_y)  # Padding for y if x is longer

    # Pad x and y with zeros to match lengths
    x_padded = tf.pad(x, [[0, pad_x]], constant_values=0)
    y_padded = tf.pad(y, [[0, pad_y]], constant_values=0)

    return x_padded, y_padded

def cosine_distance(x, y):
    # Pad x and y to the same length
    #x_padded, y_padded = pad_to_match_length(x, y)

    epsilon = 1e-8

    # Compute norms of part1 and tf.zeros_like(part1)
    norm_part1 = tf.norm(x, axis=-1, keepdims=True)
    norm_zeros = tf.norm(y, axis=-1, keepdims=True) + epsilon

    # Compute cosine similarity
    dot_product = tf.reduce_sum(x * y, axis=-1, keepdims=True)
    cosine_similarity = dot_product / (norm_part1 * norm_zeros)

    # Take the mean of the cosine similarities
    mean_cosine_similarity = tf.reduce_mean(cosine_similarity)


    # Cosine distance is 1 - cosine similarity
    #cosine_dist = 1.0 - cosine_similarity
    return mean_cosine_similarity

def gen_base_loss_diff_sorted(weig1=1.0, weig2=4.0, gamma=1.0):
    def base_loss(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))

        

        # Compute mean difference between sorted part1 and part2
        sorted_part1 = tf.sort(part1)
        sorted_part2 = tf.sort(part2)

        # Match lengths by truncating the longer array (or you can pad the shorter one)
        min_length = tf.minimum(tf.shape(sorted_part1)[0], tf.shape(sorted_part2)[0])
        sorted_part1 = sorted_part1[:min_length]
        sorted_part2 = sorted_part2[:min_length]

        loss3 = tf.reduce_mean(tf.abs(sorted_part1 - sorted_part2))
        return loss1 + weig1* loss2 + weig2*(loss3)
    return base_loss
def gen_lagg_loss():
    def lagg_loss(y_true, y_pred):
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        lagged_part1 = part1[1:] - part1[:-1]  # Subtract each value from the previous one
        #lagged_part2 = part2[1:] - part2[:-1]
        # Calculate the loss for the lagged differences
        loss1 = tf.reduce_mean(tf.abs(lagged_part1 - tf.zeros_like(lagged_part1)))
        loss2 = tf.reduce_mean(tf.abs(part2 - tf.ones_like(part2)))

        # Combine the losses
        total_loss = loss1 + loss2
        return total_loss
    return lagg_loss

def wasser():
    def base_loss(y_true, y_pred):
        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]
        part1_sorted = tf.sort(part1)
        part2_sorted = tf.sort(part2)
        loss1 = wasserstein_distance_approx(part1_sorted, tf.zeros_like(part1)) 
        loss2 = wasserstein_distance_approx(part2_sorted, tf.ones_like(part2))
        return loss1+ loss2
    return base_loss


def gen_base_loss_wasserstein(weig1=1.0, weig2=4.0, gamma=1.0):
    def base_loss(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))
        
        loss3 = wasserstein_distance_approx(part1, part2)
        return loss1 + weig1* loss2 + weig2*(loss3)
    return base_loss

def gen_base_loss_bhattacharya(weig1=1.0, weig2=4.0, gamma=1.0):
    def base_loss_bhattacharya(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))
        
        loss3=tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.sqrt(part1 * part2), axis=0))) #bhattacharya distance
        #loss8 = mahalanobis_distance(part1,part2)
        #loss9 = cosine_distance(part1,part2)
        return loss1 + weig1* loss2 + weig2*(loss3)
    return base_loss_bhattacharya

def gen_base_loss_mahalanobis(weig1=1.0, weig2=4.0, gamma=1.0):
    def base_loss(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))
    
        loss3 = mahalanobis_distance(part1,part2)
        #loss9 = cosine_distance(part1,part2)
        return loss1 + weig1* loss2 + weig2*(loss3)
    return base_loss

def cosine_dist():
    def base_loss(y_true, y_pred):
        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]
        cosine_similarity = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)

        # Compute cosine similarity
        loss1 = tf.reduce_mean(cosine_similarity(part1, tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(cosine_similarity(part2, tf.ones_like(part2)))

        #loss1 = cosine_distance(part1,tf.zeros_like(part1))
        #loss2 = cosine_distance(part1,tf.ones_like(part2))
        loss = loss1  + loss2
        return loss
    return base_loss


def gen_base_loss_cosine_distance(weig1=1.0, weig2=4.0, gamma=1.0):
    def base_loss(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))
    
        loss3 = cosine_distance(part1,part2)
        return loss1 + weig1* loss2 + weig2*(loss3)
    return base_loss



def gen_base_loss_diff_q1_q3(weig1=1.0, weig2=4.0, gamma=1.0):
    def base_loss(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))

        q1_part1 = tfp.stats.percentile(part1, 25.0)  # 25th percentile for part1
        q1_part2 = tfp.stats.percentile(part2, 25.0)  # 25th percentile for part2
        loss3 = tf.square(q1_part1 - q1_part2)
        loss4 = tf.square(tfp.stats.percentile(part1, 75.0) - tfp.stats.percentile(part2, 75.0))

        # Compute mean difference between sorted part1 and part2
        #sorted_part1 = tf.sort(part1)
        #sorted_part2 = tf.sort(part2)

        # Match lengths by truncating the longer array (or you can pad the shorter one)
        #min_length = tf.minimum(tf.shape(sorted_part1)[0], tf.shape(sorted_part2)[0])
        #sorted_part1 = sorted_part1[:min_length]
        #sorted_part2 = sorted_part2[:min_length]

        #mean_diff = tf.reduce_mean(tf.abs(sorted_part1 - sorted_part2))
        #loss5 = wasserstein_distance_approx(part1, part2)
        #loss6 = tf.square(tfp.stats.variance(part1)-tfp.stats.variance(part2))
        #loss7=tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.sqrt(part1 * part2), axis=0))) #bhattacharya distance
        #loss8 = mahalanobis_distance(part1,part2)
        #loss9 = cosine_distance(part1,part2)
        return loss1 + weig1* loss2 + weig2*(loss3 +loss4)
    return base_loss


def gen_base_loss_sorted_diff_wasserstein(weig1=1.0, weig2=4.0, gamma=1.0):
    def base_loss(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))

        # Compute mean difference between sorted part1 and part2
        sorted_part1 = tf.sort(part1)
        sorted_part2 = tf.sort(part2)

        # Match lengths by truncating the longer array (or you can pad the shorter one)
        min_length = tf.minimum(tf.shape(sorted_part1)[0], tf.shape(sorted_part2)[0])
        sorted_part1 = sorted_part1[:min_length]
        sorted_part2 = sorted_part2[:min_length]

        loss3 = tf.reduce_mean(tf.abs(sorted_part1 - sorted_part2))
        loss4 = wasserstein_distance_approx(part1, part2)
        #loss6 = tf.square(tfp.stats.variance(part1)-tfp.stats.variance(part2))
        #loss7=tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.sqrt(part1 * part2), axis=0))) #bhattacharya distance
        #loss8 = mahalanobis_distance(part1,part2)
        #loss9 = cosine_distance(part1,part2)
        return loss1 + weig1* loss2 + weig2*(loss3 +loss4)
    return base_loss

def gen_base_loss_sorted_diff_mahalanobis(weig1=1.0, weig2=4.0, gamma=1.0):
    def base_loss(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))

        # Compute mean difference between sorted part1 and part2
        sorted_part1 = tf.sort(part1)
        sorted_part2 = tf.sort(part2)

        # Match lengths by truncating the longer array (or you can pad the shorter one)
        min_length = tf.minimum(tf.shape(sorted_part1)[0], tf.shape(sorted_part2)[0])
        sorted_part1 = sorted_part1[:min_length]
        sorted_part2 = sorted_part2[:min_length]

        loss3 = tf.reduce_mean(tf.abs(sorted_part1 - sorted_part2))
        #loss4 = wasserstein_distance_approx(part1, part2)
        #loss6 = tf.square(tfp.stats.variance(part1)-tfp.stats.variance(part2))
        #loss7=tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.sqrt(part1 * part2), axis=0))) #bhattacharya distance
        loss4 = mahalanobis_distance(part1,part2)
        #loss9 = cosine_distance(part1,part2)
        return loss1 + weig1* loss2 + weig2*(loss3 +loss4)
    return base_loss

def gen_base_loss_sorted_diff_cosine(weig1=1.0, weig2=4.0, gamma=1.0):
    def base_loss(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))

        # Compute mean difference between sorted part1 and part2
        sorted_part1 = tf.sort(part1)
        sorted_part2 = tf.sort(part2)

        # Match lengths by truncating the longer array (or you can pad the shorter one)
        min_length = tf.minimum(tf.shape(sorted_part1)[0], tf.shape(sorted_part2)[0])
        sorted_part1 = sorted_part1[:min_length]
        sorted_part2 = sorted_part2[:min_length]

        loss3 = tf.reduce_mean(tf.abs(sorted_part1 - sorted_part2))
        #loss4 = wasserstein_distance_approx(part1, part2)
        #loss6 = tf.square(tfp.stats.variance(part1)-tfp.stats.variance(part2))
        #loss7=tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.sqrt(part1 * part2), axis=0))) #bhattacharya distance
        #loss4 = mahalanobis_distance(part1,part2)
        loss4 = cosine_distance(part1,part2)
        return loss1 + weig1* loss2 + weig2*(loss3 +loss4)
    return base_loss
    
def gen_base_loss_cosine_mahalanobis(weig1=1.0, weig2=4.0, gamma=1.0):
    def base_loss(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))

        #loss3 = tf.reduce_mean(tf.abs(sorted_part1 - sorted_part2))
        #loss4 = wasserstein_distance_approx(part1, part2)
        #loss6 = tf.square(tfp.stats.variance(part1)-tfp.stats.variance(part2))
        #loss7=tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.sqrt(part1 * part2), axis=0))) #bhattacharya distance
        loss3 = mahalanobis_distance(part1,part2)
        loss4 = cosine_distance(part1,part2)
        return loss1 + weig1* loss2 + weig2*(loss3 +loss4)
    return base_loss
    
def gen_base_loss_mahalanobis_wasserstein(weig1=1.0, weig2=4.0, gamma=1.0):
    def base_loss(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))

        #loss3 = tf.reduce_mean(tf.abs(sorted_part1 - sorted_part2))
        loss3 = wasserstein_distance_approx(part1, part2)
        #loss6 = tf.square(tfp.stats.variance(part1)-tfp.stats.variance(part2))
        #loss7=tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.sqrt(part1 * part2), axis=0))) #bhattacharya distance
        loss4 = mahalanobis_distance(part1,part2)
        #loss4 = cosine_distance(part1,part2)
        return loss1 + weig1* loss2 + weig2*(loss3 +loss4)
    return base_loss
    
def gen_base_loss_cosine_wasserstein(weig1=1.0, weig2=4.0, gamma=1.0):
    def base_loss(y_true, y_pred):

        # Loss2: Mean squared error for part
        part1 = y_pred[y_true == 0]
        part2 = y_pred[y_true == 1]

        loss1 = tf.reduce_mean(tf.square(part1 - tf.zeros_like(part1)))
        loss2 = tf.reduce_mean(tf.square(part2 - tf.ones_like(part2)))

        #loss3 = tf.reduce_mean(tf.abs(sorted_part1 - sorted_part2))
        loss3 = wasserstein_distance_approx(part1, part2)
        #loss6 = tf.square(tfp.stats.variance(part1)-tfp.stats.variance(part2))
        #loss7=tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.sqrt(part1 * part2), axis=0))) #bhattacharya distance
        #loss4 = mahalanobis_distance(part1,part2)
        loss4 = cosine_distance(part1,part2)
        return loss1 + weig1* loss2 + weig2*(loss3 +loss4)
    return base_loss

