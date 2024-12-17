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
    if nam=="mse":
        return gen_mse(*args,**kwargs)
    return nam


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
    x_padded, y_padded = pad_to_match_length(x, y)

    # Normalize the vectors to unit length along axis 0 (since x and y are 1D vectors)
    x_norm = tf.nn.l2_normalize(x_padded, axis=0)
    y_norm = tf.nn.l2_normalize(y_padded, axis=0)

    # Compute the cosine similarity (dot product of normalized vectors)
    cosine_similarity = tf.reduce_sum(x_norm * y_norm, axis=0)

    # Cosine distance is 1 - cosine similarity
    cosine_dist = 1.0 - cosine_similarity
    return cosine_dist

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
    return base_loss

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

