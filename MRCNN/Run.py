

from __future__ import division
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1" #Chose one piece of NVIDIA Graphics or you can use all without this declaration
import gc
import numpy as np
import tensorflow as tf
import MRCNN.Model as model
import random




def next_batch(data1, data2, num, batch_size):
 for batch_i in range(num // batch_size):
  idx = np.arange(0, len(data1))
  np.random.shuffle(idx)
  idx = idx[0:batch_size] 
  data_shuffle1 = [data1[i] for i in idx]  
  data_shuffle1 = np.asarray(data_shuffle1)  
  data_shuffle2 = [data2[i] for i in idx]  
  data_shuffle2 = np.asarray(data_shuffle2)  
  data1 = np.delete(data1, idx, axis=0)
  data2 = np.delete(data2, idx, axis=0)
  
  yield data_shuffle1, data_shuffle2

  
  
  

def batch(size, data1, data2):
    idx = np.arange(0, len(data1))  
    np.random.shuffle(idx)  
    idx = idx[0:size]  
    data_shuffle1 = [data1[i] for i in idx]  
    data_shuffle1 = np.asarray(data_shuffle1)  
    data_shuffle2 = [data2[i] for i in idx]  
    data_shuffle2 = np.asarray(data_shuffle2)  
	
    return data_shuffle1, data_shuffle2

	
	
	
	
def main(feature,label):
    results = []
    x = tf.placeholder(tf.float32, [None, 400, 4, 1])
    y = tf.placeholder(tf.float32, [None, 1])
    train_step, loss_op, accuracy = model.net_MRCNN(x,y,0.5)
    init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    '''
    Initial parameter setting
    '''
    batch_size = 20
    EP  =  100
    print("shape", feature.shape)

    keep_prob = tf.placeholder(tf.float32)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(EP):
            print("EPOCH", epoch)
            b =1
            for fs,lb in next_batch(feature,label,len(feature),batch_size):
                fs = fs.reshape(batch_size, 400, 4, 1)
                lb = lb.reshape(batch_size,1)
                sess.run(train_step,feed_dict={x:fs,y:lb,keep_prob: 0.5})
                Loss, acc = sess.run([loss_op, accuracy],feed_dict={x:fs,y:lb,keep_prob: 1.0})

                #for b in range(batch_size):
                print("Batch: ", b, "MSE: ",Loss.mean())
                print("Accuracy ", acc)
                b += 1


final = []
result = []
for i in range(1000):
    batch = []
    for x in range(400):
        array = [0,0,0,0]
        array[random.randint(0, 3)] = 1
        batch.append(array)
    final.append(batch)
    result.append(random.uniform(0,1))
main(np.array(final), np.array(result))



  #  #if(epoch==EP-1):
  #       '''
	# 	Save predicted results
	# 	'''
	# 	#fs_conv1 = sess.run(h_conv1,feed_dict={x:feature,y:label,keep_prob:1})
	# 	#fs_conv2 = sess.run(h_conv2,feed_dict={x:feature,y:label,keep_prob:1})
	# 	'''
	# 	Save the required variables
	# 	'''
	# 	#conv2  = sess.run(h_conv2,feed_dict={x:x_ts1,y:y_ts1,keep_prob:1})
	# 	#pool2  = sess.run(h_pool2,feed_dict={x:x_ts1,y:y_ts1,keep_prob:1})
	# 	#np.save(fs_name1,fs_conv1)
	# 	#np.save(fs_name2,fs_conv2)
	# 	#np.save(h_name,conv2)
	# 	#np.save(p_name,pool2)
  # #saver.save(sess, "~./model", global_step=100)