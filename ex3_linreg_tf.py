#!/usr/bin/env python
# coding: utf-8

# ### II/ Tìm nghiệm bài toán bằng `TensorFlow`
# 
# ##### 1) (Full) batch gradient descent: đưa toàn bộ X và Y vào để train:
# 
# Với cách 1, do đưa toàn bộ batch vào nên gradient ở mỗi vòng lặp ổn định. Cách này được khuyến khích sử dụng khi hàm cost của mình biết rõ là convex (không có nhiều hơn 1 điểm tối ưu cục bộ). Tuy nhiên, đối với những hàm phức tạp, thì cách 1 có thể ko bao giờ đạt tối ưu toàn cục được.
# 
# ##### 2) Stochastic gradient descent: đưa từng cặp (x, y) trong data X, Y vào để train :
# 
# Đối với cách 2, do mình đưa vào từng cặp nên gradient ở mỗi vòng lặp sẽ rất nhiễu (noisy). Chính vì sự nhiễu này mà có trong qúa trình học, nó có thể giúp mô hình vượt qua được các điểm tối ưu cục bộ. Stochastic = random, thể hiện cho sự nhiễu.
# ##### 3) Mini-batch gradient descent: bốc 1 lượng nhiều hơn 1 mẫu từ X, Y để train.
# 
# Cách 3 là sự kết hợp giữa 1 và 2, cũng là cách dùng nhiều nhất trong deep learning. Trong các bài tới sẽ đề cập sau.
# 
# Còn về bài tập thì thực ra hàm error của mình hoàn toàn convex nên dùng cách 1 hay 2 đều được. Nhưng cách 2 sẽ lâu hơn. Bạn có thể sửa code lại để kiểm tra thử.

# Đưa dữ liệu vào

# In[1]:


from utils_function import load_Boston_housing_data
import numpy as np
raw_train_X, test_X, train_Y, test_Y = load_Boston_housing_data(feature_ind = [4,5])


# #### Nhập thư viện

# In[2]:


# IMPORT
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.__version__ # '2.x'


# #### Khai báo biến

# In[16]:


learning_rate = 0.005
training_epochs = 10000
display_step = 1000
n_samples, dimension = raw_train_X.shape
batch_size = n_samples # Full Batch Gradient Descent


# #### Bài 6. Khai báo

# In[17]:


# Implement input and parameter for tensorflow.
train_X = tf.constant(raw_train_X, dtype=tf.float64)

train_Y = tf.reshape(tensor=train_Y, shape=(-1, 1))
train_Y = tf.constant(train_Y, dtype=tf.float64) # convert train_Y to tensor tf

# Set model weights
W = tf.Variable(np.random.normal(size=(dimension, 1)), trainable=True) # create weights variable to train
b = tf.Variable(np.random.normal(size=(1, 1)), trainable=True)
print(W)
print(b)


# #### Bài 7. Xây dựng mô hình hồi quy tuyến tính

# In[5]:


# GRADED FUNCTION
# TO_DO_6: implement a linear regression function
def tf_lr_hypothesis(X, W, b):
    return tf.add(tf.matmul(X, W), b)


# #### Bài 8. Viết hàm cost

# In[9]:


# GRADED FUNCTION
# TO_DO_7: implement a cost function
def tf_mse_cost(Y_hat, Y):    
    n_samples = Y.get_shape()[0]
    return tf.reduce_sum(tf.pow(Y_hat - Y, 2) / (2 * n_samples) )


# #### Bài 9. Viết hàm train

# In[7]:


# Create optimizer
optimizer = tf.optimizers.SGD(learning_rate=learning_rate)


# #### Bài 10. Chạy chương trình

# In[8]:


for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        Y_hat = tf_lr_hypothesis(train_X, W, b) # apply linear regression function here
        mse_cost = tf_mse_cost(Y_hat, train_Y) # apply mse cost here.
    grads = tape.gradient(mse_cost, [W, b])
    optimizer.apply_gradients(zip(grads, [W, b]))
    if (epoch + 1) % display_step == 0:
        print("Epoch:", epoch + 1, "| Cost:", mse_cost.numpy())


# ####  Bài 11. Tạo các đặc tính mới (Feature Engineering)

# In[9]:


X1 = raw_train_X[:,0].reshape((n_samples,1))
X2 = raw_train_X[:,1].reshape((n_samples,1))
X1_sqr = (X1**2).reshape((n_samples,1))
sin_X2 = (np.sin(X2)).reshape((n_samples,1))
X1X2 = (X1*X2).reshape((n_samples,1))
# Create new input from new features
new_train_X = np.concatenate((X1,X2,X1_sqr,sin_X2,X1X2),axis=1) # concatenate new features here


# ####  Bài 12. Khai báo

# In[10]:


# Implement input and parameter for tensorflow.
new_train_X = tf.constant(new_train_X, dtype=tf.float64)
_,new_dimension = new_train_X.shape
optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
train_Y = tf.reshape(tensor=train_Y, shape=(-1, 1))
train_Y = tf.constant(train_Y, dtype=tf.float64) # convert train_Y to tensor tf
training_epochs = 10000
display_step =1000
# set model weight
W = tf.Variable(np.random.normal(size=(new_dimension, 1)), trainable=True) # create weights variable to train
b = tf.Variable(np.random.normal(size=(1, 1)), trainable=True)


# ####  Bài 13. Chạy chương trình

# In[11]:


for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        Y_hat = tf_lr_hypothesis(new_train_X, W, b) # apply linear regression function here
        mse_cost = tf_mse_cost(Y_hat, train_Y) # apply mse cost here
    grads = tape.gradient(mse_cost, [W, b])
    optimizer.apply_gradients(zip(grads, [W, b]))
    if (epoch + 1) % display_step == 0:
        print("Epoch:", epoch + 1, "| Cost:", mse_cost.numpy())

