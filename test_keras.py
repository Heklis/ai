
# coding: utf-8

# In[1]:


from tensorflow import keras

from keras import layers
from keras import models


# 加载mnist数据

# In[2]:


from keras.datasets import mnist
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# 构造网络

# In[16]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[18]:


model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# 编译网络

# In[19]:


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


# 训练网络

# In[22]:


model.fit(train_images, train_labels, epochs=5, batch_size=64,validation_data=[X_test, y_test])


# In[13]:


test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[14]:


print(test_acc)

