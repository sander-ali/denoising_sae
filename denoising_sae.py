from tensorflow import keras 
from keras.models import Model 
from keras import Input 
from keras.layers import Dense, LeakyReLU, BatchNormalization 
from tensorflow.keras.utils import plot_model

import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt
import graphviz # for showing model diagram
import sys
import os

main_dir=os.path.dirname(sys.path[0])
print(main_dir)

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

fig, axs = plt.subplots(2, 5, sharey=False, tight_layout=True, figsize=(12,6), facecolor='white')
n=0
for i in range(0,2):
    for j in range(0,5):
        axs[i,j].matshow(X_train[n])
        axs[i,j].set(title=y_train[n])
        n=n+1
plt.show() 

level_of_noise=0.5

X_train_noisy = X_train + level_of_noise * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + level_of_noise * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

fig, axs = plt.subplots(2, 5, sharey=False, tight_layout=True, figsize=(12,6), facecolor='white')
n=0
for i in range(0,2):
    for j in range(0,5):
        axs[i,j].matshow(X_train_noisy[n])
        axs[i,j].set(title=y_train[n])
        n=n+1
plt.show() 

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train_noisy = X_train_noisy.reshape(60000, 784)
X_test_noisy = X_test_noisy.reshape(10000, 784)


n_inputs=X_train.shape[1] 
visible = Input(shape=(n_inputs,), name='Input-Layer') 

e = Dense(units=n_inputs, name='Encoder-Layer')(visible)
e = BatchNormalization(name='Encoder-Layer-Normalization')(e)
e = LeakyReLU(name='Encoder-Layer-Activation')(e)

middle = Dense(units=n_inputs, activation='linear', activity_regularizer=keras.regularizers.L1(0.0001), name='Middle-Hidden-Layer')(e)

d = Dense(units=n_inputs, name='Decoder-Layer')(middle)
d = BatchNormalization(name='Decoder-Layer-Normalization')(d)
d = LeakyReLU(name='Decoder-Layer-Activation')(d)

output = Dense(units=n_inputs, activation='sigmoid', name='Output-Layer')(d)

model = Model(inputs=visible, outputs=output, name='Denoising-Autoencoder-Model')

model.compile(optimizer='adam', loss='mse')

print(model.summary())

#plot_model(model, to_file=main_dir+'/pics/Denoising_Autoencoder.png', show_shapes=True, dpi=300)


history = model.fit(X_train_noisy, X_train, epochs=50, batch_size=64, verbose=1, validation_data=(X_test_noisy, X_test))

fig, ax = plt.subplots(figsize=(16,9), dpi=300)
plt.title(label='Model Loss by Epoch', loc='center')

ax.plot(history.history['loss'], label='Training Data', color='black')
ax.plot(history.history['val_loss'], label='Test Data', color='red')
ax.set(xlabel='Epoch', ylabel='Loss')
plt.xticks(ticks=np.arange(len(history.history['loss'])), labels=np.arange(1, len(history.history['loss'])+1))
plt.legend()

plt.show()

X_test_back = X_test.reshape(10000, 28, 28)

fig, axs = plt.subplots(2, 5, sharey=False, tight_layout=True, figsize=(12,6), facecolor='white')
n=0
for i in range(0,2):
    for j in range(0,5):
        axs[i,j].matshow(X_test_back[n])
        axs[i,j].set(title=y_test[n])
        n=n+1
plt.show() 

# Reshape back
X_test_noisy_reshaped = X_test_noisy.reshape(10000, 28, 28)

# Display images of the first 10 digits
fig, axs = plt.subplots(2, 5, sharey=False, tight_layout=True, figsize=(12,6), facecolor='white')
n=0
for i in range(0,2):
    for j in range(0,5):
        axs[i,j].matshow(X_test_noisy_reshaped[n])
        axs[i,j].set(title=y_test[n])
        n=n+1
plt.show() 

# Apply denoising model 
#X_train_denoised = model.predict(X_train_noisy).reshape(60000, 28, 28)
X_test_denoised = model.predict(X_test_noisy).reshape(10000, 28, 28)

# Display images of the first 10 digits
fig, axs = plt.subplots(2, 5, sharey=False, tight_layout=True, figsize=(12,6), facecolor='white')
n=0
for i in range(0,2):
    for j in range(0,5):
        axs[i,j].matshow(X_test_denoised[n])
        axs[i,j].set(title=y_test[n])
        n=n+1
plt.show() 

# Save the encoder model to file
model.save(main_dir+'dn_ae.h5')