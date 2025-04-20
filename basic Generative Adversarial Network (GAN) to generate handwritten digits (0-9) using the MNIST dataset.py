import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train/255.0
#optional
# for i in range(16):
#   plt.subplot(4,4,i+1)
#   plt.imshow(x_train[i])
#   plt.axis('off')
# plt.suptitle("Sample MNIST digits")
# plt.show()
#---------------------------------
import numpy as np

# Function to generate random noise
def generate_noise(batch_size, latent_dim):
    return np.random.normal(0, 1, (batch_size, latent_dim))

# Try with 5 noise vectors of 100 length
noise = generate_noise(5, 100)
print(noise.shape)
print(noise[0])  # 1st noise vector
#-----------------------------------
# generator
from tensorflow import keras
#from tensorflow.keras import sequential  # Remove this line
from tensorflow.keras.layers import Dense

def generator():
  # Use keras.Sequential instead of sequential
  #sequentiall is a step by step process
  model = keras.Sequential([
      #Dense is a fully connected layer
      Dense(128, activation='relu', input_shape=(100,)),# relu for making the negitive values to positive values
      Dense(784, activation='sigmoid') # sigmoid for classification from 0-1
  ])
  return model

generator = generator()
generator.summary()
#------------------------------------
#Discriminator
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def discriminator():
  model = Sequential([
      Dense(128, activation=LeakyReLU(0.2), input_shape=(784,)),
      Dense(1,activation = 'sigmoid')
  ])
  return model

discriminator = discriminator()
discriminator.summary()
#------------------------------------
#create some fake and real images
real_imgs  = x_train[:10].reshape(10,784)
fake_imgs = generator.predict(generate_noise(10,100))
# Labels to detect which is fake and which is original
real_labels = np.ones((10,1))
fake_labels = np.zeros((10,1))
#optimizer ka matlab: Optimizer aur loss function set karke model ko sikhaane ke liye tayyar karna.
discriminator.compile(optimizer = 'adam', loss = 'binary_crossentropy')

d_loss_real = discriminator.train_on_batch(real_imgs,real_labels)
d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)

print(f"Loss on real images: {d_loss_real:.4f}")
print(f"Loss on fake images: {d_loss_fake:.4f}")
#-------------------------------------------------
# Generate one fake image using generator
noise = generate_noise(1, 100)
generated_image = generator.predict(noise)

# Reshape and display the image
generated_image = generated_image.reshape(28, 28)

plt.imshow(generated_image, cmap='gray')
plt.title("Generated Image from Generator")
plt.axis('off')
plt.show()
