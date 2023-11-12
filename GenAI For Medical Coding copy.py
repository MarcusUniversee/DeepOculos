#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
import time 
import PIL
from IPython import display

def get_training_data(image_path):
    """Loads an image from the given path into a numpy array."""
    with Image.open(image_path) as img:
        return np.array(img)


# In[7]:


df = pd.read_csv('./eye_motion_trace/00029_U_4_19_2018_9_18_9_V001.csv')


# # Actual project

# In[8]:


def extract_frame(video_path, time_in_seconds):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame number to be captured
    frame_number = int(time_in_seconds * fps)

    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    # Release the video capture object
    cap.release()

    if ret:
        return frame  # Return the frame if successfully captured
    else:
        print("Error: Could not retrieve frame at given time.")
        return None


# ### Extracting data

# In[9]:


def convert_combine(arrays):
    grayscaled = [np.mean(array, axis=2, keepdims=True) for array in arrays]
    combined = np.stack(grayscaled, axis=0)
    
    # list of all the frames
    
    return combined
    # extract_frame('./mTBI/video_sequence/00029_U_4_19_2018_9_18_9_V001.avi', 1).shape

def avi_parser(avi_file="./video_sequence/00029_U_4_19_2018_9_18_9_V001.avi"):
    temp = []
    
    # avi_file = "00029_U_4_19_2018_9_18_9_V001.avi"
    # Open the video file
    cap = cv2.VideoCapture(avi_file)
    
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        # Loop through each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if there are no frames left
    
            # Process the frame here
            # For example, you can display the frame using cv2.imshow("Frame", frame)
            # print(type(frame))
            temp.append(frame)
    
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    return temp


# In[10]:


avi_obj = avi_parser("./video_sequence/00029_U_4_19_2018_9_18_9_V001.avi")
avi_obj[0].shape


# 

# In[11]:


avi_train = convert_combine(avi_obj)
type(avi_train)


# In[12]:


def shape_finder(csv_directory= './eye_motion_trace'):
    # Directory containing CSV files
    csv_directory = './eye_motion_trace'
    
    # Loop through each file in the directory
    for filename in os.listdir(csv_directory):
        if filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(csv_directory, filename)
    
            # Read the CSV file into a DataFrame
            temp = pd.read_csv(file_path)
    
            # Print the filename and shape of the DataFrame
            print(f"File: {filename}, Shape: {temp.shape[0]}")


# In[13]:


avi_train = avi_train.reshape(avi_train.shape[0], 512, 512, 1).astype('float32')
avi_train_images = (avi_train - 127.5) / 127.5  # Normalize the images to [-1, 1]
avi_train_images.shape


# ### Getting noise from vectors

# In[14]:


df_array = df.drop(columns=['time[s]']).to_numpy()


# In[15]:


df_array


# In[16]:


# Number of rows in the original array
num_rows = df_array.shape[0]

# Generate noise (random numbers, for example)
# Adjust the parameters of np.random.rand() as needed
noise = np.random.rand(num_rows, 92)

# Concatenate the original array and the noise array
combined_array = np.concatenate([df_array, noise], axis=1)
combined_array


# 

# # Side quest

# In[17]:


BUFFER_SIZE = 60000
BATCH_SIZE = 256


# In[18]:


# Batch and shuffle the data
data_set = tf.data.Dataset.from_tensor_slices(avi_train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# In[19]:


def make_generator_model():
    model = tf.keras.Sequential()
    # Start with a dense layer that reshapes into a 8x8x1024 tensor
    model.add(layers.Dense(8*8*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 1024)))
    # Size becomes 8x8x1024 here

    # Upsample to 16x16
    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # Size becomes 16x16x512 here

    # Upsample to 32x32
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # Size becomes 32x32x256 here

    # Upsample to 64x64
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # Size becomes 64x64x128 here

    # Upsample to 128x128
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # Size becomes 128x128x64 here

    # Upsample to 256x256
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # Size becomes 256x256x32 here

    # Upsample to 512x512
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # Final size becomes 512x512x1

    return model


# In[20]:


generator = make_generator_model()


# In[21]:


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[512, 512, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# In[22]:


discriminator = make_discriminator_model()


# In[23]:


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# In[24]:


# descrim loss function
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# In[25]:


# generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# In[26]:


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# In[27]:


# checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# In[28]:


EPOCHS = 10
noise_dim = 100
num_examples_to_generate = 1

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = combined_array[0]


# In[29]:


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      print(images.shape)
      real_output = discriminator(images, training=True)
      print('DEBUB NOISE:', noise)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# In[30]:


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    for image_batch in dataset:
      train_step(image_batch)
    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)
    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)


# In[31]:


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


# In[ ]:


# Train the model
train(data_set, EPOCHS)


# In[ ]:


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# In[ ]:


# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


# In[ ]:


display_image(EPOCHS)

