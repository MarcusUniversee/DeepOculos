# DeepOculos: Generative AI For Eye Retina Disease Tracking
by Timothy Gao, Henry Hong, Marcus Koh, Ohm Rajpal, and Henry Lee

**Second-Place Winner at Berkeley's 2023 Annual Datathon for Social Good**

Slides: https://docs.google.com/presentation/d/1p5pmAzQGegYffPu98yMIs9NvyDUnuPysgHVNScweZCU/edit#slide=id.g1388be8e982_0_3

# Generating Synthetic Retinal Eye Motion Videos with Deep Convolutional GAN

Original Retinal Eye Motion Video (data provided by C. Light Technologies):
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/umxOWhzmLYs/0.jpg)](https://www.youtube.com/watch?v=umxOWhzmLYs)

512x512

Our DCGAN-generated Synthetic Eye Motion Video (with embedded pupil trace statistics):
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/TMXJJJAKSsQ/0.jpg)](https://www.youtube.com/watch?v=TMXJJJAKSsQ)

64x64 (upscaled with cubic interpolation)

DCGAN Training Progress:
- First 1000 Epochs
<img width="320" alt="Screenshot 2023-11-12 at 10 23 30 PM" src="https://github.com/MarcusUniversee/GenerativeDisease/assets/35588167/7a328986-45de-4766-bfaa-f50f040e3161">
<img width="320" alt="Screenshot 2023-11-12 at 10 23 41 PM" src="https://github.com/MarcusUniversee/GenerativeDisease/assets/35588167/85be834c-2381-4e82-b578-115fc131ffc5">
<img width="320" alt="Screenshot 2023-11-12 at 10 23 49 PM" src="https://github.com/MarcusUniversee/GenerativeDisease/assets/35588167/73088109-1453-4845-8cc4-92bad22fd3cd">
<img width="320" alt="Screenshot 2023-11-12 at 10 24 00 PM" src="https://github.com/MarcusUniversee/GenerativeDisease/assets/35588167/d4f136bd-e965-4036-8bb0-831fd7af37e2">
<img width="320" alt="Screenshot 2023-11-12 at 10 24 19 PM" src="https://github.com/MarcusUniversee/GenerativeDisease/assets/35588167/5e452afe-e1dd-4b08-81d7-c94e99e28801">
<img width="320" alt="Screenshot 2023-11-12 at 10 24 38 PM" src="https://github.com/MarcusUniversee/GenerativeDisease/assets/35588167/7c31d2ca-b560-40d7-93bc-65989078d1b8">

- 2000 Epochs
<img width="600" alt="Screenshot 2023-11-12 at 10 25 10 PM" src="https://github.com/MarcusUniversee/GenerativeDisease/assets/35588167/7fc79b7d-6d4d-4d7d-a6d5-b5e88a56309e">

- 3000 Epochs
<img width="600" alt="Screenshot 2023-11-12 at 10 25 20 PM" src="https://github.com/MarcusUniversee/GenerativeDisease/assets/35588167/e8ee9c6f-8368-4c3e-b6b0-28a0600a37e6">

# Generating Eye Motion Trace Statistics Via Long Short-Term Memory Network (LSTM)

- We view horizontal and vertical location of pupil as independent time series and apply LSTM to generate future prediction for each
- LSTM-generated locations are fed into DCGAN for the pupil trace statistics

<img width="300" alt="Screenshot 2023-11-12 at 10 11 41 PM" src="https://github.com/MarcusUniversee/GenerativeDisease/assets/35588167/1846138a-e3a5-4bf3-971e-727fc72057fd">
<img width="300" alt="Screenshot 2023-11-12 at 10 11 13 PM" src="https://github.com/MarcusUniversee/GenerativeDisease/assets/35588167/94ab72ba-e1d2-4e5c-b681-bf19164c893a">
<img width="300" alt="Screenshot 2023-11-12 at 10 11 30 PM" src="https://github.com/MarcusUniversee/GenerativeDisease/assets/35588167/7a400de0-6e11-4825-9d97-f8d76864307d">

The Challenge:
Use generative AI to create artificial retinal video sequences & eye motion traces given a disease state. 

Implementation Details:
1. Used a four-layer LSTM network with dropout regularization after each LSTM layer and a single dense neuron output layer
2. Trained over 50 epochs, with rmsprop optimizer and mean_squared_error as loss function
3. Utilized first 80% of each time series as training data, last 20% as testing data
4. Used MinMaxScaler to normalize all pixel values to [-1, 1] domain
5. Used previous 60 frames as context for each sliding window

Technologies Used:
1. Tensorflow: To build LSTMs, GANs
2. Scikit-Learn: MinMaxScaler and Train Test Split
3. Keras: Deep learning API (Sequential Model and neural network layers)
4. Seaborn and Matplotlib: Data Visualization
5. NumPy: For calculations and array manipulation
6. Pandas: For processing CSV files 
