# GenerativeDisease

View our presentation here: https://docs.google.com/presentation/d/1p5pmAzQGegYffPu98yMIs9NvyDUnuPysgHVNScweZCU/edit?usp=sharing

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
