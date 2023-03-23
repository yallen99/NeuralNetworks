# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values        #Excludes the upper bound (2)

# Feature scaling - apply normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 timesteps and 1 output
# 60 timesteps mean 60 previous 'financial days' the network will examine and 
# based on which try to predict the output for 'tomorrow'
X_train = [] 
Y_train = []
for i in range(60, 1258) :                              # Start the loop at pos 60 and end on the last pos
    X_train.append(training_set_scaled[i-60:i, 0])      # Take the prices from i-60 to i
    Y_train.append(training_set_scaled[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train) # Transform x_train and y_Train from lists to arrays

# Add an extra dimension that would act as an indicator fro predicting the open stock price
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))     # NewShape is the shape of the tensor required for the RNN. 3D(analyzed values, timesteps, indicators)
                                                                           # If you want to add more indicators, simply change the value in the last dimension

# Build the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout  # Required to avoid overfitting

regressor = Sequential()

# First LSTM layer
regressor.add(LSTM(units= 50, return_sequences= True, input_shape = (X_train.shape[1], 1)))             # units = number of neurons in the layer (50 is a high number since the problem is quite complex)
                                                                                                        # return_sequences = True because we're using a *stacked* LSTM architecture
                                                                                                        # input_Shape = the last 2 params of the 3D shape created in the data preprocessing
# Dropout regularization - drop 20% of neurons from each training iteration
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 50, return_sequences= True))             # Second LSTM layer - no need for the input_shape because is automatically detected with the 50 neurons
regressor.add(Dropout(0.2))                                        # Regularization

regressor.add(LSTM(units= 50, return_sequences= True))             # Third LSTM layer 
regressor.add(Dropout(0.2))                                        # Regularization

regressor.add(LSTM(units= 50))                                     # Fourth LSTM layer - no need for the return_sequences because is the last layer
regressor.add(Dropout(0.2))                                        # Regularization

# Output layer
regressor.add(Dense(units= 1)) 

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # A popular choice for recurrent NN is optimizer='RMSProp'

regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

# Making predictions

# Get the real prices from file
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values 

# Get the predicted prices by RNN
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis= 0)     # Concatenate the two price lists lines, on the 'Open' columns
                                                                                      # since we will need to access the prev 60 days for each new day in the predicted list

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :].values          # These inputs represent the range of days / data needed to predict each day in January
                                                                                      # For the first day predicted (Jan, 3rd) - the lower bound - we need the prev 60 days 
                                                                                      #     => index: dataset_total - dataset_test - 60
                                                                                      # For the last day predicted - the upper bound - we need the prev 60 days from the last predicted day 
                                                                                      #     => index: dataset_total.last     

inputs = inputs.reshape(-1, 1)                                                        # Reshape the array to a readable format since we've not used iloc[]

inputs = sc.transform(inputs)                                                         # Scale the inputs with the already fitted scaler

# Create the 3D structure to predict
X_test = [] 
for i in range(60, 80) :                  # Only 20 days to predict       
    X_test.append(inputs[i-60:i, 0])     

X_test = np.array(X_test)                                               # Convert to array
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))      # Apply 3rd dimension to array

# Predict values
predicted_stock_price = regressor.predict(X_test)

# Inverse scaled prices
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualize the results
plt.plot(real_stock_price, color='green', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='red', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()