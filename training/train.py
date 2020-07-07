import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Masking, Dropout
from tensorflow.keras.callbacks import History
from data_preprocessing.preprocessing import prepare_data_train, gen_train, gen_target

sequence_length = 50
mask_value = 0

df_train = prepare_data_train(drop_cols=True)
feats = df_train.columns.drop(['UnitNumber', 'Cycle', 'RUL'])

with open('test_data/feats.pkl', "wb") as f:
    pickle.dump(feats, f)

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
df_train[feats] = min_max_scaler.fit_transform(df_train[feats])

x_train = np.concatenate(list(
    list(gen_train(df_train[df_train['UnitNumber'] == unit], sequence_length, feats)) for unit in
    df_train['UnitNumber'].unique()))

y_train = np.concatenate(list(
    list(gen_target(df_train[df_train['UnitNumber'] == unit], sequence_length, "RUL")) for unit in
    df_train['UnitNumber'].unique()))

# LSTM
nb_features = x_train.shape[2]
nb_out = 1

history = History()
with open('test_data/history.pkl', 'wb') as f:
    pickle.dump(history, f)

model = Sequential()
model.add(LSTM(
         units=256,
         return_sequences=True,
         input_shape=(sequence_length, nb_features)))
model.add(Dropout(0.2))
model.add(LSTM(
          units=128,
          return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='relu'))
model.add(Activation("relu"))
model.compile(loss="mse", optimizer="rmsprop", metrics=['mse'])

model.summary()

# fit the model
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1,
          callbacks=[history, tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                                       verbose=0, mode='auto')])

model.save('models/machine_1.h5')

scores = model.evaluate(x_train, y_train, verbose=1, batch_size=200)
print('MSE: {}'.format(scores[1]))
