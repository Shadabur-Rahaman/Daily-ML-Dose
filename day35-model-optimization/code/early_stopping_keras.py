# Early Stopping in Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Dummy Data
X, y = make_classification(n_samples=1000, n_features=20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=50, batch_size=32, callbacks=[early_stop])
