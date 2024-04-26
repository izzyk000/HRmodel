import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_inputs
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from joblib import dump  # Import joblib for saving preprocessing steps
import traceback

def load_data(filepath):
    try:
        return pd.read_excel(filepath, engine='openpyxl')
    except Exception as e:
        print(f"Failed to load data: {e}")
        raise

def train_model():
    try:
        # Load the dataset
        df = load_data('data/HR Dataset.xlsx')
        X_processed, y, preprocessor = preprocess_inputs(df, training=True)  # Ensure training=True is correctly used

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        # Convert labels to one-hot encoding
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        y_train = to_categorical(y_train).astype('float32')
        y_test = to_categorical(y_test).astype('float32')

        # Define the model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=32,
            epochs=100,
            verbose=1
        )

        # Save the model
        model.save('hr_recruitment_model.h5')
        
        # Save the fitted preprocessor
        dump(preprocessor, 'preprocessor.joblib')

        return history

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        traceback.print_exc()  # This will print the full traceback of the exception
        return None

if __name__ == '__main__':
    history = train_model()
    if history:
        # Display the final accuracy
        final_train_accuracy = history.history['accuracy'][-1]
        final_validation_accuracy = history.history['val_accuracy'][-1]
        print(f"Final training accuracy: {final_train_accuracy*100:.2f}%")
        print(f"Final validation accuracy: {final_validation_accuracy*100:.2f}%")
