import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('/content/ECGMODEL.h5')

labels = {
    0: "Normal",
    1: "Atrial Premature",
    2: "Premature ventricular contraction",
    3: "Fusion of ventricular and normal",
    4: "Fusion of paced and normal"
}

def make_predictions(test, model, labels):
    x_test = test.iloc[:, :187]
    y_test = test.iloc[:, 187]

    x_test = x_test.values
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    y_test = tf.keras.utils.to_categorical(y_test)

    y_pred = model.predict(x_test)

    y_pred_classes = np.argmax(y_pred, axis=1)

    predicted_labels = [labels[pred] for pred in y_pred_classes]

    return predicted_labels
    
def plot_ecg(test, n):
    plt.figure(figsize=(28, 12))  
    test.iloc[n, :187].plot(color='blue')  

    plt.title(f'ECG of Customer {n+1}', fontsize=16)
    plt.xlabel('Time (ms)', fontsize=14)
    plt.ylabel('Amplitude (mV)', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()

try:
    n = int(input("Which Customer ECG do you want to plot: "))
    if n < 0 or n >= len(test):
        print(f"Invalid customer index. Please enter a number between 0 and {len(test) - 1}.")
    else:
        plot_ecg(test, n)
except ValueError:
    print("Invalid input. Please enter a valid number.")

test = pd.read_csv('/path_to_data')
predicted_labels = make_predictions(test, model, labels)
print(predicted_labels)
