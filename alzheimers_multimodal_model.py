# Import required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Paths
DATA_DIR = "./alzheimers_data/"
CLINICAL_DATA_PATH = os.path.join(DATA_DIR, "oasis_cross-sectional.csv")
MRI_DATA_PATH = os.path.join(DATA_DIR, "oasis_longitudinal.npz")

# Load and preprocess clinical data
def load_clinical_data(path):
    print("Loading clinical data...")
    df = pd.read_csv(path)
    df = df.dropna(subset=['CDR', 'MMSE', 'eTIV', 'nWBV', 'ASF'])

    le = LabelEncoder()
    le.fit([0, 0.5, 1, 2])
    df['target'] = le.transform(df['CDR'])

    clinical_features = ['MMSE', 'eTIV', 'nWBV', 'ASF', 'Age', 'SES']
    for feature in clinical_features:
        if df[feature].isnull().sum() > 0:
            df[feature] = df[feature].fillna(df[feature].median())

    df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})
    clinical_features.append('M/F')

    y = df['target'].values
    X_clinical = df[clinical_features].values

    scaler = StandardScaler()
    X_clinical = scaler.fit_transform(X_clinical)

    # Save scaler and label encoder
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    return X_clinical, y, df, le.classes_, clinical_features

# Simulate or load MRI data CNN
def load_mri_data(path, num_samples):
    print("Loading MRI data...")
    try:
        data = np.load(path)
        X_mri = data['X_mri']
        print(f"Loaded preprocessed MRI data with shape: {X_mri.shape}")
    except:
        print("Simulating MRI data for demonstration...")
        X_mri = np.random.randn(num_samples, 64, 64, 1)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, X_mri=X_mri)
        print(f"Saved simulated MRI data with shape: {X_mri.shape}")
    return X_mri

# Build multimodal model
def build_multimodal_model(input_shape_mri, input_shape_clinical, num_classes):
    mri_input = Input(shape=input_shape_mri, name='mri_input')
    x = Conv2D(32, (3, 3), activation='relu')(mri_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    mri_features = Dropout(0.5)(x)

    clinical_input = Input(shape=(input_shape_clinical,), name='clinical_input')
    y = Dense(64, activation='relu')(clinical_input)
    y = Dropout(0.3)(y)
    y = Dense(32, activation='relu')(y)
    clinical_features = Dropout(0.3)(y)

    combined = concatenate([mri_features, clinical_features])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    output = Dense(num_classes, activation='softmax', name='output')(z)

    model = Model(inputs=[mri_input, clinical_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Main
def main():
    print("Starting Alzheimer's Stage Prediction (7-feature version)")

    X_clinical, y, df, classes, features_used = load_clinical_data(CLINICAL_DATA_PATH)
    X_mri = load_mri_data(MRI_DATA_PATH, len(X_clinical))

    X_mri_train, X_mri_test, X_clinical_train, X_clinical_test, y_train, y_test = train_test_split(
        X_mri, X_clinical, y, test_size=0.2, random_state=42, stratify=y)

    model = build_multimodal_model(X_mri.shape[1:], X_clinical.shape[1], len(np.unique(y)))
    model.summary()

    callbacks = [
        #EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
    ]

    print("\nTraining model...")
    history = model.fit(
        [X_mri_train, X_clinical_train], y_train,
        validation_data=([X_mri_test, X_clinical_test], y_test),
        epochs=1000,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating model...")
    loss, accuracy = model.evaluate([X_mri_test, X_clinical_test], y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")

    y_pred = np.argmax(model.predict([X_mri_test, X_clinical_test]), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model.save("alzheimers_multimodal_model.h5")

    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred, classes)

    print("\nCorrelation Heatmap (for used clinical features)...")
    plt.figure(figsize=(12, 8))
    corr = df[features_used + ['CDR']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig('feature_correlation.png')
    plt.show()

    print("✅ Training complete.")
if __name__ == "__main__":
    main()
