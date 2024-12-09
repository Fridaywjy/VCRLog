import os
import sys
sys.path.append("../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from official.nlp import optimization
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from VCRLog.models import VCRLog
from VCRLog.utils import classification_report
from Data import load
import time
embed_dim = 768 # Embedding size for each token
max_len = 75

class BatchGenerator(Sequence):

    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        # print(self.batch_size)
        dummy = np.zeros(shape=(embed_dim,))
        x = self.X[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.X))]
        X = np.zeros((len(x), max_len, embed_dim))
        Y = np.zeros((len(x), 2))
        item_count = 0
        for i in range(idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.X))):
            x = self.X[i]
            if len(x) > max_len:
                x = x[-max_len:]
            x = np.pad(np.array(x), pad_width=((max_len - len(x), 0), (0, 0)), mode='constant',
                       constant_values=0)
            X[item_count] = np.reshape(x, [max_len, embed_dim])
            Y[item_count] = self.Y[i]
            item_count += 1
        return X[:], Y[:, 0]


def train_generator(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                    epoch_num, model_name=None):
    epochs = epoch_num
    steps_per_epoch = num_train_samples
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 1e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    loss_object = SparseCategoricalCrossentropy()

    model = VCRLog(768, ff_dim=2048, max_len=75, num_heads=12, dropout=0.1)


    model.compile(loss=loss_object, metrics=['accuracy'],
                  optimizer=optimizer)

    print(model.summary())

    # checkpoint
    filepath = model_name
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
    )
    callbacks_list = [checkpoint, early_stop]

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=int(num_train_samples / batch_size),
                        epochs=epoch_num,
                        verbose=1,
                        validation_data=validate_generator,  
                        validation_steps=int(num_val_samples / batch_size),
                        workers=16,
                        max_queue_size=32,
                        callbacks=callbacks_list,
                        shuffle=True
          )
    return model


def train(X, Y, epoch_num, batch_size, model_file=None):
    X, Y = shuffle(X, Y)
    n_samples = len(X)
    train_x, train_y = X[:int(n_samples * 90 / 100)], Y[:int(n_samples * 90 / 100)]
    val_x, val_y = X[int(n_samples * 90 / 100):], Y[int(n_samples * 90 / 100):]

    training_generator, num_train_samples = BatchGenerator(train_x, train_y, batch_size), len(train_x)
    validate_generator, num_val_samples = BatchGenerator(val_x, val_y, batch_size), len(val_x)

    print("Number of training samples: {0} - Number of validating samples: {1}".format(num_train_samples,
                                                                                       num_val_samples))

    model = train_generator(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                            epoch_num, model_name=model_file)

    return model


def test_model(model, x, y, batch_size):
    x, y = shuffle(x, y)
    x, y = x[: len(x) // batch_size * batch_size], y[: len(y) // batch_size * batch_size]
    test_loader = BatchGenerator(x, y, batch_size)
    prediction = model.predict_generator(test_loader, steps=(len(x) // batch_size), workers=16, max_queue_size=32,
                                         verbose=1)
    print(prediction)
    probility = np.max(prediction, axis=1)
    print(probility)
    prediction = np.argmax(prediction, axis=1)
    y = y[:len(prediction)]
    report = classification_report(np.array(y), prediction)
    # report = precision_recall_fscore_support(np.array(y), prediction, average='weighted')
    cm = confusion_matrix(np.array(y),prediction)
    print(report)
    TN, FP, FN, TP = cm.ravel()
    print("True Positive:", TP)
    print("False Negative:", FN)
    print("False Positive:", FP)
    print("True Negative:", TN)
    FPR = FP/(FP + TN)
    print("FPR: " + str(FPR))
    SPEC = TN/(TN + FP)
    print("SPEC: " + str(SPEC))
    # fpr, tpr, thresholds = roc_curve(np.array(y), prediction)
    # roc_auc = auc(fpr, tpr)
    # print(roc_auc)

def set_random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
if __name__ == '__main__':
    start_time = time.time()
    set_random_seed(2)
    # (x_tr, y_tr), (x_te, y_te) = load.load_HDFS(
    #     "log/Data/HDFS.log", "log/Data/anomaly_label.csv", e_type="bert", train_ratio=0.01, split_type='uniform')
    (x_tr, y_tr), (x_te, y_te) = load.load_data(
        "log/Data/HDFS.log", "log/Data/anomaly_label.csv", train_ratio=0.8, split_type='uniform')
    # (x_tr, y_tr), (x_te, y_te) = load.ab_load_data(
    #     "log/Data/HDFS.log", "log/Data/anomaly_label.csv", train_ratio=0.1, split_type='uniform')
    model = train(x_tr, y_tr, 10, 64, "HDFS_transformer.hdf5")
    test_model(model, x_te, y_te, batch_size=1024)
    end_time = time.time()
    run_time = end_time - start_time