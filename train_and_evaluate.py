import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.optimizers import schedules
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping

from spektral.data import DisjointLoader
from spektral.data.dataset import Dataset
from spektral.data.graph import Graph
from spektral.layers import ECCConv, GlobalAttentionPool
from spektral.models.gnn_explainer import GNNExplainer

################################################################################
# Initialize data and configure 
################################################################################

# Load data and create dataset
V = pickle.load(open("V.pickle","rb"))
A = pickle.load(open("A.pickle","rb"))
E = pickle.load(open("E.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

# Class to retrieve ionization dataset
class Ionization(Dataset):
	def read(self):
		return [Graph(x=x, a=adj, e=e, y=y) for x, adj, e, y in zip(V, A, E, y)]
data = Ionization()
n_out = data.n_labels

# Train/valid/test split
idxs = np.random.permutation(len(data))
split_va, split_te = int(0.7 * len(data)), int(0.8 * len(data))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
data_tr = data[idx_tr]
data_va = data[idx_va]
data_te = data[idx_te]

## Train/test split
#split = open('split_9.txt', 'r').read()
#idx_te = np.array([int(x.strip('\n')) for x in re.findall(r'test: \[(.*?)\]', split,re.DOTALL)[0].split(' ') if x])
#tr_indx = np.array([int(x.strip('\n')) for x in re.findall(r'training: \[(.*?)\]',split,re.DOTALL)[0].split(' ') if x])
#
#split_va = int(0.875 * len(list(tr_indx)))
#idx_tr, idx_va = np.split(tr_indx, [split_va])
#
#data_te = data[idx_te] 
#data_tr = data[idx_tr]
#data_va = data[idx_va]

# Configure 
epochs = 5
es_patience = 25 
batch_size = 64 

# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(data_va, batch_size=batch_size)
loader_te = DisjointLoader(data_te, batch_size=batch_size)


################################################################################
# Build model
################################################################################
class Net(Model):
	def __init__(self):
		super().__init__()
		self.batchnorm = BatchNormalization()
		self.conv1 = ECCConv(64,activation='relu')
		self.conv2 = ECCConv(64,activation='relu')
		self.global_pool = GlobalAttentionPool(32)
		self.dense2 = Dense(n_out)

	def call(self,inputs):
		x, a, e, i = inputs
		x = self.batchnorm(x)
		x = self.conv1([x, a, e])
		x = self.conv2([x, a, e])
		x = self.global_pool([x,i])
		output = self.dense2(x)

		return(output)

model = Net()


################################################################################
# Fit model
################################################################################
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)

# training step
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(mean_squared_error(target, predictions))
    return loss, acc

# evaluation step
def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(mean_squared_error(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

epoch = step = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
start_learning_rate = 0.001 
learning_rate = start_learning_rate
optimizer = Adamax(learning_rate=learning_rate)
loss_fn = MeanAbsoluteError()
results = []
for batch in loader_tr:
    step += 1
    loss, acc = train_step(*batch)
    results.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1
        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va)
        te_loss, te_acc = evaluate(loader_te)
        print(
            "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Test loss: {:.3f}".format(
                epoch, *np.mean(results, 0), val_loss, te_loss
            )
        )
        # Check if loss improved for early stopping
        if best_val_loss - val_loss > 0.001:
            best_val_loss = val_loss
            patience = es_patience
            learning_rate = learning_rate
            optimizer.learning_rate.assign(learning_rate)
            print("New best val_loss {:.3f}".format(val_loss), "Learning rate {:.3f}".format(learning_rate))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                 model.set_weights(best_weights)
                 learning_rate = learning_rate * 0.5
                 if learning_rate < 0.0000001:
                    print("Early stopping (best val_loss: {})".format(best_val_loss))
                    break
                 else:
                     optimizer.learning_rate.assign(learning_rate)
                     patience = es_patience 
                     print("New learning rate is " + str(learning_rate))
        results = []

################################################################################
# Evaluate model
################################################################################
model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_te)
print("Done. Test loss: {:.4f}. Test acc: {:.2f}".format(test_loss, test_acc))
model.summary()







