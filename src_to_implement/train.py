import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('data.csv', sep=';')
epoch = 20
batch_size = 32
learning_rate = 1e-4
early_stop_patience = 30

len_data = len(data)
val_size = 0.2
#train_split = int((1-val_size)*len_data)
#train_data = data.iloc[0:train_split]
#val_data = data.iloc[train_split:len_data]

X_train, X_test, y_train, y_test = train_test_split(data, data, test_size=val_size, random_state=42)
# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules

train_ds = ChallengeDataset(X_train, 'train')
train_dl = t.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_ds = ChallengeDataset(X_test, 'val')
val_test_dl = t.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)


# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
model = model.ResNet()
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
crit = t.nn.BCELoss()
# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model=model, crit=crit, optim=optimizer, train_dl=train_dl, val_test_dl=val_test_dl, early_stopping_patience=early_stop_patience, cuda=True)
# go, go, go... call fit on trainer
res = trainer.fit(epoch)

# create an instance of our ResNet model


# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion


# go, go, go... call fit on trainer

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')