import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):

        self._optim.zero_grad()
        # -propagate through the network
        output = self._model(x)
        # -calculate the loss
        loss = self._crit(output, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss.item()

    def val_test_step(self, x, y):

        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        pred = self._model(x)
        # propagate through the network and calculate the loss and predictions
        loss = self._crit(pred, y)
        # return the loss and the predictions
        return loss.item(), pred

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        self._model.train()
        train_loss = 0.0
        # iterate through the training set
        for i, (x, y) in enumerate(self._train_dl):
            # print("train epoch i: ", i)
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x, y = x.cuda(), y.cuda()
            # perform a training step
            train_loss += self.train_step(x, y)

        # calculate the average loss for the epoch and return it
        train_loss /= i
        return train_loss

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
            # iterate through the validation set
            validation_loss = 0.0
            pred_labels = []
            f1_scores = []
            auc_scores = []
            for i, (x, y) in enumerate(self._val_test_dl):
                # transfer the batch to the gpu if given
                if self._cuda:
                    x, y = x.cuda(), y.cuda()
                # perform a validation step
                loss, pred = self.val_test_step(x, y)
                validation_loss += loss
                # auc_scores.append(roc_auc_score(y.cpu(), pred.cpu()))
                f1_scores.append(f1_score(y.cpu(), pred.cpu() > 0.5, average="micro"))
                # save the predictions and the labels for each batch
                pred_labels.append((pred, y))

        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        validation_loss /= i
        print(f"Val Loss:{round(validation_loss, 3)} | F1:{round(sum(f1_scores) / len(f1_scores), 2)}")
        # return the loss and print the calculated metrics
        return validation_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        validation_losses = []
        epoch_counter = 0
        early_stop_counter = 0
        min_loss = float("inf")

        while True:

            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            print(f"Epoch: {epoch_counter}")
            # stop by epoch number
            if epoch_counter >= epochs:
                print("Training finished")
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            training_loss = self.train_epoch()
            validation_loss = self.val_test()

            # append the losses to the respective lists
            train_losses.append(training_loss)
            validation_losses.append(validation_loss)

            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if validation_loss < min_loss:
                print(
                    f"Epoch:{epoch_counter} | Model is saved. | Loss decreased from {min_loss} to {validation_loss}...")
                min_loss = validation_loss
                self.save_checkpoint(0)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if early_stop_counter >= self._early_stopping_patience:
                print("Early stopping...")
                break
            # return the losses for both training and validation
            epoch_counter += 1
        return train_losses, validation_losses
