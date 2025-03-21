import numpy as np
import torch
import torchvision
from torchvision import transforms
import dill
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from math import cos, pi
import wandb
import multiprocessing

torch.manual_seed(42)


class Relu:
    def __init__(self):
        pass

    def __call__(self, x):
        self.x = x
        return torch.maximum(x, torch.tensor(0.0, device=x.device))

    def backward(self, grad):
        return grad * (self.x > 0).float()


class Softmax:
    def __init__(self, dim=-1, device=None):
        self.dim = dim
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def __call__(self, x):
        x = x.to(self.device)
        x_max = torch.max(x, dim=self.dim, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        sum_exp_x = torch.sum(exp_x, dim=self.dim, keepdim=True)
        self.softmax_output = exp_x / sum_exp_x
        return self.softmax_output

    def backward(self, grad_output):
        batch_size, num_classes = self.softmax_output.shape
        eye = torch.eye(num_classes, device=self.device).unsqueeze(0)
        softmax_diag = self.softmax_output.unsqueeze(2) * eye
        softmax_outer = torch.matmul(
            self.softmax_output.unsqueeze(2), self.softmax_output.unsqueeze(1)
        )
        jacobian = softmax_diag - softmax_outer
        grad_input = torch.matmul(jacobian, grad_output.unsqueeze(2)).squeeze(2)
        return grad_input


class CrossEntropy:
    def __init__(self, device=None, l2_reg=False, l=0.01):
        self.device = device
        self.l2_reg = l2_reg
        self.l = l
        self.weights = None
        self.softmax = Softmax(dim=-1, device=self.device)

    def __call__(self, logits, y, weights=None):
        self.weights = weights if weights is not None else []
        probs = self.softmax(logits)
        self.probs = torch.clamp(probs, min=1e-8)
        loss = -torch.mean(torch.sum(y * torch.log(self.probs), dim=1))
        if self.l2_reg and self.weights:
            loss += (self.l / 2) * sum(torch.sum(w**2) for w in self.weights)
        return loss

    def backward(self, logits, y):
        batch_size = logits.shape[0]
        grad = (self.probs - y) / batch_size
        return grad


class Linear:
    def __init__(self, input_size, output_size, activation, device=None, dropout=1.0):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.w = torch.randn(output_size, input_size, device=self.device) * (
            (2 / input_size) ** 0.5
        )
        self.b = torch.zeros(output_size, device=self.device)
        self.activation = activation
        self.dropout = dropout
        self.mask = None
        self.training_mode = False

    def __call__(self, x, training=True):
        self.training_mode = training
        self.x = x.to(self.device)
        self.z = torch.matmul(self.x, self.w.T) + self.b
        self.a_activation = self.activation(self.z)
        if self.training_mode and self.dropout < 1.0:
            self.mask = (
                torch.rand(self.a_activation.shape, device=self.device) < self.dropout
            ).float()
            self.a = self.a_activation * self.mask / self.dropout
        else:
            self.a = self.a_activation
        return self.a

    def backward(self, grad):
        if self.training_mode and self.dropout < 1.0:
            grad = grad * self.mask / self.dropout
        if isinstance(self.activation, (Softmax, Relu)):
            grad = self.activation.backward(grad)
        dw = torch.matmul(grad.T, self.x)
        db = torch.sum(grad, dim=0)
        dx = torch.matmul(grad, self.w)
        return dx, dw, db


class Conv2D:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation,
        stride=1,
        padding=0,
        device=None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.device = device
        init_scale = (2.0 / in_channels) ** 0.5
        self.w = (
            torch.randn(
                out_channels, in_channels, kernel_size, kernel_size, device=device
            )
            * init_scale
        )
        self.b = torch.zeros(out_channels, device=device)
        self.x_padded = None
        self.output_shape = None

    def forward(self, x):
        self.x_padded = F.pad(x, (self.padding,) * 4)
        batch_size, _, in_h, in_w = x.shape
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.output_shape = (batch_size, self.out_channels, out_h, out_w)
        x_unfolded = F.unfold(self.x_padded, self.kernel_size, stride=self.stride)
        w_reshaped = self.w.view(self.out_channels, -1)
        out = torch.matmul(w_reshaped, x_unfolded) + self.b.view(-1, 1)
        out = out.view(self.output_shape)
        if self.activation:
            return self.activation(out)
        return out

    def backward(self, grad):
        grad = grad.reshape(self.output_shape)
        if isinstance(self.activation, Relu):
            grad = self.activation.backward(grad)
        batch_size = self.x_padded.shape[0]
        grad_unfolded = grad.reshape(batch_size, self.out_channels, -1)
        x_unfolded = F.unfold(self.x_padded, self.kernel_size, stride=self.stride)
        dw = (
            torch.matmul(grad_unfolded, x_unfolded.transpose(1, 2))
            .sum(0)
            .view(self.w.shape)
        )
        db = grad.sum(dim=(0, 2, 3))
        w_reshaped = self.w.view(self.out_channels, -1)
        dx_unfolded = torch.matmul(w_reshaped.T, grad_unfolded)
        dx = F.fold(
            dx_unfolded,
            output_size=self.x_padded.shape[-2:],
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        if self.padding > 0:
            dx = dx[:, :, self.padding : -self.padding, self.padding : -self.padding]
        return dx, dw, db

    def __call__(self, x):
        return self.forward(x)


class MaxPool2D:
    def __init__(self, kernel_size, stride, device=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.indices = None

    def __call__(self, x):
        x = x.to(self.device)
        out, self.indices = F.max_pool2d(
            x, self.kernel_size, self.stride, return_indices=True
        )
        return out

    def backward(self, grad):
        grad = grad.to(self.device)
        return F.max_unpool2d(grad, self.indices, self.kernel_size, stride=self.stride)


class BatchNorm2D:
    def __init__(self, num_features, device, eps=1e-5, momentum=0.1):
        self.gamma = torch.ones(
            1, num_features, 1, 1, device=device, requires_grad=False
        )
        self.beta = torch.zeros(
            1, num_features, 1, 1, device=device, requires_grad=False
        )
        self.eps = eps
        self.momentum = momentum
        self.running_mean = torch.zeros(1, num_features, 1, 1, device=device)
        self.running_var = torch.ones(1, num_features, 1, 1, device=device)
        self.training = True

    def __call__(self, x):
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var
            self.x_hat = F.batch_norm(
                x,
                batch_mean,
                batch_var,
                self.gamma.squeeze(),
                self.beta.squeeze(),
                training=True,
                eps=self.eps,
            )
            return self.x_hat
        else:
            self.x_hat = F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.gamma.squeeze(),
                self.beta.squeeze(),
                training=False,
                eps=self.eps,
            )
            return self.x_hat

    def backward(self, grad_output):
        return grad_output

    def get_params(self):
        return [self.gamma, self.beta]


class Flatten:
    def __init__(self, device=None):
        self.device = device
        self.original_shape = None

    def __call__(self, x):
        self.original_shape = x.shape
        return torch.flatten(x, 1)

    def backward(self, grad):
        return grad.reshape(self.original_shape)


class Adam:
    def __init__(self, learning_rate, beta1, beta2):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 1

    def optimize(self, dw, db, layer):
        if not hasattr(self, "vw") or self.vw.shape != dw.shape:
            self.vw = torch.zeros_like(dw)
            self.vb = torch.zeros_like(db)
            self.sw = torch.zeros_like(dw)
            self.sb = torch.zeros_like(db)

        self.vw = self.beta1 * self.vw + (1 - self.beta1) * dw
        self.vb = self.beta1 * self.vb + (1 - self.beta1) * db
        self.sw = self.beta2 * self.sw + (1 - self.beta2) * dw.pow(2)
        self.sb = self.beta2 * self.sb + (1 - self.beta2) * db.pow(2)

        vw_hat = self.vw / (1 - self.beta1**self.t)
        vb_hat = self.vb / (1 - self.beta1**self.t)
        sw_hat = self.sw / (1 - self.beta2**self.t)
        sb_hat = self.sb / (1 - self.beta2**self.t)

        layer.w -= self.lr * vw_hat / (sw_hat.sqrt() + 1e-8)
        layer.b -= self.lr * vb_hat / (sb_hat.sqrt() + 1e-8)
        self.t += 1


def step_decay(epoch, initial_lr=1e-3):
    if epoch < 3:
        return initial_lr * min(1.0, float(epoch + 1) / 5)
    elif epoch < 10:
        return initial_lr / 2
    elif epoch < 15:
        return initial_lr / 2


def cosine_decay(epoch, num_epochs, initial_lr):
    return 0.5 * initial_lr * (1 + cos((epoch * pi) / num_epochs))


def warmup(epoch, lr):
    return lr * (epoch + 1) / 5 if epoch < 5 else lr


class Model:
    def __init__(self, device=None):
        self.conv1 = Conv2D(3, 64, 3, lambda x: x, padding=1, device=device)
        self.bn1 = BatchNorm2D(64, device=device)
        self.relu1 = Relu()
        self.pool1 = MaxPool2D(2, 2)

        self.conv2 = Conv2D(64, 128, 3, lambda x: x, padding=1, device=device)
        self.bn2 = BatchNorm2D(128, device=device)
        self.relu2 = Relu()
        self.pool2 = MaxPool2D(2, 2)

        self.flatten = Flatten()
        self.linear1 = Linear(128 * 8 * 8, 256, Relu(), device=device)
        self.linear2 = Linear(256, 10, lambda x: x, device=device)
        self.layers = [
            self.conv1,
            self.bn1,
            self.relu1,
            self.pool1,
            self.conv2,
            self.bn2,
            self.relu2,
            self.pool2,
            self.flatten,
            self.linear1,
            self.linear2,
        ]

    def forward(self, x, train=True):
        self.bn1.training = train
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        self.bn2.training = train
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        return self.linear2(self.linear1(x, train), train)

    def __call__(self, x, train=True):
        return self.forward(x, train)

    def get_weights(self):
        return [layer.w for layer in self.layers if isinstance(layer, (Linear, Conv2D))]



def save_model(model, name):
    with open(f"{name}.dill", "wb") as f:
        dill.dump(model, f)


def import_data():
    data = torchvision.datasets.CIFAR10(root="./", download=True)
    xs = []
    labels = []
    for i in range(1, 6):
        with open(f"cifar-10-batches-py/data_batch_{i}", "rb") as f:
            dict = pickle.load(f, encoding="bytes")
            xs.append(dict[b"data"])
            labels += dict[b"labels"]

    labels = np.array(labels).reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(labels)

    xs = np.array(xs)
    x = xs.reshape(50000, 3, 32, 32) / 255
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.247, 0.243, 0.261]).reshape(1, 3, 1, 1)
    x = (x - mean) / std

    return x, y


def load_data(x, y, device=None):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.166, random_state=42
    )
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_loader, train_loader





def train(
    model,
    epochs,
    lr,
    loss_function,
    optimizer,
    name,
    train_loader,
    test_loader,
    decay_algo=None,
    use_wandb=True,
    save=True,
    device=None,
):
    softmax = Softmax(device=device)
    if use_wandb:
        run = wandb.init(
            project="ConvNet",
            entity="jacob-ashoo",
            name=name,
            config={
                "name": name,
                "learning_rate": lr,
                "epochs": epochs,
                "optimizer": type(optimizer).__name__,
                "beta1": optimizer.beta1,
                "beta2": optimizer.beta2 if isinstance(optimizer, Adam) else 0,
                "l2_reg": loss_function.l2_reg,
                "l": loss_function.l if loss_function.l2_reg else 0,
                "dropout": "dropout" in name.lower(),
                "decay_algo": decay_algo,
            },
        )

    for epoch in range(epochs):
        losses = []
        n_total = 0
        n_correct = 0

        if decay_algo == "cosine":
            lr = cosine_decay(epoch, epochs, lr)
        elif decay_algo == "step":
            lr = step_decay(epoch, lr)
        elif decay_algo == "warmup":
            lr = warmup(epoch, lr)

        for iteration, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            ypred = model(x)
            weights = model.get_weights() if loss_function.l2_reg else None
            loss = loss_function(ypred, y, weights)
            losses.append(loss)

            n_total += y.size(dim=0)
            ypred = softmax(ypred)
            guesses = torch.argmax(ypred, dim=1)
            truths = torch.argmax(y, dim=1)
            n_correct += (guesses == truths).sum().item()

            grad = loss_function.backward(ypred, y)
            dx = grad

            for layer in reversed(model.layers):
                if isinstance(layer, (Conv2D, Linear)):
                    dx, dw, db = layer.backward(dx)
                    if loss_function.l2_reg:
                        dw += loss_function.l * layer.w
                    optimizer.optimize(dw, db, layer)
                elif isinstance(layer, BatchNorm2D):
                    dx = layer.backward(dx)
                    gamma_grad = torch.mean(
                        dx * layer.x_hat, dim=(0, 2, 3), keepdim=True
                    )
                    beta_grad = torch.mean(dx, dim=(0, 2, 3), keepdim=True)
                    layer.gamma -= optimizer.lr * gamma_grad
                    layer.beta -= optimizer.lr * beta_grad

                elif hasattr(layer, "backward"):
                    dx = layer.backward(dx)

        loss = sum(losses) / len(losses)
        accuracy = n_correct / n_total
        print(f"epoch {epoch} loss: {loss} accuracy: {accuracy}")
        if use_wandb:
            run.log({"train_loss": loss, "accuracy": accuracy})

    test_loss_function = CrossEntropy(device=device, l2_reg=False)
    losses = []
    n_total = 0
    n_correct = 0
    for iteration, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        ypred = model(x, train=False)
        loss = test_loss_function(ypred, y)
        losses.append(loss)

        n_total += y.size(dim=0)
        ypred = softmax(ypred)
        guesses = torch.argmax(ypred, dim=1)
        truths = torch.argmax(y, dim=1)
        n_correct += (guesses == truths).sum().item()

    loss = sum(losses) / len(losses)
    accuracy = n_correct / n_total
    if save:
        save_model(model, name)
    if use_wandb:
        run.log({"test_loss": loss, "test_accuracy": accuracy})
        if save:
            run.save(f"{name}.dill")
        run.finish()


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    x, y = import_data()
    train_loader, test_loader = load_data(x, y, device=device)

    with open ("model.dill", "rb") as f:
        model = dill.load(f)

    test_loss_function = CrossEntropy(device=device, l2_reg=False)
    losses = []
    n_total = 0
    n_correct = 0
    softmax = Softmax()
    for iteration, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        ypred = model(x, train=False)
        loss = test_loss_function(ypred, y)
        losses.append(loss)

        n_total += y.size(dim=0)
        ypred = softmax(ypred)
        guesses = torch.argmax(ypred, dim=1)
        truths = torch.argmax(y, dim=1)
        n_correct += (guesses == truths).sum().item()

    loss = sum(losses) / len(losses)
    accuracy = n_correct / n_total
    print(f"Loss: {loss} Accuracy: {accuracy}")



    
