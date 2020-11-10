import math

import matplotlib.pylab as plt
import numpy as np
from prettytable import PrettyTable

from utils.tf_helper import tf


class NeuralNet:
    def __init__(self, settings, verbose=False):
        """Построение ИНС
        Arguments:
            settings (dict): Словарь с описанием ИНС
            verbose (bool): Флаг для вывода на печать информации о ИНС
        """

        self.settings = settings

        self.outs = settings["outs"]
        self.m = settings["batch_size"]

        self.x = tf.compat.v1.placeholder(tf.float64, shape=[self.m, settings["inputs"]])
        self.y = tf.compat.v1.placeholder(tf.float64, shape=[self.m, settings["outs"]])

        self.nn = []
        for i in self.settings["architecture"].keys():
            if (
                self.settings["architecture"][i]["type"] == "fully_conneted"
                or self.settings["architecture"][i]["type"] == "out"
            ):
                self.nn.append(self.settings["architecture"][i]["neurons"])

        self.st = [self.settings["inputs"]] + self.nn

        # Полносвязная часть
        self.sizes = []
        self.shapes = []
        for i in range(1, len(self.nn) + 1):
            self.shapes.append((self.st[i - 1], self.st[i]))
            self.shapes.append((1, self.st[i]))
        self.sizes = [h * w for h, w in self.shapes]
        self.neurons_cnt = sum(self.sizes)

        activations = {"relu": tf.nn.relu, "sigmoid": tf.nn.sigmoid, "tanh": tf.nn.tanh, "softmax": tf.nn.softmax}

        # Граф для прямого вычисления
        # TODO Другие инициалезеры
        self.initializer = tf.compat.v1.initializers.lecun_uniform()
        self.p = tf.Variable(self.initializer([self.neurons_cnt], dtype=tf.float64))
        self.parms = tf.split(self.p, self.sizes, 0)
        for i in range(len(self.parms)):
            self.parms[i] = tf.reshape(self.parms[i], self.shapes[i])
        self.Ws = self.parms[0:][::2]
        self.bs = self.parms[1:][::2]

        # TODO Перепсать под два класса сетей СНС и ПИНС
        self.y_hat = self.x
        for i in range(len(self.nn)):
            self.activation = activations[
                self.settings["architecture"][list(settings["architecture"].keys())[i]]["activation"]
            ]
            self.y_hat = self.activation(tf.matmul(self.y_hat, self.Ws[i]) + self.bs[i])
        self.y_hat_flat = tf.squeeze(self.y_hat)

        self.r = self.y - self.y_hat

        # TODO Переписать под любую ошибку
        self.loss = tf.reduce_mean(tf.square(self.r))

        # Граф для Левенберга-Марквардта
        # TODO Добавить возможность изменения критерия из настроек
        self.error_estimate = 10 * math.log10(1 / (4 * self.m * int(self.outs)))

        self.opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1)
        self.mu = tf.compat.v1.placeholder(tf.float64, shape=[1])
        self.p_store = tf.Variable(tf.zeros([self.neurons_cnt], dtype=tf.float64))
        self.save_parms = tf.compat.v1.assign(self.p_store, self.p)
        self.restore_parms = tf.compat.v1.assign(self.p, self.p_store)

        def jacobian(y, x, m):
            loop_vars = [
                tf.constant(0, tf.int32),
                tf.TensorArray(tf.float64, size=m),
            ]

            _, jacobian = tf.while_loop(
                lambda i, _: i < m, lambda i, res: (i + 1, res.write(i, tf.gradients(y[i], x)[0])), loop_vars
            )

            return jacobian.stack()

        self.I = tf.eye(self.neurons_cnt, dtype=tf.float64)
        self.j = jacobian(self.y_hat_flat, self.p, self.m)
        self.jT = tf.transpose(self.j)
        self.jTj = tf.matmul(self.jT, self.j)
        self.jTr = tf.matmul(self.jT, self.r)
        self.jTj = tf.hessians(self.loss, self.p)[0]
        self.jTr = -tf.gradients(self.loss, self.p)[0]
        self.jTr = tf.reshape(self.jTr, shape=(self.neurons_cnt, 1))

        self.jTj_store = tf.Variable(tf.zeros((self.neurons_cnt, self.neurons_cnt), dtype=tf.float64))
        self.jTr_store = tf.Variable(tf.zeros((self.neurons_cnt, 1), dtype=tf.float64))
        self.save_jTj_jTr = [
            tf.compat.v1.assign(self.jTj_store, self.jTj),
            tf.compat.v1.assign(self.jTr_store, self.jTr),
        ]

        self.dx = tf.matmul(tf.linalg.inv(self.jTj_store + tf.multiply(self.mu, self.I)), self.jTr_store)
        self.dx = tf.squeeze(self.dx)
        self._dx = tf.matmul(tf.linalg.inv(self.jTj + tf.multiply(self.mu, self.I)), self.jTr)
        self._dx = -tf.squeeze(self._dx)

        self.lm = self.opt.apply_gradients([(-self.dx, self.p)])

        self.session = tf.compat.v1.Session()

        self.session.run(tf.compat.v1.global_variables_initializer())

        if verbose:
            print(5 * "=" + ">Neural net info<" + 5 * "=", "\n")
            print("Settings: ")
            t = PrettyTable(["Setting", "Value"])
            for i in settings.keys():
                if i != "architecture":
                    t.add_row([i, settings[i]])
            print(t)
            print("\ntf version: ", tf.__version__, "\n")
            print(f"Complex:\n        [parameters]x[data lenth]\n        {self.neurons_cnt}x{self.m}\n")
            print("Architecture:")
            t = PrettyTable(["Name", "Type", "Neurons number", "Activation"])
            t.add_row(["input", "input", settings["inputs"], "-"])
            for i in settings["architecture"].keys():
                t.add_row(
                    [
                        i,
                        settings["architecture"][i]["type"],
                        settings["architecture"][i]["neurons"],
                        settings["architecture"][i]["activation"],
                    ]
                )
            print(t)
            print("\n")

        return

    def fit_lm(
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        mu_init=3.0,
        min_error=1e-10,
        max_steps=100,
        mu_multiply=10,
        mu_divide=10,
        m_into_epoch=10,
        verbose=False,
    ):

        # Приведение батчей к одной форме [shape]
        self.min_error = min_error
        if len(x_train) <= self.settings["batch_size"] and len(y_valid) <= self.settings["batch_size"]:
            len_of_test = len(x_valid)
            len_of_train = len(x_train)
            x_train, x_valid, y_train, y_valid = batch_expansion(x_train, x_valid, y_train, y_valid)

        else:
            len_of_test = len(x_valid)
            len_of_train = len(x_train)
            x_train, x_valid, y_train, y_valid = self.get_batches(x_train, x_valid, y_train, y_valid)

        #train_dict = {self.x: x_train, self.y: y_train}

        #valid_dict = {self.x: x_valid, self.y: y_valid}

        train_dict = {self.mu : np.array([mu_init])}

        self.error_train = {"mse": [], "mae": []}
        self.error_test = {"mse": [], "mae": []}

        step = 0

        current_loss = self.session.run(self.loss, train_dict)

        while current_loss > min_error and step < max_steps:
            step += 1

            if step % int(max_steps / 5) == 0 and verbose:
                error_string = ""
                for err in self.error_train.keys():
                    error_string += f"{err}: {self.error_train[err][-1]:.2e} "
                for err in self.error_test.keys():
                    error_string += f"{err}: {self.error_test[err][-1]:.2e} "
                print(f"LM step: {step}, mu: {train_dict[self.mu][0]:.2e}, {error_string}")

            # Start batch
            self.session.run(self.save_parms)
            self.session.run(self.save_jTj_jTr, train_dict)
            success = False
            for i in range(m_into_epoch):
                self.session.run(self.lm, train_dict)
                new_loss = self.session.run(self.loss, train_dict)
                if new_loss < current_loss:
                    train_dict[self.mu] /= mu_divide
                    current_loss = new_loss
                    success = True
                    break
                train_dict[self.mu] *= mu_multiply
                self.session.run(self.restore_parms)

            self.error_train["mse"].append(current_loss)
            y_pred_valid = self.session.run(self.y_hat, valid_dict)
            y_pred = self.session.run(self.y_hat, train_dict)
            self.error_train["mae"].append(
                mae(
                    np.argmax(np.asarray(y_pred)[:len_of_train], axis=1),
                    np.argmax(np.asarray(y_train)[:len_of_train], axis=1),
                )
            )
            self.error_test["mse"].append(
                mse(np.asarray(y_pred_valid)[:len_of_test].ravel(), np.asarray(y_valid)[:len_of_test].ravel())
            )
            self.error_test["mae"].append(
                mae(
                    np.argmax(np.asarray(y_pred_valid)[:len_of_test], axis=1),
                    np.argmax(np.asarray(y_valid)[:len_of_test], axis=1),
                )
            )

            if not success:
                error_string = ""
                for err in self.error_train.keys():
                    error_string += f"{err}: {self.error_train[err][-1]:.2e} "
                for err in self.error_test.keys():
                    error_string += f"{err}: {self.error_test[err][-1]:.2e} "

                print(f"LM failed to improve, on step {step:}, {error_string}\n")
                self.session.run(self.p)
                return

        print(f"LevMarq ended on: {step:},\tfinal loss: {current_loss:.2e}\n")
        self.session.run(self.p)

    def predict(self, data_to_predict, raw=True):

        init_len = len(data_to_predict)
        predict_dict = {
            self.x: np.vstack(
                (
                    data_to_predict,
                    np.zeros(shape=(self.settings["batch_size"] - len(data_to_predict), data_to_predict.shape[1])),
                )
            )
        }
        preds = self.session.run(self.y_hat, predict_dict)[:init_len]
        if raw:
            return preds
        return np.argmax(preds, axis=1)

    def plot_lw(self, path, save=False):

        best_result = np.min(self.error_test["mae"])

        plt.rcParams.update({"font.size": 15})
        fig, ax = plt.subplots(2, 1)

        ax[0].plot(
            [10 * np.log10(float(self.min_error))] * int(len(self.error_train["mse"])), "r--", label="Критерий останова"
        )
        ax[0].plot(10 * np.log10(self.error_train["mse"]), "g", label="MSE обучение")
        ax[0].plot(10 * np.log10(self.error_test["mse"]), "b", label="MSE тест")
        ax[0].legend(loc="best")

        ax[1].plot(
            [best_result] * int(len(self.error_train["mse"])),
            "r--",
            label=f"Лучший результат на тесте {round(best_result, 3)}",
        )
        ax[1].plot(self.error_train["mae"], "g", label="MAE обучение")
        ax[1].plot(self.error_test["mae"], "b", label="MAE тест")
        ax[1].legend(loc="best")

        ax[0].set_xlabel("Эпохи обучения")
        ax[0].set_ylabel("Ошибка MSE, дБ")
        ax[0].set_title("График MSE")

        ax[1].set_xlabel("Эпохи обучения")
        ax[1].set_ylabel("Ошибка MAE")
        ax[1].set_title("График MAE")

        plt.subplots_adjust(hspace=0.3)

        if save:
            plt.savefig(path, dpi=300)

        plt.show()

    def get_batches(self, x_train, x_test, y_train, y_test):
        
        x_train_count = math.floor(len(x_train) / self.settings["batch_size"]) + 1
        x_test_count = math.floor(len(x_test) / self.settings["batch_size"]) + 1
        x_train_batches = []
        x_test_batches = []
        y_train_batches = []
        y_test_batches = []

        for i in range(x_train_count - 1):
            x_train_batches.append(
                x_train[i * self.settings["batch_size"] : (i + 1) * self.settings["batch_size"]]
            )
            y_train_batches.append(
                y_train[i * self.settings["batch_size"] : (i + 1) * self.settings["batch_size"]]
            )
        x_train_batches.append(np.vstack([
            x_train[(x_train_count - 1) * self.settings["batch_size"] : ],
            np.zeros(shape=([
                self.settings["batch_size"] - len(x_train[(x_train_count - 1) * self.settings["batch_size"] : 
            ])] + list(x_train.shape[1:])))
        ]))
        y_train_batches.append(np.vstack([
            y_train[(x_train_count - 1) * self.settings["batch_size"] : ],
            np.zeros(shape=([
                self.settings["batch_size"] - len(y_train[(x_train_count - 1) * self.settings["batch_size"] : 
            ])] + list(y_train.shape[1:])))
        ]))

        for i in range(x_test_count - 1):
            x_test_batches.append(
                x_test[i * self.settings["batch_size"] : (i + 1) * self.settings["batch_size"]]
            )
            y_test_batches.append(
                y_test[i * self.settings["batch_size"] : (i + 1) * self.settings["batch_size"]]
            )
        x_test_batches.append(np.vstack([
            x_test[(x_test_count - 1) * self.settings["batch_size"] : ],
            np.zeros(shape=([
                self.settings["batch_size"] - len(x_test[(x_test_count - 1) * self.settings["batch_size"] : 
            ])] + list(x_test.shape[1:])))
        ]))
        y_test_batches.append(np.vstack([
            y_test[(x_test_count - 1) * self.settings["batch_size"] : ],
            np.zeros(shape=([
                self.settings["batch_size"] - len(y_test[(x_test_count - 1) * self.settings["batch_size"] : 
            ])] + list(y_test.shape[1:])))
        ]))

        return (
            np.asarray(x_train_batches), 
            np.asarray(x_test_batches), 
            np.asarray(y_train_batches), 
            np.asarray(y_test_batches)
        )

def mae(vec_pred, vec_true):
    err = 0
    for j in range(vec_true.shape[0]):
        err += np.abs(vec_true[j] - vec_pred[j])

    return err / len(vec_pred)


def mse(vec_pred, vec_true):
    err = 0
    for j in range(vec_true.shape[0]):
        err += np.sqrt(np.power(vec_true[j] - vec_pred[j], 2))

    return err / len(vec_pred)


def batch_expansion(x_train, x_test, y_train, y_test):
    if len(x_test) == len(x_train):
        return (
            x_train.reshape([1] + list(x_train.shape)), 
            x_test.reshape([1] + list(x_test.shape)), 
            y_train.reshape([1] + list(y_train.shape)), 
            y_test.reshape([1] + list(y_test.shape))
        )

    if len(x_test) < len(x_train):
        x_test = np.vstack((x_test, np.zeros(shape=([len(x_train) - len(x_test)] + list(x_test.shape[1:])))))
        y_test = np.vstack((y_test, np.zeros(shape=([len(y_train) - len(y_test)] + list(y_test.shape[1:])))))
        return (
            x_train.reshape([1] + list(x_train.shape)), 
            x_test.reshape([1] + list(x_test.shape)), 
            y_train.reshape([1] + list(y_train.shape)), 
            y_test.reshape([1] + list(y_test.shape))
        )

    if len(y_test) > len(x_train):
        x_train = np.vstack((x_train, np.zeros(shape=([len(x_test) - len(x_train)] + list(x_train.shape[1:])))))
        y_train = np.vstack((y_train, np.zeros(shape=([len(y_test) - len(y_train)] + list(y_train.shape[1:])))))
        return (
            x_train.reshape([1] + list(x_train.shape)), 
            x_test.reshape([1] + list(x_test.shape)), 
            y_train.reshape([1] + list(y_train.shape)), 
            y_test.reshape([1] + list(y_test.shape))
        )
