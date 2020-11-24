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

        self.x = tf.compat.v1.placeholder(tf.float64, shape=[settings["batch_size"]] + settings["inputs"], name="input_data")
        self.y = tf.compat.v1.placeholder(tf.float64, shape=[settings["batch_size"]] + [settings["outs"]], name="input_labels")


        #====================================================#
        
        self.conv_shape_checker()
        self.neurons_cnt = 0
        
        keys = list(self.settings["architecture"].keys())
        last_conv_dim = self.settings["inputs"][-1]
        self.weights_shapes = []
        for layer in range(len(keys)):
            if self.settings["architecture"][keys[layer]]["type"] == "convolution":
                self.neurons_cnt += np.prod(self.settings["architecture"][keys[layer]]["kernel"]) * last_conv_dim * self.settings["architecture"][keys[layer]]["filtres"] + self.settings["architecture"][keys[layer]]["filtres"]
                self.weights_shapes.append(self.settings["architecture"][keys[layer]]["kernel"] + [last_conv_dim] + [self.settings["architecture"][keys[layer]]["filtres"]])
                self.weights_shapes.append([1, 1, 1] + [self.settings["architecture"][keys[layer]]["filtres"]])
                last_conv_dim = self.settings["architecture"][keys[layer]]["filtres"]                    
                
            if self.settings["architecture"][keys[layer]]["type"] == "fully_conneted" or self.settings["architecture"][keys[layer]]["type"] == "out":
                if self.settings["architecture"][keys[layer - 1]]["type"] == "flatten":
                    self.weights_shapes.append([self.settings["architecture"][keys[layer - 1]]["out_shape"], self.settings["architecture"][keys[layer]]["neurons"]])
                    self.weights_shapes.append([1] + [self.settings["architecture"][keys[layer]]["neurons"]])
                    self.neurons_cnt += self.settings["architecture"][keys[layer - 1]]["out_shape"] * self.settings["architecture"][keys[layer]]["neurons"] + self.settings["architecture"][keys[layer]]["neurons"]
                else:
                    if layer == 0:
                        self.weights_shapes.append([self.settings["inputs"][0], self.settings["architecture"][keys[layer]]["neurons"]])
                        self.weights_shapes.append([1] + [self.settings["architecture"][keys[layer]]["neurons"]])
                        self.neurons_cnt += self.settings["inputs"][0] * self.settings["architecture"][keys[layer]]["neurons"] + self.settings["architecture"][keys[layer]]["neurons"]
                    else:
                        self.weights_shapes.append([self.settings["architecture"][keys[layer - 1]]["neurons"], self.settings["architecture"][keys[layer]]["neurons"]])
                        self.weights_shapes.append([1] + [self.settings["architecture"][keys[layer]]["neurons"]])
                        self.neurons_cnt += self.settings["architecture"][keys[layer - 1]]["neurons"] * self.settings["architecture"][keys[layer]]["neurons"] + self.settings["architecture"][keys[layer]]["neurons"]
        
        self.initial = tf.keras.initializers.glorot_normal()
        self.p = tf.Variable(self.initial([self.neurons_cnt], dtype=tf.float64))
        self.params = tf.split(self.p, [np.prod(x) for x in self.weights_shapes], 0)

        # Build computation graph for neural network
        activations = {"relu": tf.nn.relu, "sigmoid": tf.nn.sigmoid, "tanh": tf.nn.tanh, "softmax": tf.nn.softmax}

        self.y_hat = self.x
        arch_index = 0

        for layer in range(len(keys)):
            if self.settings["architecture"][keys[layer]]["type"] == "convolution":
                self.activation = activations[
                    self.settings["architecture"][keys[layer]]["activation"]
                ]
                pad_h = [self.settings["architecture"][keys[layer]]["pad"][0] / 2 for x in range(2)]
                pad_w = [self.settings["architecture"][keys[layer]]["pad"][1] / 2 for x in range(2)]
                self.y_hat = self.activation(
                    tf.nn.conv2d(
                        self.y_hat, 
                        tf.reshape(self.params[arch_index], self.weights_shapes[arch_index]), 
                        [1] + self.settings["architecture"][keys[layer]]["stride"] + [1], 
                        padding=[[0, 0], pad_h, pad_w, [0, 0]], 
                        name=keys[layer]
                    ) + tf.reshape(self.params[arch_index + 1], self.weights_shapes[arch_index + 1]))
                arch_index += 2
            if self.settings["architecture"][keys[layer]]["type"] == "max_pool":
                self.y_hat = tf.nn.max_pool(
                    self.y_hat,
                    [1] + self.settings["architecture"][keys[layer]]["kernel"] + [1],
                    [1] + self.settings["architecture"][keys[layer]]["stride"] + [1],
                    padding="SAME", 
                    name=keys[layer]
                )
            if self.settings["architecture"][keys[layer]]["type"] == "flatten":
                self.y_hat = tf.reshape(
                    self.y_hat, 
                    [self.settings["batch_size"]] + [self.settings["architecture"][keys[layer]]["out_shape"]],
                    name=keys[layer]
                )
            if self.settings["architecture"][keys[layer]]["type"] == "fully_conneted" or self.settings["architecture"][keys[layer]]["type"] == "out":
                self.activation = activations[
                    self.settings["architecture"][keys[layer]]["activation"]
                ]
                self.y_hat = self.activation(
                    tf.matmul(
                        self.y_hat, 
                        tf.reshape(self.params[arch_index], self.weights_shapes[arch_index])
                    ) + tf.reshape(self.params[arch_index + 1], self.weights_shapes[arch_index + 1]), 
                    name=keys[layer]
                )
                arch_index += 2
            
        self.y_hat_flat = tf.squeeze(self.y_hat)
        self.r = self.y - self.y_hat
        self.loss = tf.reduce_mean(tf.square(self.r), name="Loss")

        # Build computation graph for Levenberg-Marqvardt algorithm
        self.opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1)
        self.mu = tf.compat.v1.placeholder(tf.float64, shape=[1], name="mu")
        self.p_store = tf.Variable(tf.zeros([self.neurons_cnt], dtype=tf.float64))
        self.save_parms = tf.compat.v1.assign(self.p_store, self.p)
        self.restore_parms = tf.compat.v1.assign(self.p, self.p_store)

        #=============================================
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
            print(f"Complex:\n        [parameters]x[batch size]\n        {self.neurons_cnt}x{self.m}\n")
            print("Architecture:")
            t = PrettyTable(["Name", "Type", "Neurons number", "Activation"])
            t.add_row(["input", "input", settings["inputs"], "-"])
            for i in settings["architecture"].keys():
                row = [i, settings["architecture"][i]["type"]]
                if settings["architecture"][i]["type"] == "convolution":
                    row.append(str(settings["architecture"][i]["filtres"]) + " x " + str(settings["architecture"][i]["kernel"]))
                    row.append(settings["architecture"][i]["activation"])
                if settings["architecture"][i]["type"] == "flatten":
                    row.append("-")
                    row.append("-")
                if settings["architecture"][i]["type"] == "max_pool":
                    row.append("-")
                    row.append("-")
                if settings["architecture"][i]["type"] == "fully_conneted" or settings["architecture"][i]["type"] == "out":
                    row.append(settings["architecture"][i]["neurons"])
                    row.append(settings["architecture"][i]["activation"])
                t.add_row(row)
            print(t)
            print("\n")

        return

    def fit_lm(
        self,
        x_train,
        y_train,
        x_valid=None,
        y_valid=None,
        train_test_split=0.3,
        mu_init=3.0,
        min_error=1e-10,
        max_steps=100,
        mu_multiply=10,
        mu_divide=10,
        m_into_epoch=10,
        verbose=False,
        random_batches=False
    ):

        # Batches to one shape
        self.min_error = min_error
        batch_operate_flag = False
        if len(x_train) <= self.settings["batch_size"] and len(y_valid) <= self.settings["batch_size"]:
            len_of_test = len(x_valid)
            len_of_train = len(x_train)
            x_train, x_valid, y_train, y_valid = self.batch_expansion(x_train, x_valid, y_train, y_valid)

        else:
            len_of_test = len(x_valid)
            len_of_train = len(x_train)
            x_train, x_valid, y_train, y_valid = self.get_batches(x_train[0], x_valid[0], y_train[0], y_valid[0])
            batch_operate_flag = True
        
        mu_track = {}
        for i in range(len(x_train)):
            mu_track[i] = mu_init

        self.error_train = {"mse": [], "mae": []}
        self.error_test = {"mse": [], "mae": []}

        mse_train, mae_train, mse_test, mae_test = self.get_errors(x_train, y_train, x_valid, y_valid, mu_init)
        self.error_train["mse"].append(mse_train)
        self.error_train["mae"].append(mae_train)
        self.error_test["mse"].append(mse_test)
        self.error_test["mae"].append(mae_test)

        step = 0

        current_loss = self.current_learn_loss(x_train, y_train, np.array([mu_init]))
        init_loss = current_loss        

        while current_loss / init_loss > min_error and step < max_steps:
            step += 1

            for batch in range(len(mu_track)):
                if mu_track[batch] > 1e100 or mu_track[batch] < 1e-100:
                    mu_track[batch] = mu_init

            if step % int(max_steps / 5) == 0 and verbose:
                error_string = ""
                for err in self.error_train.keys():
                    error_string += f"train {err}: {self.error_train[err][-1]:.2e} "
                for err in self.error_test.keys():
                    error_string += f"test {err}: {self.error_test[err][-1]:.2e} "
                print(f"LM step: {step}, {error_string}")

            if random_batches == True and batch_operate_flag == True:
                x_train, x_valid, y_train, y_valid = self.get_batches(x_train, x_valid, y_train, y_valid)

            # Start batch
            for batch in range(len(x_train)):
                current_loss_batch = self.session.run(
                    self.loss, {self.x : x_train[batch], self.y : y_train[batch], self.mu : np.asarray([mu_init])}
                )
                train_dict = {self.mu : np.asarray([mu_track[batch]])}
                train_dict[self.x] = x_train[batch]
                train_dict[self.y] = y_train[batch]
                self.session.run(self.save_parms)
                self.session.run(self.save_jTj_jTr, train_dict)
                sub_epoch = 0
                while sub_epoch < m_into_epoch:
                    self.session.run(self.lm, train_dict)
                    new_loss = self.session.run(self.loss, train_dict)
                    sub_epoch += 1
                    if new_loss < current_loss_batch:
                        mu_track[batch] = mu_track[batch] / mu_divide
                        train_dict[self.mu] = np.asarray([mu_track[batch]])
                        sub_epoch = m_into_epoch + 1
                    else:
                        mu_track[batch] = mu_track[batch] * mu_multiply
                        train_dict[self.mu] = np.asarray([mu_track[batch]])
                        self.session.run(self.restore_parms)
                
                    # End batch
            
            mse_train, mae_train, mse_test, mae_test = self.get_errors(x_train, y_train, x_valid, y_valid, mu_init)
            self.error_train["mse"].append(mse_train)
            self.error_train["mae"].append(mae_train)
            self.error_test["mse"].append(mse_test)
            self.error_test["mae"].append(mae_test)
            current_loss = self.current_learn_loss(x_train, y_train, np.asarray([mu_init]))

        print(f"LevMarq ended on: {step:},\tfinal loss: {self.error_train['mse'][-1]:.2e}\n")
        self.session.run(self.p)

    def get_errors(self, x_train, y_train, x_valid, y_valid, mu):
        (mse_train, mae_train) = (0, 0)
               
        for batch in range(len(x_train)):
            train_dict = {self.x : x_train[batch], self.y : y_train[batch], self.mu : np.asarray([mu])}
            y_pred = self.session.run(self.y_hat, train_dict)
            mse_train += mae(
                np.asarray(y_pred).ravel(),
                np.asarray(y_train)[batch].ravel(),
            )
            mae_train += mae(
                    np.argmax(np.asarray(y_pred), axis=1),
                    np.argmax(np.asarray(y_train)[batch], axis=1),
            )

        (mse_test, mae_test) = (0, 0)
        for batch in range(len(x_valid)):
            valid_dict = {self.x : x_valid[batch], self.y : y_valid[batch], self.mu : np.asarray([mu])}
            y_pred_valid = self.session.run(self.y_hat, valid_dict)                         
            mse_test += mse(
                np.asarray(y_pred_valid).ravel(), 
                np.asarray(y_valid)[batch].ravel()
            )
            mae_test += mae(
                    np.argmax(np.asarray(y_pred_valid), axis=1),
                    np.argmax(np.asarray(y_valid)[batch], axis=1),
            )
        
        return (
            mse_train / len(x_train),
            mae_train / len(x_train),
            mse_test / len(x_valid),
            mae_test / len(x_valid),
        )
    
    def current_learn_loss(self, x_train, y_train, mu):
        loss = 0
        for batch in range(len(x_train)):
            train_dict = {self.x : x_train[batch], self.y : y_train[batch], self.mu : mu}
            loss += self.session.run(self.loss, train_dict)
        return loss

    def predict(self, data_to_predict, raw=True):

        init_len = len(data_to_predict)

        x_pred_count = math.floor(len(data_to_predict) / self.settings["batch_size"]) + 1
        x_pred_batches = []

        for i in range(x_pred_count - 1):
            x_pred_batches.append(
                data_to_predict[i * self.settings["batch_size"] : (i + 1) * self.settings["batch_size"]]
            )
        
        x_pred_batches.append(np.vstack([
            data_to_predict[(x_pred_count - 1) * self.settings["batch_size"] : ],
            np.zeros(shape=([
                self.settings["batch_size"] - len(data_to_predict[(x_pred_count - 1) * self.settings["batch_size"] : 
            ])] + list(data_to_predict.shape[1:])))
        ]))

        predict_dict = {self.x: x_pred_batches[0]}
        preds = self.session.run(self.y_hat, predict_dict)

        for batch in range(1, len(x_pred_batches)):
            predict_dict = {self.x: x_pred_batches[batch]}
            preds = np.vstack([preds, self.session.run(self.y_hat, predict_dict)])

        if raw:
            return preds[:init_len]

        return np.argmax(preds[:init_len], axis=1)

    def plot_lw(self, path, save=False, logscale=True):

        best_result = np.min(self.error_test["mae"])

        plt.rcParams.update({"font.size": 15})
        fig, ax = plt.subplots(2, 1)

        if logscale == True:
            ax[0].plot(
                [10 * np.log10(float(self.min_error))] * int(len(self.error_train["mse"])), "r--", label="Критерий останова"
            )
            ax[0].plot(10 * np.log10(self.error_train["mse"] / self.error_train["mse"][0]), "g", label="MSE обучение")
            ax[0].plot(10 * np.log10(self.error_test["mse"] / self.error_test["mse"][0]), "b", label="MSE тест")
            ax[0].set_ylabel("Ошибка MSE, дБ")
        else:
            ax[0].plot(
                [float(self.min_error)] * int(len(self.error_train["mse"])), "r--", label="Критерий останова"
            )
            ax[0].plot(self.error_train["mse"], "g", label="MSE обучение")
            ax[0].plot(self.error_test["mse"], "b", label="MSE тест")
            ax[0].set_ylabel("Ошибка MSE")

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
        ax[0].set_title("График MSE")

        ax[1].set_xlabel("Эпохи обучения")
        ax[1].set_ylabel("Ошибка MAE")
        ax[1].set_title("График MAE")

        plt.subplots_adjust(hspace=0.3)

        if save:
            plt.savefig(path, dpi=300)

        plt.show()

    def get_batches(self, x_train, x_test, y_train, y_test):
        
        if len(x_train) % self.settings["batch_size"] == 0:
            x_train_count = math.floor(len(x_train) / self.settings["batch_size"])
            x_test_count = math.floor(len(x_test) / self.settings["batch_size"])
        else:
            x_train_count = math.floor(len(x_train) / self.settings["batch_size"]) + 1
            x_test_count = math.floor(len(x_test) / self.settings["batch_size"]) + 1
        
        x_train_batches = []
        x_test_batches = []
        y_train_batches = []
        y_test_batches = []

        x_train, y_train = self.shuffle_input_data(x_train, y_train) 
        x_test, y_test = self.shuffle_input_data(x_test, y_test)

        for i in range(x_train_count - 1):
            x_train_batches.append(
                x_train[i * self.settings["batch_size"] : (i + 1) * self.settings["batch_size"]]
            )
            y_train_batches.append(
                y_train[i * self.settings["batch_size"] : (i + 1) * self.settings["batch_size"]]
            )
   
        x_train_batches.append(np.vstack([
            x_train[(x_train_count - 1) * self.settings["batch_size"] : ],
            x_train[
                len(x_train[(x_train_count - 1) * self.settings["batch_size"] : ]) : self.settings["batch_size"]
            ]
        ]))
        y_train_batches.append(np.vstack([
            y_train[(x_train_count - 1) * self.settings["batch_size"] : ],
            y_train[
                len(y_train[(x_train_count - 1) * self.settings["batch_size"] : ]) : self.settings["batch_size"]
            ]
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
            x_test[
                len(x_test[(x_test_count - 1) * self.settings["batch_size"] : ]) : self.settings["batch_size"]
            ]
        ]))
        y_test_batches.append(np.vstack([
            y_test[(x_test_count - 1) * self.settings["batch_size"] : ],
            y_test[
                len(y_test[(x_test_count - 1) * self.settings["batch_size"] : ]) : self.settings["batch_size"]
            ]
        ]))

        return (
            np.asarray(x_train_batches), 
            np.asarray(x_test_batches), 
            np.asarray(y_train_batches), 
            np.asarray(y_test_batches)
        )

    def shuffle_input_data(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def conv_shape_checker(self):
        keys = list(self.settings["architecture"].keys())
        last_filters_count = 0
        for layer in range(len(keys)):
            if self.settings["architecture"][keys[layer]]["type"] == "convolution":
                if layer == 0:
                    width = math.ceil((
                        self.settings["inputs"][0] - 
                        self.settings["architecture"][keys[layer]]["kernel"][0] + 
                        2 * self.settings["architecture"][keys[layer]]["pad"][0]
                    ) / self.settings["architecture"][keys[layer]]["stride"][0] + 1)

                    height = math.ceil((
                        self.settings["inputs"][1] - 
                        self.settings["architecture"][keys[layer]]["kernel"][1] + 
                        2 * self.settings["architecture"][keys[layer]]["pad"][1]
                    ) / self.settings["architecture"][keys[layer]]["stride"][1] + 1)

                    self.settings["architecture"][keys[layer]]["out_shape"] = [width, height]
                    last_filters_count = self.settings["architecture"][keys[layer]]["filtres"]
                else:
                    width = math.ceil((
                        self.settings["architecture"][keys[layer - 1]]["out_shape"][0] - 
                        self.settings["architecture"][keys[layer]]["kernel"][0] + 
                        2 * self.settings["architecture"][keys[layer]]["pad"][0]
                    ) / self.settings["architecture"][keys[layer]]["stride"][0] + 1)

                    height = math.ceil((
                        self.settings["architecture"][keys[layer - 1]]["out_shape"][1] - 
                        self.settings["architecture"][keys[layer]]["kernel"][1] + 
                        2 * self.settings["architecture"][keys[layer]]["pad"][1]
                    ) / self.settings["architecture"][keys[layer]]["stride"][1] + 1)

                    self.settings["architecture"][keys[layer]]["out_shape"] = [width, height]
                    last_filters_count = self.settings["architecture"][keys[layer]]["filtres"]
            
            if self.settings["architecture"][keys[layer]]["type"] == "max_pool":
                if layer == 0:
                    width = math.ceil((
                        self.settings["inputs"][0] - 
                        self.settings["architecture"][keys[layer]]["kernel"][0]
                    ) / self.settings["architecture"][keys[layer]]["stride"][0] + 1)

                    height = math.ceil((
                        self.settings["inputs"][1] - 
                        self.settings["architecture"][keys[layer]]["kernel"][1]
                    ) / self.settings["architecture"][keys[layer]]["stride"][1] + 1)

                    self.settings["architecture"][keys[layer]]["out_shape"] = [width, height]
                else:
                    width = math.ceil((
                        self.settings["architecture"][keys[layer - 1]]["out_shape"][0] - 
                        self.settings["architecture"][keys[layer]]["kernel"][0]
                    ) / self.settings["architecture"][keys[layer]]["stride"][0] + 1)

                    height = math.ceil((
                        self.settings["architecture"][keys[layer - 1]]["out_shape"][1] - 
                        self.settings["architecture"][keys[layer]]["kernel"][1]
                    ) / self.settings["architecture"][keys[layer]]["stride"][1] + 1)

                    self.settings["architecture"][keys[layer]]["out_shape"] = [width, height]
            
            if self.settings["architecture"][keys[layer]]["type"] == "flatten":
                self.settings["architecture"][keys[layer]]["out_shape"] = np.prod(self.settings["architecture"][keys[layer - 1]]["out_shape"]) * last_filters_count
            
    def batch_expansion(self, x_train, x_test, y_train, y_test):
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


