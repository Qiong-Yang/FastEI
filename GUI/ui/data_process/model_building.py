# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 17:51:41 2022

@author: yang
"""

"""This module contains functions that will help users to train a word2vec model
through gensim.
"""
from typing import List
from typing import Tuple
from typing import Union
import gensim
from gensim.models.callbacks import CallbackAny2Vec
import time
import copy
import matplotlib.pyplot as plt
class TrainingProgressLogger(CallbackAny2Vec):
    """Callback to log training progress."""

    def __init__(self, num_of_epochs: int):
        """

        Parameters
        ----------
        num_of_epochs:
            Total number of training epochs.
        """
        self.epoch = 0
        self.num_of_epochs = num_of_epochs
        self.loss = 0

    def on_epoch_end(self, model):
        """Return progress of model training"""
        loss = model.get_latest_training_loss()
        self.epoch += 1
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        
          # reset loss inside model
        model.running_training_loss = 0.0
        self.loss = loss


class ModelSaver(CallbackAny2Vec):
    """Callback to save model during training (when specified)."""

    def __init__(self, num_of_epochs: int, iterations: List, filename: str):
        """

        Parameters
        ----------
        num_of_epochs:
            Total number of training epochs.
        iterations:
            Number of total iterations or list of iterations at which to save the
            model.
        filename:
            Filename to save model.
        """
        self.epoch = 0
        self.num_of_epochs = num_of_epochs
        self.iterations = iterations
        self.filename = filename

    def on_epoch_end(self, model):
        """Allow saving model during training when specified in iterations."""
        self.epoch += 1

        if self.filename and self.epoch in self.iterations:
            if self.epoch < self.num_of_epochs:
                filename = self.filename.split(".model")[0] + "_iter_{}.model".format(self.epoch)
            else:
                filename = self.filename
            print("Saving model with name:", filename)
            model.save(filename)


def train_new_word2vec_model(documents, iterations, filename, progress_logger, **settings):
    settings = set_word2vec_defaults(**settings)

    num_of_epochs = max(iterations) if isinstance(iterations, list) else iterations

    # Convert spec2vec style arguments to gensim style arguments
    settings = learning_rates_to_gensim_style(num_of_epochs, **settings)

    # Set callbacks
    callbacks = []
    if progress_logger:
        training_progress_logger = TrainingProgressLogger(num_of_epochs)
        callbacks.append(training_progress_logger)
    if filename:
        if isinstance(iterations, int):
            iterations = [iterations]
        model_saver = ModelSaver(num_of_epochs, iterations, filename)
        callbacks.append(model_saver)

    # Train word2vec model
    model = gensim.models.Word2Vec(documents, callbacks=callbacks, **settings)

    return model


def set_word2vec_defaults(**settings):
    """
    Adapted from: https://github.com/iomega/spec2vec
    Set spec2vec default argument values"(where no user input is give)".
    """
    defaults = {
        "sg": 0,
        "negative": 5,
        "vector_size": 500,
        "window": 750,
        "min_count": 1,
        "learning_rate_initial": 0.025,
        "learning_rate_decay": 0.00025,
        "workers": 10,
        "compute_loss": True,
    }
    assert "min_alpha" not in settings, "Expect 'learning_rate_decay' to describe learning rate decrease."
    assert "alpha" not in settings, "Expect 'learning_rate_initial' instead of 'alpha'."

    # Set default parameters or replace by **settings input
    for key in defaults:
        if key in settings:
            print("The value of {} is set from {} (default) to {}".format(key, defaults[key],
                                                                          settings[key]))
        else:
            settings[key] = defaults[key]
    return settings


def learning_rates_to_gensim_style(num_of_epochs, **settings):
    """Convert "learning_rate_initial" and "learning_rate_decay" to gensim
    "alpha" and "min_alpha"."""
    alpha, min_alpha = set_learning_rate_decay(settings["learning_rate_initial"],
                                               settings["learning_rate_decay"], num_of_epochs)
    settings["alpha"] = alpha
    settings["min_alpha"] = min_alpha
    settings["epochs"] = num_of_epochs

    # Remove non-Gensim arguments from settings
    del settings["learning_rate_initial"]
    del settings["learning_rate_decay"]
    return settings


def set_learning_rate_decay(learning_rate_initial, learning_rate_decay, num_of_epochs):
    min_alpha = learning_rate_initial - num_of_epochs * learning_rate_decay
    if min_alpha < 0:
        print("Warning! Number of total iterations is too high for given learning_rate decay.")
        print("Learning_rate_decay will be set from {} to {}.".format(learning_rate_decay,
                                                                      learning_rate_initial/num_of_epochs))
        min_alpha = 0
    return learning_rate_initial, min_alpha
