########################################################
# Classifying surface EMG signals
# Using 1D convolutionnal neural networks
# -------------------------------------------------
# Written by Ladislas Nalborczyk
# Last updated on March 7, 2021
#############################################

# https://letyourmoneygrow.com/2018/05/27/classifying-time-series-with-keras-in-r-a-step-by-step-example/

library(fastDummies)
library(tidyverse)
library(keras)

# importing data
df <- read.csv("data/filtered_centered_emg_data.csv")

# retrieves EMG signals for participant "S_01", OOI in the overt speech condition
x <- df %>%
    filter(trigger < 21) %>%
    filter(participant == "S_01") %>%
    select(OOI, trigger) %>%
    mutate(sample = rep(1:1000, length.out = nrow(.) ) ) %>%
    mutate(repetition = rep(rep(1:6, each = 1000), 20) ) %>%
    # removes duplicated OOI values but keeps all columns
    distinct(OOI, .keep_all = TRUE) %>%
    # pivots from long to wide
    pivot_wider(names_from = c(trigger, repetition), values_from = OOI) %>%
    select(-sample) %>%
    as.matrix %>%
    t

dim(x)

# train/test split
b <- 80 # number of lines in training set
x_train <- array_reshape(x = x[1:b, ], dim = c(dim(x[1:b, ]), 1), order = "C")
x_test <- array_reshape(x = x[(b+1):nrow(x), ], dim = c(dim(x[(b+1):nrow(x), ]), 1), order = "C")
print(c("x_train dimension is: ", dim(x_train), "x_test dimension is: ", dim(x_test) ) )

# retrieves labels (i.e., trigger values)
y <- df %>%
    filter(trigger < 21) %>%
    filter(participant == "S_01") %>%
    # removes duplicated OOI values but keeps all columns
    distinct(OOI, .keep_all = TRUE) %>%
    select(trigger)

# dummy encoding
y <- rep(0:1, length.out = dim(x)[1])
# y <- rep(1:20, length.out = dim(x)[1])
y_categ <- to_categorical(y = y-1, num_classes = n_distinct(y) )

# train/test split
y_train <- y_categ[1:b, ]
y_test <- y_categ[(b+1):nrow(y_categ), ]

#############################################################################
# creates the 1D CNN model
# with one temporal dimension and depth 2 (for OOI and ZYG muscles)
########################################################################

# input_shape should be [samples, time_steps, features]

model <- keras_model_sequential()

model %>%
    layer_conv_1d(
        filters = 16, kernel_size = 5, activation = "relu", input_shape = c(1000, 1)
        ) %>%
    layer_dropout(rate = 0.2) %>% 
    layer_max_pooling_1d(pool_size = 5) %>%
    layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>% 
    layer_max_pooling_1d(pool_size = 5) %>%
    # layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu") %>%
    # layer_dropout(rate = 0.2) %>% 
    # layer_max_pooling_1d(pool_size = 2) %>%
    layer_global_average_pooling_1d() %>%
    # layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 2, activation = "softmax")

summary(model)

model %>%
    compile(
        loss = "categorical_crossentropy",
        optimizer = "adam",
        metrics = c("accuracy")
        )

history <- model %>%
    fit(
        x_train, y_train,
        epochs = 100,
        batch_size = 10,
        validation_split = 0.2,
        callbacks = list(
            callback_early_stopping(monitor = "val_loss", patience = 10, verbose = 1)
            )
        )

# plots evolution of accuracy and loss (categorical cross-entropy)
plot(history)

# evaluating the model's predictions
model %>% evaluate(x_test, y_test)

# makes predictions
predictions <- model %>% predict_classes(x_test)

# saves the whole model
save_model_hdf5(model, "models/emg_1d_cnn_model.h5")
loaded_model <- load_model_hdf5("models/emg_1d_cnn_model.h5")

# saves JSON config to disk
json_config <- model_to_json(model)
writeLines(json_config, "models/emg_1d_cnn_model_config.json")
