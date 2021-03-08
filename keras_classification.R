########################################################
# Classifying surface EMG signals
# Using 1D convolutionnal neural networks
# -------------------------------------------------
# Written by Ladislas Nalborczyk
# Last updated on March 8, 2021
##############################################

# tutorials
# https://letyourmoneygrow.com/2018/05/27/classifying-time-series-with-keras-in-r-a-step-by-step-example/
# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

library(tidyverse) # data formatting
library(keras) # neural networks
library(abind) # stacking high-dimensional arrays

# importing data
df <- read.csv("data/filtered_centered_emg_data.csv")

# trigger values provide the condition
# overt speech condition: trigger values = 1 to 20
# inner speech condition: trigger values = 21 to 40
# listening condition: trigger values = 41 to 60

# retrieving mot ID
lookup <- read.csv("data/lookup.csv", sep = ";") # %>% rename(stims = trigger)
df$word <- lookup$mot[match(unlist(df$trigger), lookup$trigger)]
df$class <- ifelse(df$trigger %% 2 == 0, 0, 1)
head(df, 50)

# gets the number of participants
n_ppts <- n_distinct(df$participant) %>% as.numeric

# retrieves EMG signals for participant for OOI and ZYG in the overt speech condition
x <- df %>%
    filter(between(trigger, 1, 20) ) %>%
    # filter(participant == "S_01") %>%
    select(OOI, ZYG, trigger, class, participant) %>%
    mutate(sample = rep(1:1000, length.out = nrow(.) ) ) %>%
    mutate(repetition = rep(rep(1:6, each = 1000), length.out = nrow(.) ) ) %>%
    # removes duplicated OOI values but keeps all columns
    distinct(OOI, .keep_all = TRUE) %>%
    # pivots from long to wide
    pivot_wider(names_from = sample, values_from = c(OOI, ZYG) ) %>%
    select(-trigger, -class, -repetition, -participant) %>%
    as.matrix

# prints the dimensions of x (input surface EMG data)
dim(x)
ooi <- x[, 0:1000]
zyg <- x[, 1001:2000]
# fro <- x[, 2001:3000]
# cor <- x[, 3001:4000]
ooi_reshaped <- array_reshape(x = ooi, dim = c(dim(ooi), 1), order = "C")
zyg_reshaped <- array_reshape(x = zyg, dim = c(dim(zyg), 1), order = "C")
# fro_reshaped <- array_reshape(x = fro, dim = c(dim(fro), 1), order = "C")
# cor_reshaped <- array_reshape(x = cor, dim = c(dim(cor), 1), order = "C")
x_reshaped <- abind(ooi_reshaped, zyg_reshaped, along = 3)
dim(x_reshaped)

# train/test split
# number of lines in training set (6 repetitions * 20 words * 18 participants)
b <- 6 * 20 * 18
# x_train <- array_reshape(x = x[1:b, ], dim = c(dim(x[1:b, ]), 1), order = "C")
# x_test <- array_reshape(x = x[(b+1):nrow(x), ], dim = c(dim(x[(b+1):nrow(x), ]), 1), order = "C")
x_train <- x_reshaped[1:b, , ]
x_test <- x_reshaped[(b+1):nrow(x_reshaped), , ]
print(c("x_train dimension is: ", dim(x_train), "\n", "x_test dimension is: ", dim(x_test) ) )

# retrieves labels (word class) in the overt speech condition
y <- df %>%
    filter(between(trigger, 1, 20) ) %>%
    # filter(participant == "S_01") %>%
    select(OOI, trigger, class, participant) %>%
    mutate(sample = rep(1:1000, length.out = nrow(.) ) ) %>%
    mutate(repetition = rep(rep(1:6, each = 1000), length.out = nrow(.) ) ) %>%
    # removes duplicated OOI values but keeps all columns
    distinct(OOI, .keep_all = TRUE) %>%
    # pivots from long to wide
    pivot_wider(names_from = sample, values_from = OOI) %>%
    pull(class)

# dummy encoding
num_classes <- n_distinct(y) %>% as.numeric
y_categ <- to_categorical(y = y, num_classes = num_classes)

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
        filters = 40, kernel_size = 10, strides = 2,
        padding = "same", activation = "relu",
        input_shape = c(dim(x_reshaped)[2], dim(x_reshaped)[3])
        ) %>%
    layer_dropout(rate = 0.2) %>%
    layer_max_pooling_1d(pool_size = 3) %>%
    # layer_batch_normalization() %>%
    layer_conv_1d(
        filters = 32, kernel_size = 5, strides = 2,
        padding = "same", activation = "relu"
        ) %>%
    layer_dropout(rate = 0.2) %>%
    layer_max_pooling_1d(pool_size = 3) %>%
    # layer_batch_normalization() %>%
    # layer_conv_1d(
    #     filters = 40, kernel_size = 4, strides = 1,
    #     padding = "same", activation = "relu"
    #     ) %>%
    # layer_dropout(rate = 0.2) %>% 
    # layer_max_pooling_1d(pool_size = 3) %>%
    # layer_batch_normalization() %>%
    layer_global_max_pooling_1d() %>%
    # layer_flatten() %>%
    # bidirectional(layer_lstm(units = 128) ) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(rate = 0.3) %>% 
    layer_dense(units = num_classes, activation = "softmax")

summary(model)

model %>%
    compile(
        loss = "categorical_crossentropy",
        optimizer = "adam",
        metrics = c("accuracy") #, "binary_accuracy", "categorical_accuracy")
        )

history <- model %>%
    fit(
        x_train, y_train,
        epochs = 20,
        batch_size = 10,
        validation_split = 0.2,
        callbacks = list(
            # callback_early_stopping(monitor = "val_loss", patience = 10, verbose = 1)
            )
        )

# validation accuracy is around 75% just using the OOI...
# validation accuracy is around 80% when using both the OOI and the ZYG...
# validation accuracy is around 80% when using all four facial muscles (OOI, ZYG, COR, FRO)...
# NB: it was 0.848 [0.816, 0.877] using random forest and all four facial muscles...
# plots evolution of accuracy and loss (categorical cross-entropy)
plot(history)

# evaluating the model's predictions
model %>% evaluate(x_test, y_test)

# makes predictions
predictions <- model %>% predict_classes(x_test)

# confusion matrix
table(predictions, y[(b+1):nrow(y_categ)])

# saves the whole model
save_model_hdf5(model, "models/emg_1d_cnn_model_overt.h5")
loaded_model <- load_model_hdf5("models/emg_1d_cnn_model_overt.h5")

# saves JSON config to disk
json_config <- model_to_json(model)
writeLines(json_config, "models/emg_1d_cnn_model_config_overt.json")
