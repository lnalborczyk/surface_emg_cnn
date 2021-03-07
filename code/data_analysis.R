library(tidyverse)
library(sjPlot)
library(brms)

rm(list=ls())
data <- read.csv("/Users/Ladislas/Desktop/EMG/EMG2017.csv", header = TRUE, sep = ",") # load dataset

options(mc.cores = parallel::detectCores() ) # run MCMCs on multiple cores

data <- data[order(data$trigger),]
data <- data[order(data$participant),]
data$words <- rep(1:20, 3 * length(unique(data$participant) ) )

data$item <- ifelse(data$item=="rounded",1,0) # dummy coding of item

mod1 <- lme4::lmer(OOI ~ condition + item + (1|participant) + (1|trigger), data = data)
mod2 <- lme4::lmer(OOI ~ (1|participant) + (1|trigger), data = data)
mod3 <- lme4::lmer(OOI ~ condition:item + (1|participant) + (1|trigger), data = data)
mod3 <- lme4::lmer(OOI ~ condition + item + condition:item + (1|participant), data = data)

mod3 <- lme4::lmer(OOI ~ item + (1|participant), data = data) # random intercept
mod4 <- lme4::lmer(OOI ~ item + (1+item|participant), data = data) # random slope

plot_fixef <- function(model){
    ci <- confint(model, parm = names(fixef(model)), quiet = TRUE)
    dotchart(fixef(model), xlim = c(min(ci), max(ci) ) )
    abline(v = 0, lty = 3)
    for (i in 1:length(fixef(model))) lines(c(ci [i, 1], ci [i, 2]), rep (i, 2), lwd = 2)
}

plot_fixef(mod1)

########################################
# bayesian brms model
############################
library(brms)
mod1 <- brm(formula = OOI ~ 0 + condition * item + (1|participant) + (1|trigger),
    data = data, family = gaussian(),
    #prior = prior,
    warmup = 1000, thin = 10, iter = 10000,
    chains = 2, cores = parallel::detectCores(),
    control = list(adapt_delta = 0.95) )

library(rethinking)
glimmer(OOI ~ 0 + condition * item + (1|participant) + (1|trigger), data = data)
