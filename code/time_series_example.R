# time series example
# https://letyourmoneygrow.com/2018/05/27/classifying-time-series-with-keras-in-r-a-step-by-step-example/

LOOPBACK = 240 #length of series in each sample
N_FILES = 1000 #number of samples
PROB_CLASS_1 = 0.55
SPLT = 0.8 #80% train, 20% test
X = array(0.0, dim=c(N_FILES, LOOPBACK))  
Y = array(0, dim=N_FILES) #time series class

for(fl in 1:N_FILES)
{
    z = rbinom(1, 1, PROB_CLASS_1)
    if(z==1)
        X[fl, ] = cumprod(1.0 + rnorm(LOOPBACK, 0.0, 0.01))
    else
        X[fl, ] = exp(rnorm(LOOPBACK, 0.0, 0.05))
    
    X[fl, ] = X[fl, ] / max(X[fl,]) #rescale
    Y[fl] = z
}

dim(X)
dim(Y)

b = floor(SPLT*N_FILES)
x_train = X[1:b,]
x_test = X[(b+1):N_FILES,]
y_train = to_categorical(Y[1:b], 2)
y_test = to_categorical(Y[(b+1):N_FILES], 2)

x_train = array_reshape(X[1:b,], c(dim(X[1:b,]), 1))
x_test = array_reshape(X[(b+1):N_FILES,], c(dim(X[(b+1):N_FILES,]), 1))

dim(y_train)
