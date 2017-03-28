library( dplyr )

# Load dataset
print( "... loading dataset ..." )
data_raw = read.csv( "../dataset/vehicle.csv", stringsAsFactor = FALSE )

data = data_raw %>% filter( class == "bus" | class == "opel" | class == "van" )
#data = data_raw

# Scale dataset
print( "... scaling dataset ..." )
maxs = apply( data[,1:18], 2, max )
mins = apply( data[,1:18], 2, min )
dataset = as.data.frame( scale( data[,1:18], center = mins, scale = maxs - mins ) )
dataset = cbind( dataset, "class" = data$class )

# Split dataset
index = sample( 1:nrow( dataset ), round( nrow( dataset )*0.6), replace = FALSE )
X_train = dataset[ index, ]
test = dataset[ -index, ]

# Dimensionality Reduction -> LDA   
print( "... dimensionality reduction - LDA ..." )
library( MASS )
lda = lda( class ~ ., data = X_train )

#new dataset
new_X_train = as.matrix( X_train[,1:18] ) %*% lda$scaling
new_X_train = as.data.frame( new_X_train )
new_X_train$class = X_train$class

# Multi-Class -> Manipulate Labels
print( "... transforming labels ..." )
new_X_train = cbind( new_X_train, opel = new_X_train$class == "opel" )
new_X_train = cbind( new_X_train, van = new_X_train$class == "van" )
#new_X_train = cbind( new_X_train, saab = new_X_train$class == "saab" )
new_X_train = cbind( new_X_train, bus = new_X_train$class == "bus" )
new_X_train = new_X_train[, !( names( new_X_train ) %in% c( "class" ) ) ]

# Model Neural Network
print( "... training neuralnetwork ..." )
library( neuralnet )
n = names( new_X_train )
#f = as.formula( "opel+van+saab+bus ~ LD1+LD2+LD3" )
f = as.formula( "opel+van+bus ~ LD1+LD2" )
nn = neuralnet( f, new_X_train, hidden = 3, linear.output = FALSE, lifesign = "full", 
                threshold = 0.02, stepmax = 1e6 )

# Plotting model 
projected.data = as.matrix( X_train[,1:18] ) %*% lda$scaling
plot( projected.data, col = X_train[, 19], pch = 19 )

# Testing
X_test = as.matrix( test[,1:18] ) %*% lda$scaling
nn.results = compute( nn, X_test )

# Results
print( "... resulting ..." )
idx = apply( nn.results$net.result, c(1), function( x ){ which( x == max( x ) ) } )
#predictions = c( "opel", "van", "saab", "bus")[idx]
predictions = c( "opel", "van", "bus")[idx]

# Confusion Matrix
library( caret )
t = table( predictions, test$class )
print( confusionMatrix( t ) )
