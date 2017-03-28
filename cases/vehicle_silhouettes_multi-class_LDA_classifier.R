library( dplyr )

# Load dataset
print( "... loading dataset ..." )
data_raw = read.csv( "../dataset/vehicle.csv", stringsAsFactor = FALSE )

data = data_raw %>% filter( class == "bus" | class == "opel" | class == "van" )

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

# Model Discriminant Analysis
print( "... lda ..." )
library( MASS )
model = lda( class ~ ., data = X_train )

# Ploting LDA Model
projected_data = as.matrix( X_train[, 1:18] ) %*% model$scaling
plot( projected_data, col = X_train[,19], pch = 19 )

# Testing
X_test = test[, !( names( test ) %in% c( "class" ) ) ]  
model.results = predict( model, X_test )

# Results
print( "... resulting ..." )

# Confusion Matrix
library( caret )
t = table( model.results$class, test$class )
print( confusionMatrix( t ) )
