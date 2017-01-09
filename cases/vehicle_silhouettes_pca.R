library( dplyr )

## Load dataset
print( "... load dataset ..." )
data = read.csv( "../dataset/vehicle.csv", stringsAsFactor = FALSE )

## Transform dataset
dataset = data %>% 
            filter( class == "bus" | class == "saab" ) %>%
            transform( class = ifelse( class == "saab", 0, 1 ) )
dataset = as.data.frame( sapply( dataset, as.numeric ) )

## Spliting training and testing dataset
index = sample( 1:nrow( dataset ), nrow( dataset ) * 0.6, replace = FALSE ) 
trainset = dataset[ index, ]
testset = dataset[ -index, ]

## Dimensionality Reduction 
# PCA
print( "... principal component analysis ..." )
pca_trainset = trainset %>% select( -class )
pca_testset = testset %>% select( -class )

pca = prcomp( pca_trainset, scale = T )
pr_var = (pca$sdev)^2 # variance
prop_varex = pr_var / sum( pr_var )
#plot( prop_varex, xlab = "Principal Component", ylab = "Proportion of Variance Explained", type = "b" ) #scree plot
#plot( cumsum( prop_varex ), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", type = "b" )

# Creating a new dataset
train = data.frame( class = trainset$class, pca$x )
test = as.data.frame( predict( pca, newdata = pca_testset ) )

new_trainset = train[, 1:9]
new_testset =  test[, 1:8]

## Build the neural network (NN)
library( neuralnet )
print( "... training ..." )
n = names( new_trainset )
f = as.formula( paste( "class ~", paste( n[!n %in% "class" ], collapse = "+" ) ) )
nn = neuralnet( f, new_trainset, hidden = 4, linear.output = FALSE, threshold=0.01 )

## Plot the NN
#plot( nn, rep = "best" )

## Test the resulting output
print( "... testing ..." )
nn.results = compute( nn, new_testset )

## Results
results = data.frame( actual = testset$class, 
                      prediction = round( nn.results$net.result ) )

mse = sum(( results$actual - results$prediction )^2 ) / nrow( results )
print( paste( "mean square error: ",  mse ) )

## Confusion Matrix
library( caret )
t = table( results ) 
print( confusionMatrix( t ) )
