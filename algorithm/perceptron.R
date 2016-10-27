#f
f = function( net_matrix ){
    id = which( net_matrix > 0.5 )
    net_matrix[ id ] = 1
    net_matrix[ -id ] = 0
    net_matrix
}

# Activation function
activation_function = function( net ){
    if( net > 0.5 )
        return( 1 )
    return( 0 )
}

# Perceptron function
# input: dataset, learning rate( eta ), error limit ( threshold )
# ouput: weights from trained perceptron 
perceptron = function( dataset, neta = 0.1, threshold = 1e-5 ){
    data = as.matrix( dataset )

    # Initial Random Weights 
    weights = rnorm( mean = 0, sd = 0.1, n = ncol( data ) )
    num_col = ncol( data ) - 1
    num_out = num_col + 1

    sqerror = threshold * 2

    # While square error is less than threshold
    # Epochs
    while( sqerror > threshold ){
        sqerror = 0
        for( i in 1:nrow( data) ){
            # Add bias and calculate forward phase
            net = c( data[ i, 1:num_col ], 1 ) %*% weights

            # Activation function
            y_hat = activation_function( net )

            # Compute square error
            error = ( y_hat - data[ i, num_out ] )
            sqerror = sqerror + error^2
            cat( paste( "Square error = ",  sqerror, "\n" ) )

            # Update weights
            weights = weights - neta * ( error ) * c( data[ i, 1:num_col ], 1 ) 

        }
    }
    return( weights )
}

main = function( plot = FALSE ){ 
    # AND logical dataset 
    dataset = data.frame( x1 = c( 0, 0, 1, 1 ), x2 = c( 0, 1, 0, 1 ), t = c( 0, 0, 0, 1 ) )

    # Call Perceptron Algorithm
    weights = perceptron( dataset, neta = 0.1, threshold = 1e-3 )

    # Plot shattering plane
    if( plot == TRUE ){
        X = seq( 0, 1, length = 100 )
        data = outer( X, X, function( X, Y ){ cbind( X, Y, 1 ) %*% weights } )
        data = f( data )
        filled.contour( data )
    }
} 
