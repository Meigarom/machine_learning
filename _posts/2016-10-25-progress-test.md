---
title: "Prevendo o progresso dos jogadores"
author: Meigarom Diego Fernandes Lopes 
output:
   html_document:
   toc: true
   highlight: zenburn
---

## Objetivo

## Metodologia
Para cumprir o objetivo desse teste, utilizarei um algoritmo de Machine Learning.
A metodologia para resolver o teste esta organizado da seguinte maneira:

1) **Identificação da natureza do problema.**

    Temos um problema regressão que consiste na predição de uma variável ( completed_post ), ou seja, é necessário
encontrar uma função modelo que seja uma boa aproximação da função real que gerou os dados. O algoritmo precisa
ser de aprendizado supervisionado, já que a variável "target" é apresentada.

2) **Definição do algoritmo de aprendizado ( predição )**

    Para modelar esse problema, utilizarei o modelo de machine learning chamado de redes neurais.
    As redes neurais tem a capacidade de modelar tanto problemas de classificação como de regressão, para 
aprendizado supervisionado, através de algumas decisões sobre a arquitetura da rede. A arquitetura será constituida
de uma única camada escondida, cujo o número de neuronios será definido através do treinamento e a função de
ativação da camada de saída será linear e não sigmoide ( utilizada para problemas de classificação ). 
    A decisão de uma única camada escondida vem da literatura de redes neurais que aponta uma única camada escondida
suficiente para gerar todas os pesos que irão compor o modelo.
    Ainda sobre a arquitetura das redes neurais:
    1) Critério de parada do treinamento igual a 0,01 ( limite máximo para a função de erro ) 
    2) Algortimo de backpropagação para o aprendizado.
    3) Erro quadrático médio ao invés de entropia cruzada ( cross entropy ).
    4) Função de ativação linear. 


4) **Treinamento e teste do modelo.**

    O treinamento será realizado utilizando a técnica de k-fold. Devido aos pesos iniciais das redes neurais serem
escolhidos aleatoriamente, o gradiente descendente do algoritmo ( método que usa o sentido gradiente para 
caminhar no espaço de dados dos erros, a fim de encontrar o melhor conjunto de pesos que apresente
o menor erro ) pode encontrar um mínimo local e não global, portanto randomizar uma única vez os dados de 
treinamento, corre-se o risco de estagnar em um mínimo local. Para amenizar esse problema, a técnica do k-fold
divide os dados em k partes e realiza o treinamento com 9 partes e teste com 1 parte, fazendo um sistema de rotação
de modo que todas as partes são treinadas e testadas contras todas as outras. Esse método traz maior confiança sobre
os resultados do erro quadrático médio da rede neural.
    O número de partições sugerido pela literatura é de 10 ( k = 10 ), porém nesse problema escolhi um número de 
partições igual a 3, por motivos de esforço computacional.
    Por fim, utilizarei um conjunto de 5 modelos de redes neurais, cada um com um número diferente de neuronios na
camada escondida ( 2, 3, 5, 7, 10 ). Esse modelos serão avaliados por um conjunto de métricas que definirão o 
melhor número para os neurônios da camada escondida.

5) **Linguagem de implementação**

    Os scripts serão implementados na linguagem R 

## Manipulando o conjunto de dados 
### Bibliotecas
Utilizarei duas bibliotecas básicas do R para manipulação de dados e modelo de redes neurais.

```{r}
suppressMessages( library( dplyr ) )
suppressMessages( library( neuralnet ) )
suppressMessages( library( MASS ) )
```

### Selecionando as entradas
O conjunto de dados de treinamento apresenta 7000 observações ( exemplos ), composto por 14 características e 1 
rótulo ( target, label ).

```{r}
data_raw = Boston
head( data_raw )
```

Descartando as variáveis que julguem não trazerem relevantes informações para o treinamento, nesse problema em 
específico.

Substituindo os valores vazior por zeros.

```{r}
data = data.frame( sapply( data, as.numeric ) )
```

Uma característica desse conjunto de dados é os valores nulos das features em algumas observações. Os valores 
nulos podem representar um valor desconhecido para uma observação. 

As redes neurais não trabalham com valores de entrada vazios, assim alguns estudos dos dados foram realizados
para decidir se descartar as observações com variáveis nulas faria mais sentido do que torná-las zero.
Assim, encontrei alguns padrões de usuários interessantes:

São jogadores que não compraram nenhum "in-app purchase" no jogo para ajudá-lo a vencer alguma fase. São usuário
que jogam "free"

## Normalizando as entradas
    A normalização será realizada utilizando o valor mínimo de cada variável como centro e a diferença dos valores
máximos e mínimos será utiliza para a nova escala. Assim, os valores das colunas ficarão entre 0 e 1.

## Treinamento e Teste
    O treinamento será realizado utilizando o método k-fold, como mencionado anteriormente. O valor das partições
será de 3 ( k = 3 ).  
    Para a definição do número de neuronios na camada escondida. O modelo escolhido será aquele que apresentar 
o menor erro quadrático médio e também o menor desvio padrão. Redes com poucos neuronios na camada escondida são 
preferíveis devido sua simplicidade, menor custo computacional para o treinamento e ainda segue a orientação do 
teorema de Occam'Razor que diz que solução mais simples são favorecidas sobre soluções mais complexas.

```{r eval = FALSE}
set.seed( 200 )
neurons = c(  2, 3, 5, 7, 10 )
k = 3
cv.error = matrix( , length( neurons ), k )

for( j in 1:length( neurons ) ){
    for( i in 1:k ){
        index = sample( 1:nrow( data ), round( 0.9* nrow( data ) ) )
        train.cv = scaled[ index, ]
        test.cv = scaled[ -index, ]

        n = names( train.cv )
        nn = neuralnet( f, data = train.cv, hidden = neurons[ j ], linear.output = TRUE, stepmax = 1e+07 )
        
        pr.nn = compute( nn, test.cv[1:11] )

        cv.error[j,i] = sum( ( test.cv.r - pr.nn )^2 ) / nrow( test.cv )
    }
}

```

## Resultados

    Podemos observar que o modelo com 3 neuronios na camada escondida apresenta o melhor erro e também o 
menor desvio padão em relação aos outros modelos. Portanto, esse modelo será escolhido para performar
sobre os dados de teste.

```{r eval=FALSE}
m = apply( cv.error, 1, mean ) 
s = apply( cv.error, 1, sd ) 
ma = apply( cv.error, 1, max )
mi = apply( cv.error, 1, mean )
results = data.frame( "No of hidden nodes" = neurons, "Mean error" = m, 
                      "Standard Deviation" = s,
                      "Max error" = ma, "Min erro" = mi )
```

```{r eval= FALSE}
    Os resultados das redes com diferentes números de neurônios na camada escondida pode ser visto a seguir:
print( results )
```

```{r }
 No.of.hidden.nodes    Mean.error Standard.Deviation     Max.error  
1                  2 0.06188212397      0.08487676151 0.15978336390 
2                  3 0.02039006315      0.01434344202 0.03587878026
3                  5 0.03876987321      0.03400769329 0.07543931878
4                  7 0.08782071814      0.03337568133 0.11749271899
5                 10 0.13410148245      0.09921123390 0.24843357361
       Min.erro
1 0.06188212397
2 0.02039006315
3 0.03876987321
4 0.08782071814
5 0.13410148245
```

## Aplicando a rede ao conjunto de teste
    Com o modelo definido, podemos aplicá-lo ao conjunto de treinamento.

## Training 
    Vamor usar o mesmo conjunto de treinamento para treinar o modelo, agora com o número de 
neuronios definido.

```{r eval = FALSE}
neuron = 3
n = names( data )
f = as.formula(paste( "completed_post ~",paste(n[ !n %in% "completed_post" ],collapse="+")))
nn = neuralnet( f, data = data, hidden = neuron, linear.output = TRUE, stepmax = 1e+07 )
```

## Arquitetura da rede neural
    A arquitetura da rede neural:

## Testing
    Aplicando o modelo no conjunto de teste

Por fim, anexamos a coluna de predição 

### Conclusão do teste
    O teste tem o objetivo de construir um modelo de predição para o progresso de um jogador
observando algumas variáveis descritivas.
    Um modelo de redes neurais foi implementado e sob certas condições foi treinado e 
utilizado sobre o conjunto de testes.
    O erro do classificador é de 2.0% com desvio padrão de 1.4%. Aparentemente parece um erro
baixo, porém devido as características dos dados e do problema, é um erro muito alto para 
utilizar esse classificador nesse estágio. 

## Sugestão de próximos passos
    1)  Estudo do conjunto de dados é essencial para remover observações e variáveis que não 
apresentam relevantes informações.
    2) Utilizar 10 partições no k-fold
    3) Utilizar todos os dados para treinamento

