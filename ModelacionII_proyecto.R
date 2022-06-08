################## Bibliotecas necesarias #############
library(ggplot2)
library(dplyr)
library(andrews)
library(vcd)
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(corrplot)
library(psych)
library(coin)
library(nortest)
library(ROCR)
library(car)


################### Cargamos el dataset  ###########
redwine <- read.csv(file.choose(), #winequality-red.csv 
                    header = T, sep = ";")

#Creamos una variable categorica con base al puntaje del vino
#1 si calidad >= 6 (buen vino), 0 e.o.c. (mal vino)
redwine <- redwine %>%
  mutate(Good.quality = ifelse(quality >= 6, 1,0))

redwine$Good.quality <- as.factor(redwine$Good.quality)

#Eliminamos la variable calidad ya que creamos la otra
redwine$quality <- NULL


################## Analisis de los datos###########
View(redwine)

##Resumen
summary(redwine)

##Tambien calculamos el CV y la sd de cada covariable explicativa
coefvar <- function(x){sd(x)/mean(x) * 100}
apply(redwine[1:11], 2, coefvar)
sd_vector <- apply(redwine[1:11], 2, sd) #Lo ocuparemos en el C. de Chauvenet
sd_vector

### Correlacion
corrplot(cor(redwine[1:11]), method = "circle")

### Distribucion
multi.hist(redwine[1:11], bcol = "light blue", dcol = c("dark blue", "red"),
           dlty = c("dashed", "dotted"), lwd = 3)

### Valores atipicos y agrupamiento de datos

#Curva de Andrews
andrews(df = redwine, type = 1, clr = 3, step = 70, ymax = 4.1)

#Boxplots
par(mfrow = c(2,3))
for(i in 1:11){
  boxplot(redwine[i], border = "dark blue", main = names(redwine[i]))
}
par(mfrow = c(1,1))
#Existen multiples valores atipicos en distintas variables
#Ajustaremos el modelo con valores atipicos y posteriormente sin ellos

## Comparamos Medidas estadisticas de vinos de buena calidad y mala calidad
aggregate(. ~ Good.quality, data = redwine, median)
aggregate(. ~ Good.quality, data = redwine, mean)

# Si queremos aplicar la prueba t, aplicamos prueba de Anderson-Darling 
# A las variables explicativas
apply(redwine[1:11], 2 , FUN = ad.test)

#En todas las variables se rechaza normalidad pues p-value < 0.05
#Recurrimos a la prueba no parametrica de Wilcoxon

## Prueba de Mann-Withney- Wilxcoxon

#Ho: La diferencia de medias es igual a cero (m1 = m2)
#H1: La diferencia de medias es distinta de cero (m1 != m2)

wilcox_test(fixed.acidity ~ Good.quality, data = redwine)
wilcox_test(volatile.acidity ~ Good.quality, data = redwine) 
wilcox_test(citric.acid ~ Good.quality, data = redwine)
wilcox_test(residual.sugar ~ Good.quality, data = redwine) #No rechazo H0
wilcox_test(chlorides ~ Good.quality, data = redwine)
wilcox_test(free.sulfur.dioxide ~ Good.quality, data = redwine)
wilcox_test(total.sulfur.dioxide ~ Good.quality, data = redwine)
wilcox_test(density ~ Good.quality, data = redwine)
wilcox_test(pH ~ Good.quality, data = redwine) #No rechazo H0
wilcox_test(sulphates ~ Good.quality, data = redwine)
wilcox_test(alcohol ~ Good.quality, data = redwine)
#Solo el azucar residual y el pH son estadisticamente iguales

############ Particion para el conjunto para prueba-entrenamiento #############
n <- dim(redwine)[1]
set.seed(1989)  ##Taylor's version
entrenamiento <- sample(1:n, 0.66*n)

redwine.prueba <- redwine[-entrenamiento,]
redwine.entrenamiento <- redwine[entrenamiento,]

yentrenamiento <- redwine$Good.quality[entrenamiento]
yprueba <- redwine$Good.quality[-entrenamiento]


############# Ajuste del Primer modelo MRL ##########
logistic_model <- glm(Good.quality ~ ., data = redwine.entrenamiento, family = binomial)
summary(logistic_model)
confint(logistic_model)
vif(logistic_model)
#Existe un problema de multicolinealidad debido a que 5 covariables
#No son estadisticamente significativas, ademas de tener intervalos de 
#Confianza (al 95%) muy amplios y VIF mayores a 5

#Eliminamos Acidez fija, azucar residual, densidad, pH, acido citrico

## Analisis del ajuste
dif_residuos <- logistic_model$null.deviance - logistic_model$deviance
df <- logistic_model$df.null - logistic_model$df.residual
p_value <- pchisq(q = dif_residuos, df = df, lower.tail = FALSE)
#El analisis es significativo, sin embargo el azucar residual no es sig.

## Matriz de confusion
pred1 <- predict.glm(logistic_model, newdata = redwine.prueba, type = "response")
matriz_conf1 <- table(yprueba, floor(pred1+0.5),
                      dnn = c("Observaciones", "Predicciones"))
matriz_conf1
mosaicplot(matriz_conf1, col = c("#7b1c23", "#1c7b45"), main = "Modelo 1")

##Curva ROC
pred = ROCR::prediction(pred1, yprueba)
perf <- performance(pred, "tpr" , "fpr")
plot(perf, col = "#e30b5d", lwd = 3, print.auc = TRUE, ylim = c(0.0,1.0), 
     main = "Modelo 1")
##AUC
AUC1 = performance(pred, measure = "auc")@y.values[[1]]
AUC1

##Precision
accurate1 <- (matriz_conf1[1,1] + matriz_conf1[2,2]) / (sum(matriz_conf1))
accurate1

##Sensibilidad
sensib1 <- (matriz_conf1[2,2]) / (matriz_conf1[2,2] + matriz_conf1[2,1])
sensib1

##Especificidad
especf1 <- matriz_conf1[1,1] / (matriz_conf1[1,1] + matriz_conf1[1,2])
especf1

##Intervalo de confianza para p al 95%
Qalpha <- qnorm(0.025, mean = 0, sd = 1)
inf1 <- accurate1 + Qalpha*(sqrt((accurate1*(1 - accurate1))/dim(redwine.prueba)[1]))
sup1 <- accurate1 - Qalpha*(sqrt((accurate1*(1 - accurate1))/dim(redwine.prueba)[1]))

cat("Intervalo para P al 95% de Confianza",
    "\n", c(inf1, sup1), "\n", "Longitud: ", sup1 - inf1)


########## Ajuste del segundo MRL ###########
log_model2 <- glm(Good.quality ~ volatile.acidity + chlorides + 
                    free.sulfur.dioxide + total.sulfur.dioxide + sulphates + 
                    alcohol, data = redwine.entrenamiento, family = binomial)

summary(log_model2)
confint(log_model2)
vif(log_model2)
#No hay multicolinealidad en nuestras variables
## Analisis del modelo 2

dif_residuos2 <- log_model2$null.deviance - log_model2$deviance
df2 <- log_model2$df.null - log_model2$df.residual
p_value2 <- pchisq(q = dif_residuos2, df = df2, lower.tail = FALSE)

##Matriz de confusion
pred2 <- predict.glm(log_model2, newdata = redwine.prueba, type = "response")
matriz_conf2 <- table(yprueba, floor(pred2+0.5),
                      dnn = c("Observaciones", "Predicciones"))
matriz_conf2
plot(matriz_conf2, col = c("#7b1c23", "#1c7b45"), main = "Modelo 2")

##Precision
accurate2 <- (matriz_conf2[1,1] + matriz_conf2[2,2]) / (sum(matriz_conf2))
accurate2

##Sensibilidad
sensib2 <- (matriz_conf2[2,2]) / (matriz_conf2[2,2] + matriz_conf2[2,1])
sensib2

##Especificidad
especf2 <- matriz_conf2[1,1] / (matriz_conf2[1,1] + matriz_conf2[1,2])
especf2

##Curva ROC
pred = ROCR::prediction(pred2, yprueba)
perf <- performance(pred, "tpr" , "fpr")
plot(perf, col = "#e30b5d", lwd = 3, main = "Modelo 2")

##AUC
AUC2 = performance(pred, measure = "auc")@y.values[[1]]
AUC2

##Intervalo de confianza
inf2 <- accurate2 + Qalpha*(sqrt((accurate2*(1 - accurate2))/dim(redwine.prueba)[1]))
sup2 <- accurate2 - Qalpha*(sqrt((accurate2*(1 - accurate2))/dim(redwine.prueba)[1]))

cat("Intervalo para P al 95% de Confianza",
    "\n", c(inf2, sup2), "\n", "Longitud: ", sup2 - inf2)

############################## Valores atipicos #####################

#Guardamos un data frame al cual le quitaremos valores atipicos
#Aplicaremos el criterio de Chauvenet, el cual establece que podemos 
#eliminar valores atipicos 

#Calculamos el coeficiente Kn

Qn <- 1/(4*dim(redwine)[1]) #Cuantil para la normal
Kn <- qnorm(Qn, mean = 0, sd = 1, lower.tail = FALSE)
mean_vector <- apply(redwine[1:11], 2, mean) #Vector de medias
#Como queremos los datos |Xi - Xbarra | > KnS
#Xi debe estar en el intervalo (Xbarra - KnS, Xbarra + KnS)

redwine_outliers <- redwine %>%
  filter(fixed.acidity < mean_vector[1]+ Kn * sd_vector[1]) %>%
  filter(volatile.acidity < mean_vector[2] + Kn * sd_vector[2]) %>%
  filter(citric.acid < mean_vector[3] + Kn * sd_vector[3]) %>%
  filter(residual.sugar < mean_vector[4] + Kn * sd_vector[4]) %>%
  filter(chlorides < mean_vector[5] + Kn * sd_vector[5]) %>%
  filter(free.sulfur.dioxide < mean_vector[6] + Kn * sd_vector[6]) %>%
  filter(total.sulfur.dioxide < mean_vector[7] + Kn * sd_vector[7]) %>%
  filter(density < mean_vector[8] + Kn * sd_vector[8]) %>%
  filter(pH < mean_vector[9] + Kn * sd_vector[9]) %>%
  filter(sulphates < mean_vector[10] + Kn * sd_vector[10]) %>%
  filter(alcohol < mean_vector[11] + Kn * sd_vector[11])

##Hacemos un resumen
summary(redwine_outliers)
#Quedaron 1520 vinos (el 95%)
##Analisis de medidas
aggregate(. ~ Good.quality, data = redwine_outliers, mean)
aggregate(. ~ Good.quality, data = redwine_outliers, median)

##DIferencia de medias
wilcox_test(fixed.acidity ~ Good.quality, data = redwine_outliers)
wilcox_test(volatile.acidity ~ Good.quality, data = redwine_outliers) 
wilcox_test(citric.acid ~ Good.quality, data = redwine_outliers)
wilcox_test(residual.sugar ~ Good.quality, data = redwine_outliers) #No rechazo H0
wilcox_test(chlorides ~ Good.quality, data = redwine_outliers)
wilcox_test(free.sulfur.dioxide ~ Good.quality, data = redwine_outliers)
wilcox_test(total.sulfur.dioxide ~ Good.quality, data = redwine_outliers)
wilcox_test(density ~ Good.quality, data = redwine_outliers)
wilcox_test(pH ~ Good.quality, data = redwine_outliers) #No rechazo H0
wilcox_test(sulphates ~ Good.quality, data = redwine_outliers)
wilcox_test(alcohol ~ Good.quality, data = redwine_outliers)

#Solo azucar residual y pH tienen medias estadisticamente iguales

##Curva de andrews
par(mfrow = c(1,1))
andrews(redwine_outliers, type = 1, clr = 3, step = 60, ymax = 5) 

##Boxplots
par(mfrow = c(2,3))
for(i in 1:11){
  boxplot(redwine_outliers[i], border = "dark blue", main = names(redwine_outliers[i]))
}

##Histogramas
par(mfrow = c(1,1))
multi.hist(redwine_outliers[1:11], bcol = "light blue", 
           dcol = c("#0e3d22", "#e30b5d"),lwd = 3)

#################### particion de data 2 (sin valores atipicos) ###########
n <- dim(redwine_outliers)[1]
set.seed(1989)  ##Taylor's version
entrenamiento2 <- sample(1:n, 0.66*n)

redwine.prueba2 <- redwine_outliers[-entrenamiento2,]
redwine.entrenamiento2 <- redwine_outliers[entrenamiento2,]

yentrenamiento2 <- redwine_outliers$Good.quality[entrenamiento2]
yprueba2 <- redwine_outliers$Good.quality[-entrenamiento2]


############# Modelo 3 (no datos atipicos)################
log_model3 <- glm(Good.quality ~ .,
                  data = redwine.entrenamiento2, family = binomial)
summary(log_model3)
confint(log_model3)
vif(log_model3)
#Existe multicolinealidad
#QUitamos Acidez fija, acido citrico, azucar residual, cloruros
#SO2 libre, densidad y pH.
##Analisis del modelo 3
dif_residuos3 <- log_model3$null.deviance - log_model3$deviance
df3 <- log_model3$df.null - log_model3$df.residual
p_value3 <- pchisq(q = dif_residuos3, df = df3, lower.tail = FALSE)

##Matriz de confusion
pred3 <- predict.glm(log_model3, newdata = redwine.prueba2, type = "response")
matriz_conf3 <- table(yprueba2, floor(pred3+0.5),
                      dnn = c("Observaciones", "Predicciones"))
matriz_conf3
plot(matriz_conf3, col = c("#7b1c23", "#1c7b45"), main = "Modelo 3")

##Precision
accurate3 <- (matriz_conf3[1,1] + matriz_conf3[2,2]) / (sum(matriz_conf3))
accurate3

##Sensibilidad
sensib3 <- (matriz_conf3[2,2]) / (matriz_conf3[2,2] + matriz_conf3[2,1])
sensib3

#Especificidad
especf3 <- matriz_conf3[1,1] / (matriz_conf3[1,1] + matriz_conf3[1,2])
especf3

##Curva ROC
pred = ROCR::prediction(pred3, yprueba2)
perf <- performance(pred, "tpr" , "fpr")
plot(perf, col = "#c715b2", lwd = 3, main = "Modelo 3")

##AUC
AUC3 = performance(pred, measure = "auc")@y.values[[1]]
AUC3

##Intervalo de confianza
inf3 <- accurate3 + Qalpha*(sqrt((accurate3*(1 - accurate3))/dim(redwine.prueba2)[1]))
sup3 <- accurate3 - Qalpha*(sqrt((accurate3*(1 - accurate3))/dim(redwine.prueba2)[1]))

cat("Intervalo para P al 95% de Confianza",
    "\n", c(inf3, sup3), "\n", "Longitud: ", sup3 - inf3)


################ Modelo 4 (outliers 2)############
log_model4 <- glm(Good.quality ~ volatile.acidity + total.sulfur.dioxide + 
                    sulphates + alcohol, data = redwine.entrenamiento2, 
                  family = binomial)
summary(log_model4)
confint(log_model4)
vif(log_model4)
#Todos los coeficientes son significativos y no hay multicolinealidad

##Analisis 
dif_residuos4 <- log_model4$null.deviance - log_model4$deviance
df4 <- log_model4$df.null - log_model4$df.residual
p_value4 <- pchisq(q = dif_residuos4, df = df4, lower.tail = FALSE)

##Matriz de confusion
pred4 <- predict.glm(log_model4, newdata = redwine.prueba2, type = "response")
matriz_conf4 <- table(yprueba2, floor(pred4+0.5),
                      dnn = c("Observaciones", "Predicciones"))
matriz_conf4
plot(matriz_conf4, col = c("#7b1c23", "#1c7b45"), main = "Modelo 4")

##Precision
accurate4 <- (matriz_conf4[1,1] + matriz_conf4[2,2]) / (sum(matriz_conf4))
accurate4

#Sensibilidad
sensib4 <- (matriz_conf4[2,2]) / (matriz_conf4[2,2] + matriz_conf4[2,1])
sensib4

#Especificidad
especf4 <- matriz_conf4[1,1] / (matriz_conf4[1,1] + matriz_conf4[1,2])
especf4

##Curva ROC
pred = ROCR::prediction(pred4, yprueba2)
perf <- performance(pred, "tpr" , "fpr")
plot(perf, col = "#c715b2", lwd = 3, main = "Modelo 4")

##AUC
AUC4 = performance(pred, measure = "auc")@y.values[[1]]
AUC4

##Intervalo de confianza
inf4 <- accurate4 + Qalpha*(sqrt((accurate4*(1 - accurate4))/dim(redwine.prueba2)[1]))
sup4 <- accurate4 - Qalpha*(sqrt((accurate4*(1 - accurate4))/dim(redwine.prueba2)[1]))

cat("Intervalo para P al 95% de Confianza",
    "\n", c(inf4, sup4), "\n", "Longitud: ", sup4 - inf4)

################ Comparacion de los 4 modelos ####################
modelos <- c("Modelo 1" , "Modelo 2", "Modelo 3", "Modelo 4")
dif_resid <- c(dif_residuos, dif_residuos2, dif_residuos3, dif_residuos4)
grados_lib <- c(df, df2, df3, df4)
p_values <- c(p_value, p_value2, p_value3, p_value4)
AIC <- c(logistic_model$aic, log_model2$aic, log_model3$aic, log_model4$aic)
Devianza <- c(logistic_model$deviance, log_model2$deviance, log_model3$deviance, 
              log_model4$deviance)
Precision <- c(accurate1, accurate2, accurate3, accurate4)
Sensibilidad <- c(sensib1, sensib2, sensib3, sensib4)
Especificidad <- c(especf1, especf2, especf3, especf4)
AUC <- c(AUC1, AUC2, AUC3, AUC4)
Long_IC <- c(sup1 - inf1, sup2 - inf2, sup3 - inf3, sup4 - inf4)

Comp_df <- data.frame(modelos, dif_resid, grados_lib, p_values, AIC,
                      Devianza, Precision, Sensibilidad, Especificidad, AUC,
                      Long_IC)
Comp_df
View(Comp_df)

par(mfrow = c(2,2))

hist(logistic_model$residuals, probability = TRUE, col = "#0be391",
     main = "Residuales Modelo 1")
lines(density(logistic_model$residuals), col = "#e30b5d", lwd = 3)

hist(log_model2$residuals, probability = TRUE, col = "#0be391",
     main = "Residuales Modelo 2")
lines(density(log_model2$residuals), col = "#e30b5d", lwd = 3)

hist(log_model3$residuals, probability = TRUE, col = "#0be391",
     main = "Residuales Modelo 3")
lines(density(log_model3$residuals), col = "#e30b5d", lwd = 3)

hist(log_model4$residuals, probability = TRUE, col = "#0be391",
     main = "Residuales Modelo 4")
lines(density(log_model4$residuals), col = "#e30b5d", lwd = 3)

##### Nos quedamos con el 3 o 4, depende la metrica que tomemos

