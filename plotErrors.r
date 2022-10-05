library(ggstatsplot)
library(gapminder)
dados <- read.csv2("C:/Users/eweri/Desktop/TCC/erros.csv", sep = ";", dec = ".") # nolint

dplyr::glimpse(dados)

plot(ggbetweenstats(
  data  = dados,
  x     = Estruturas,
  y     = MAE,
  type = "np",
  title = "Distribuição dos erros nas estruturas"
))
