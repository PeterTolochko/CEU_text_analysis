#LSX in R

#script based on: https://koheiw.github.io/LSX/articles/pkgdown/basic.html


require(LSX)
require(quanteda)
require(tidyverse)


articles_en <- read_csv("headlines.csv")

corp <- corpus(articles_en, text_field = "headline")

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE, 
               remove_numbers = TRUE, remove_url = TRUE)
dfmt <- dfm(toks) |> 
  dfm_remove(stopwords("en"))

# build-in seed words
seed <- as.seedwords(data_dictionary_sentiment)
print(seed) 

# computes the polarity scores of all the words in the corpus based on their semantic similarity to the seed words
lss <- textmodel_lss(dfmt, seeds = seed, k = 300, cache = TRUE, 
                     include_data = TRUE, group_data = TRUE)

# Vizualisation of seed words
library(ggplot2)
textplot_terms(lss, highlighted = NULL) #highlighted = NULL, it randomly samples 50 words and highlights them

# Visualization of seed words + specified words
refugee <- featnames(dfm_select(dfmt, "ukip", valuetype = "regex"))
textplot_terms(lss, highlighted = c(refugee, names(seed)))

#Predict the polarity of documents
dat <- docvars(lss$data)
dat$lss <- predict(lss)
print(nrow(dat))

dat$publication_date <-as.Date(dat$publication_date)

#smooth polarity scores
smo <- smooth_lss(dat, lss_var = "lss", date_var = "publication_date")

#Plot over time

ggplot(smo, aes(x = date, y = fit)) + 
  geom_line() +
  geom_ribbon(aes(ymin = fit - se.fit * 1.96, ymax = fit + se.fit * 1.96), alpha = 0.1) +
  geom_vline(xintercept = as.Date("2022-02-24"), linetype = "dotted") +
  scale_x_date(date_breaks = "years", date_labels = "%Y") +
  labs(title = "Sentiment in UK Refugee/Asylum Coverage", x = "Date", y = "Sentiment")
