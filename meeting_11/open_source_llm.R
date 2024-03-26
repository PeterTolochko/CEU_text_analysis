# ollama https://ollama.com
# https://github.com/JBGruber/rollama

# install.packages("rollama")
# install.packages("httr2")

require(rollama)
require(tidyverse)

model_info <- pull_model("gemma")

rollama::embed_text("test")


# Zero-Shot Learning:
# Suppose you want to generate speeches on various political topics without providing specific examples.
# You can generate synthetic descriptions of political topics and ask the model to produce speeches. 

# Synthetic political topic descriptions
political_topics <- c(
  "An argument for free healthcare for all citizens.",
  "A speech advocating for stricter environmental regulations.",
  "A discussion on income inequality and potential solutions."
)

# Query the model to generate speeches based on the topic descriptions
for (topic in political_topics) {
  print(query(topic))
}


# One-shot learning example

# Provide a single example of a political speech
political_example <- "A speech advocating for stronger gun control laws."

# Query the model to generate more speeches based on the provided example
print(query(political_example))



# Few-Shot Classification:
# Suppose you want to classify political statements as liberal or conservative based on a few examples of each ideology.
# Let's generate synthetic data for this scenario:

# Few-shot classification example

# Provide a few examples of liberal and conservative statements
liberal_statements <- c(
  "Support for LGBTQ+ rights and marriage equality.",
  "Advocacy for universal healthcare.",
  "Promotion of renewable energy and climate action."
)

conservative_statements <- c(
  "Emphasis on traditional family values and marriage.",
  "Support for lower taxes and smaller government.",
  "Defense of Second Amendment rights."
)

# Query the model to classify additional statements as liberal or conservative
additional_statements <- c(
  "We must protect individual freedoms and limit government intervention.",
  "Government should provide more social welfare programs to support citizens.",
  "We need to prioritize national security and border control."
)

for (statement in additional_statements) {
  print(query(paste("Is the statement:", statement, "liberal or conservative? Choose only one option")))
}


# classification
# Query the model to classify into one of the provided classes
reviews_test <- read_csv("~/Desktop/Teaching/CEU_text_as_data/meeting_11/rotten_tomatoes_nb_svm_test.csv")

reviews_test$polarity_llama <- NA

set.seed(123)
small_test <- reviews_test[sample(1:nrow(reviews_test), 200), ]

for (i in 1:nrow(small_test)) {
  print(i)
  question <- "Is the sentiment of this text positive or negative? Answer only with a number: 1 if positive and 2 if negative. Here is the text:"
  text <- small_test[i,1]       
  concat <- paste(question, text)
  result <- query(concat, model = "gemma")
  # while(length(result) == 0){
  #   result <- query(concat)
  #   print(result)
  # }
  print(result$message$content)
  small_test$polarity_llama[i] <- result$message$content
}



# extract numbers 1 or 2 from the response
small_test$polarity_llama_binary <- as.numeric(gsub("[^1 | 2]", "", small_test$polarity_llama))

# small_test %>% write_csv("~/Desktop/Teaching/CEU_text_as_data/meeting_11/rotten_tomatoes_llama.csv")
small_test <- read_csv("~/Desktop/Teaching/CEU_text_as_data/meeting_11/rotten_tomatoes_llama.csv")


small_test <- small_test %>%
  mutate(polarity = case_when(
    polarity == 'positive' ~ 1,
    polarity == 'negative' ~ 2
  ))

small_test <- small_test %>%
  mutate(polarity_nb = case_when(
    polarity_nb == 'positive' ~ 1,
    polarity_nb == 'negative' ~ 2
  ))

small_test <- small_test %>%
  mutate(polarity_svm = case_when(
    polarity_svm == 'positive' ~ 1,
    polarity_svm == 'negative' ~ 2
  ))



metrics <- function(reviews_test, categ) {
  results_table <- table(ifelse(reviews_test$polarity == categ, 1, 0), ifelse(reviews_test$polarity_llama_binary == categ, 1, 0))
  recall <- results_table[2, 2] / (results_table[2, 2] + results_table[2, 1])
  precision <- results_table[2, 2] / (results_table[2, 2] + results_table[1, 2])
  f1 <- 2 * recall * precision / (recall + precision)
  
  return(c(precision, recall, f1))
}




package_metrics <- function(data, n_classes) { #n_classes is the number of categories here 2 (negative and positive)
  res <- matrix(NA, n_classes, 3) #3 relates to three metrics here precision, recall and F1
  for (c in 1:n_classes) {
    res[c, ] <- metrics(data, c)
  }
  return(res)
}


macro_f1_sent <- function(reviews_test){
  res_pos <- metrics(reviews_test,1)# Metrics for positive class
  res_neg <- metrics(reviews_test,2)# Metrics for negative class
  avg_f1 <- mean(c(res_pos[3], res_neg[3])) ## Calculate average F1 score
  avg_f1
  return(avg_f1)
}


accuracy <- function(reviews_test){
  res_table <- table(reviews_test$polarity, reviews_test$polarity_gpt)
  acc <- sum(diag(res_table))/sum(res_table)
  return(acc)
}




res_pos_sent <- metrics(small_test, 1) # precision, recall, F1 for the 'positive' label in sentiment analysis
res_pos_sent

res_neg_sent <- metrics(small_test, 2) # precision, recall; F1 for the 'negative' label in sentiment analysis
res_neg_sent


res_sent <- package_metrics(small_test, 2)
res_sent



avg_f1 <- macro_f1_sent(small_test) 
avg_f1