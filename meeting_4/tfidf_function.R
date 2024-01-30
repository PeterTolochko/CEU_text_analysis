create_dtm <- function(texts) {
  # Creates Document-Term Matrix from raw text
  
  texts <- stringr::str_to_lower(texts)
  all_words <- stringr::str_split(texts, " ") %>% unlist()
  unique_terms <- stringr::str_split(texts, " ") %>% unlist() %>% unique()
  
  n <- length(texts)
  k <- length(unique_terms)
  x <- matrix(nrow = n, ncol = k, 0)
  
  for (i in 1:nrow(x)) {
    current_text <- stringr::str_split(texts[i], " ")[[1]]
    for (j in 1:ncol(x)) {
      for (y in 1:length(current_text)) {
        if (unique_terms[j] %in% current_text[y]) {
          x[i, j] = x[i, j] + 1
        } 
      }
    }
  }
  
  rownames(x) <- paste0("document_", 1:3)
  colnames(x) <- unique_terms
  
  return(x)
}

calculate_tfidf <- function(dtm) {
  
  n_docs <- nrow(dtm) # total number of documents
  tf <- dtm / rowSums(dtm) # Calculate term frequency (TF)
  idf <- log(n_docs / colSums(dtm > 0)) # Calculate IDF
  tfidf <- tf * idf # combine into TF-IDF
  
  return(tfidf)
}


# example usage:

texts <- c("John loves icecream",
           "John loves oranges",
           "Marry hates icecream")

dtm <- create_dtm(texts)

calculate_tfidf(my_dtm)
