setwd("C:/Users/ghjeuewb/Desktop")
library(rvest)
library(stringr)
library(xml2)
library(RSelenium)
library(tm)

### initializing ith URL in one variable

surl_1 <-'https://www.amazon.in/OnePlus-Mirror-Black-64GB-Storage/product-reviews/B0756Z43QS/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'


# getting text from URL for review using web scrapping 

oneplus_review<-NULL
for(i in 1:40){
  Amazonwebpage <- read_html(as.character(paste(surl_1,sep=as.character(i))))
  srev <- Amazonwebpage %>%
    html_nodes(".review-text") %>%
    html_text()
  oneplus_review <- c(oneplus_review,srev)
}

## removing spces and line breaks withing statements  using reguar expression

reviews<-NULL
for (i in 1:length(oneplus_review)){
  trm<-trimws(str_replace_all(oneplus_review[i],"[\r\n]",""))
  reviews <- c(reviews,trm)
}

## writing in one text file this reviews
write.table(reviews,"oneplus.txt",row.names = FALSE)

#readlines to make corpus from given text 
x=readLines("C:/Users/ghjeuewb/Desktop/oneplus.txt")

mydata.corpus <- Corpus(VectorSource(x))


# Text filtering using normalization,removing stop words,removing punctuations and numbers
mydata.corpus <- tm_map(mydata.corpus, removePunctuation)

my_stopwords <- c(stopwords('english'))

mydata.corpus <- tm_map(mydata.corpus, removeWords, my_stopwords)

mydata.corpus <- tm_map(mydata.corpus, removeNumbers)

mydata.corpus <- tm_map(mydata.corpus, stripWhitespace)

## build a term-document matrix
mydata.dtm3 <- TermDocumentMatrix(mydata.corpus)
dtm <- t(mydata.dtm3)


# finding total word counts per document
rowTotals <- apply(dtm, 1, sum)

#  considering documents with word count grater than 0
dtm.new<-dtm[rowTotals>0,]

#Latent Direchlt Allocation to finding hidden topics from the model
library(topicmodels)
lda <- LDA(dtm.new, 10) 

term <- terms(lda, 3) # first 3 terms of every topic
term

tops <- terms(lda)

tb <- table(names(tops), unlist(tops))
tb <- as.data.frame.matrix(tb)
View(tb)

##############################################################

library(syuzhet)
library(lubridate)
library(ggplot2)
library(scales)
library(dplyr)
library(reshape2)

x <- iconv(x, "UTF-8") #Unicode Transformation Format. The '8' means it uses 8-bit blocks to represent a character


######### emotions scores ######

# getting first 5 documnets emotions

tx <- get_nrc_sentiment(x)
head(tx,n=5)

# NRC happy and boring scores measure

get_nrc_sentiment('happy')
get_nrc_sentiment('boring')

# 
# afinn happy and boring scores measure

get_sentiment('boring',method="afinn")
get_sentiment('happy',method="afinn")


#each sentences by eight 
example<-get_sentences(x)
nrc_data<-get_nrc_sentiment(example)

# Bar plot for emotion mining
windows()
barplot(colSums(nrc_data), las = 1, col = rainbow(10), ylab = 'Count', main = 'Emotion scores')

# sentiments using three lexicons

sentiment_vector<-get_sentiment(example,method="bing")
sentiment_afinn<-get_sentiment(example,method="afinn")
sentiment_nrc<-get_sentiment(example,method="nrc")

# using afinn lexicon

sum(sentiment_afinn)
mean(sentiment_afinn)
summary(sentiment_afinn)


windows()
plot(sentiment_vector,type='l',main='Plot trajectory',xlab='Narative time',ylab='Emotional valence')
abline(h=0,col='red')

##Shape smoothing and normalization using a Fourier based transformation and low pass filtering is achieved using the get_transformed_values function as shown below.
ft_values <- get_transformed_values(
  sentiment_vector, 
  low_pass_size = 3, 
  x_reverse_len = 100,
  padding_factor = 2,
  scale_vals = TRUE,
  scale_range = FALSE
)

plot(
  ft_values, 
  type ="l", 
  main ="Oneplus reviews using Transformed Values", 
  xlab = "Narrative Time", 
  ylab = "Emotional Valence", 
  col = "red"
)

#Most Negative and Positive reviews
negative<-example[which.min(sentiment_vector)]
positive<-example[which.max(sentiment_vector)]

################
# > negative
# [1] "\"Worst product and embarrassing customer Care of Amazon...."
# > positive
# [1] "Gesture based actions is very useful, like for pausing a song, or to open flashlight.Overall, the phone is just pure awesome."

#############




