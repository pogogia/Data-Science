#loading the libraries
library(rvest)
library(stringr)
library(xml2)
library(tm)

### getting the procduct for review

reviews_url <-'https://www.amazon.in/Redmi-Note-Pro-Storage-Processor/dp/B07X2KLKRZ/ref=sr_1_5?crid=TYDVH3Q7FVDQ&dchild=1&keywords=redmi+mobile+phones+all&qid=1590158901&sprefix=redmi+%2Caps%2C294&sr=8-5#customerReviews'


# getting text from URL for review using web scrapping 

MI_8_review<-NULL
for(i in 1:40){
  Amazonwebpage <- read_html(as.character(paste(reviews_url,sep=as.character(i))))
  srev <- Amazonwebpage %>%
    html_nodes(".review-text") %>%
    html_text()
  MI_8_review <- c(MI_8_review,srev)
}

# removing spaces and line breaks withing statements

reviews<-NULL
for (i in 1:length(MI_8_review)){
  trm<-trimws(str_replace_all(MI_8_review[i],"[\r\n]",""))
  reviews <- c(reviews,trm)
}

# writing in one text file this reviews
setwd('E:\\pooja\\DS\\ExcelR\\R\\Assignments')
write.table(reviews,"MI_8_review.txt",row.names = FALSE)

#readlines for corpus from given text 
x=readLines("E:\\pooja\\DS\\ExcelR\\R\\Assignments//MI_8_review.txt")

mydata.corpus <- Corpus(VectorSource(x))


# Data Pre processing using normalization,removing stop words,removing punctuations and numbers
mydata.corpus <- tm_map(mydata.corpus, removePunctuation)

my_stopwords <- c(stopwords('english'))

mydata.corpus <- tm_map(mydata.corpus, removeWords, my_stopwords)

mydata.corpus <- tm_map(mydata.corpus, removeNumbers)

mydata.corpus <- tm_map(mydata.corpus, stripWhitespace)

# build a term-document matrix
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
  main ="MI 8  reviews using Transformed Values", 
  xlab = "Narrative Time", 
  ylab = "Emotional Valence", 
  col = "red"
)

#Most Negative and Positive reviews
negative<-example[which.min(sentiment_vector)]
positive<-example[which.max(sentiment_vector)]

################
# > negative
# [1] "\"Worst phone, lot of problems  Read more\""
# > positive
# [1] "to light weight, great feature, photo quality with ultra wide angel, micro & 64 megapixel is awesome."

#############




