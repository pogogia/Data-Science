#load the data files

groceries <- read.transactions("E:/pooja/DS/ExcelR/R/Assignments/groceries.csv")
View(groceries)

movies <- read.csv("E:/pooja/DS/ExcelR/R/Assignments/my_movies.csv")
View(movies)

books <- read.csv("E:/pooja/DS/ExcelR/R/Assignments/book.csv")
View(books)

#checkingif we have nulls 

is.null(groceries)
is.null(books)
is.null(movies)

#loading the arules and arulesViz library
library(arules)
library(arulesViz)

##################  BOOKS DATASET ##################################################

#converting the books into categorical using factor

for(i in 1:ncol(books)){
  books[,i] <- as.factor(books[,i])
}

#Creating association rules
book_rules <- apriori(books)
summary(book_rules)

#inspecting the rules created
arules:: inspect(book_rules)

#creating rules with some filters
book_rules <- apriori(books, parameter = list(supp = 0.7, conf = 0.7))
arules::inspect(book_rules[1:5])

book_rules <- apriori(books, parameter = list(supp = 0.7, conf = 0.7, minlen = 3), 
                      appearance = list(rhs = c("Florence=0")))
arules::inspect(book_rules)

book_rules <- apriori(books, parameter = list(supp = 0.7, conf = 0.7, minlen = 3), 
                      appearance = list(rhs = c("Florence=0")))

book_rules <- apriori(books, parameter = list(supp = 0.7, conf = 0.7, minlen=3), 
                      appearance = list(lhs = c("ItalCook=0")))
arules::inspect(book_rules)

#creating different visualizations using arulesViz library
plot(book_rules)
plot(book_rules, method = "grouped")
plot(book_rules, method = "graph")


####################  GROCERY DATASET ########################################

#checking item frequency and plotting them 
itemFrequency(groceries, type = "absolute")
itemFrequencyPlot(groceries, topN=5,type="absolute")

#generating different association rules
groceries_rules <- apriori(groceries, parameter = list(support = 0.001 ,conf = 0.8,maxlen= 2))
arules:: inspect(groceries_rules[1:5])

groceries_rules <- apriori(groceries, parameter = list(support = 0.001 ,conf = 0.5,maxlen= 3),
                           appearance = list(lhs = c("sparkling")))
arules:: inspect(groceries_rules)

#creating different visualizations using arulesViz library
plot(groceries_rules[1:5], method = "graph")
plot(groceries_rules)
plot(groceries_rules, method = "grouped")


##################  MOVIES DATASET  #########################################


#getting only required variables
movies <- movies[,-c(1,2,3,4,5)]

#converting the movies into categorical using factor and then converting them to transactions

for(i in 1:ncol(movies)){
  movies[,i] <- as.factor(movies[,i])
}

movies <- as(movies, "transactions")

#generating different association rules
movies_rules <- apriori(movies, parameter = list(maxlen=3, supp =0.1, conf = 0.1 ))
arules::inspect(movies_rules[1:10])

movies_rules <- apriori(movies, parameter = list(maxlen=3, supp =0.1, conf = 0.1),
                                                 appearance = list(rhs= "Patriot=0"))
arules::inspect(movies_rules)

#creating different visualizations using arulesViz library
plot(movies_rules[1:5], method = "graph")
plot(movies_rules, jitter=0)
plot(movies_rules, method = "grouped")

