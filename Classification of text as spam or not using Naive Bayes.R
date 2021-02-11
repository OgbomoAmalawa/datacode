#####  Classification using Naive Bayes, B.Lantz, Machine Learning with R, PACKT Open Source-----

## Example: Filtering spam SMS messages ----

## Step 1: Collecting data from http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/ or 
## download the data sms_spam.csv file from the Packt website

## Step 2: Exploring and preparing the data ---- 

# read the sms data into the sms data frame
sms_raw <- read.csv("sms_spam.csv", stringsAsFactors = FALSE)

# examine the structure of the sms data: 5,559 total SMS messages, 2 features: type, text
str(sms_raw)

# convert spam/ham (character vector) to factor, as this is a categorical variable.
sms_raw$type <- factor(sms_raw$type)

# examine the type variable more carefully
str(sms_raw$type)
table(sms_raw$type)

# build a corpus using the text mining (tm) package providing some functionalities such as:
# removing numbers and punctuation; handling uninteresting words (e.g.and, but, or); breaking apart sentences into individual words
install.packages("tm")
library(tm)

# Creating a corpus, which is a collection of text document.
# In this example, the corpus will be a collection of SMS messages.
# VCorpus() - create a volatile corpus stored in memory rather than on disk (PCorpus()).
# VectorSource() reader function - create a source object from the existing sms_raw$text vector.
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

# examine the sms corpus
print(sms_corpus)
inspect(sms_corpus[1:2]) # view a summary of the 1st and 2nd SMS message in the corpus

as.character(sms_corpus[[1]]) # view the actual message text
lapply(sms_corpus[1:2], as.character) # view multiple documents

# clean up the corpus using tm_map() - provides a method to apply a transformation to a tm corpus
# tolower - returns a lowercase version of text strings.
# content_transformer() - treats tolowr() as a transformation function that can be used to access the corpus
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

# show the difference between sms_corpus and corpus_clean
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]]) # uppercase letters replaced by lowercase versions of the same.

sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers) # remove numbers
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords()) # remove stop words (to, and, but, or)
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation) # remove punctuation

# tip: create a custom function to replace (rather than remove) punctuation
removePunctuation("hello...world") # "helloworld"
replacePunctuation <- function(x) { gsub("[[:punct:]]+", " ", x) } # replace punctuation using a blank space
replacePunctuation("hello...world") # "hello world"

# illustration of word stemming (takes words like learned, learning, learns, and strips the suffix 
#  in order to transform them into the base form, learn)
library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns"))

# tm package includes a stemDocument() transformation to implement wordStem
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace) # eliminate unneeded whitespace

# examine the final clean corpus
lapply(sms_corpus[1:3], as.character)
lapply(sms_corpus_clean[1:3], as.character)

sms_corpus_clean <- Corpus(VectorSource(sms_corpus_clean))
# create a document-term sparse matrix - split the messages into individual words
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

# alternative solution: create a document-term sparse matrix directly from the SMS corpus
sms_dtm2 <- DocumentTermMatrix(sms_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

# alternative solution: using custom stop words function ensures identical result
sms_dtm3 <- DocumentTermMatrix(sms_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = function(x) { removeWords(x, stopwords()) },
  removePunctuation = TRUE,
  stemming = TRUE
))


# creating training and test datasets. SMS messages are sorted in a random order
sms_dtm_train <- sms_dtm2[1:4169, ] # 75% for training
sms_dtm_test  <- sms_dtm2[4170:5559, ] # 25% for testing

# also save the labels
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels  <- sms_raw[4170:5559, ]$type

# check that the proportion of spam is similar, both the traning and testing data contain ~13% spam
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

# word cloud visualization - visually depict the frequency at which words appear in text data.
# words appearing more often in the text are shown in a larger font
# while less common terms are shown in smaller fonts.
library(wordcloud)
# min.freq=50 : specifies the number of times (50, in this case 1% of the corpus) a word must appear in the corpus 
# before it will be displayed inthe cloud.
# random.order=False: the cloud will be arranged in a nonrandom order with higher frequency words placed closer to the centre.
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE) 

# subset the training data into spam and ham groups
spam <- subset(sms_raw, type == "spam")
ham  <- subset(sms_raw, type == "ham")

# max.words=40: look at the 40 most common words in each of the two sets
# scale() allows us to adjust the maximum and minimum font size for words in the cloud
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

sms_dtm_freq_train <- removeSparseTerms(sms_dtm_train, 0.999)
sms_dtm_freq_train

# creat indicator features for frequent words
findFreqTerms(sms_dtm_train, 5)

# save frequently-appearing terms to a character vector
sms_freq_words <- findFreqTerms(sms_dtm_train, 5) # the words appearing at least 5 times in the sms_dtm_train
str(sms_freq_words) # 1164 terms appearing in at least five SMS messages

# create DTMs with only the frequent terms. The training and testing sets now include 1164 features, which correspond to words appearing in at least five messages
sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

# convert counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# apply() convert_counts() to columns of train/test data
# MAGINE=2 as we're interested in the columns (1 for rows)
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)
# The results will be two character type matrixes, each with cells indicating 
# "Yes" or "No" for whether the word represented by the column appears at any point in the message represented by the row. 

## Step 3: Training a model on the data ----
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

## Step 4: Evaluating model performance ----
sms_test_pred <- predict(sms_classifier, sms_test)

library(gmodels)
# We ignore Pearson's Chi-squared test for independence between two variabes (prop.chisq)
# prop.t: table proportions; prop.r: row proportions; dnn: dimension names 
# for details see help(CrossTable)
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## Step 5: Improving model performance ----
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
