#import libraries
import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#define a readfile path that will walk through the emails and read the message body skiping the header
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

#define the dataframe function
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

#create a pandas dataframe from a dictoinary that initially contains empty lists of message and class

data = DataFrame({'message': [], 'class': []})
#get all emails from spam folder
data = data.append(dataFrameFromDirectory('C:\\Users\MADUABUCHI\\Desktop\\workSpace\\machinelearning\\SpamDetector\\emails\\spam', 'spam'))
#get all emails from ham folder
data = data.append(dataFrameFromDirectory('C:\\Users\MADUABUCHI\\Desktop\\workSpace\\machinelearning\\SpamDetector\\emails\\ham', 'ham'))
#use the head command to preview the first few entries
data.head()
#tokenize the words in the datafram using vectorizer
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

#Build a MultinomialNB classifier from sklearn
classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
examples = ['Do you like to play for free and can still win the broadcast?', "Did you see my previous mail?"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
predictions
