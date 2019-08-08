#Analysis000.ipynb

Sentiment analysis is natural language processing method to identify and quantify the subjective information. 
It can classify some information into three reaction, they are positive, negative and neutral.

https://miro.medium.com/max/689/1*jHzNpL-KagnaHUSHzPTPkA.jpeg

You can create simple sentiment analysis easily with python.
All dependency that you need is TextBlob. You can install it with pip by this following command.

# pip install textblob

Lets begin with creating a python file and add the following code.
First, import the TextBlob function.

# from textblob import TextBlob

Then, create a variable to store input text that you want to know its sentiment with TextBlob function.

#put your input inside TextBlob
# analysis = TextBlob('not a very great experiment')

Finally you can print the sentiment value with this following code.

# print(analysis.sentiment)

You will see the output with the two value, they are polarity and subjectivity.
The polarity score is a float within the range [-1.0, 1.0].
The subjectivity is a float within the range [0.0, 1.0].
Polarity indicates the sentiment, minus is for negative, 0 is for neutral and positive is for positive statement.
Subjectivity indicates that if its close to 0 means objective statement, but if its close to 1 means the statement is very subjective.
The output of code above is such the following.

#Sentiment(polarity=-0.30769230692377,subjectivity=0.57692369230769)

The sentence “not a very great experiment” has a polarity of about -0.3, meaning it is slightly negative, and a subjectivity of about 0.6, meaning it is fairly subjective.
Congratulation, you have created your simple sentiment analysis. 

Good luck at your experiment.
