Sentiment Analysis with Python (Part 1)
Classifying IMDb Movie Reviews


Sentiment Analysis is a common NLP task that Data Scientists need to perform. This is a straightforward guide to creating a barebones movie review classifier in Python. Future parts of this series will focus on improving the classifier.

All of the code used in this series along with supplemental materials can be found in this GitHub Repository.

Data Overview

For this analysis we’ll be using a dataset of 50,000 movie reviews taken from IMDb. The data was compiled by Andrew Maas and can be found here: IMDb Reviews.

The data is split evenly with 25k reviews intended for training and 25k for testing your classifier. Moreover, each set has 12.5k positive and 12.5k negative reviews.

IMDb lets users rate movies on a scale from 1 to 10. To label these reviews the curator of the data labeled anything with = 4 stars as negative and anything with = 7 stars as positive. Reviews with 5 or 6 stars were left out.

Step 1: Download and Combine Movie Reviews

If you haven’t yet, go to IMDb Reviews and click on “Large Movie Review Dataset v1.0”. Once that is complete you’ll have a file called aclImdb_v1.tar.gz in your downloads folder.

Shortcut: If you want to get straight to the data analysis and/or aren’t super comfortable with the terminal, I’ve put a tar file of the final directory that this step creates here: Merged Movie Data. Double clicking this file should be sufficient to unpack it (at least on a Mac), otherwise gunzip -c movie_data.tar.gz | tar xopf — in a terminal will do it.

Unpacking and Merging

Follow these steps or run the shell script here:
 
Preprocessing Script

Move the tar file to the directory where you want this data to be stored.

Open a terminal window and cd to the directory that you put aclImdb_v1.tar.gz in.

gunzip -c aclImdb_v1.tar.gz | tar xopf -

cd aclImdb && mkdir movie_data

for split in train test; do for sentiment in pos neg; do for file in $split/$sentiment/*; do cat $file >> movie_data/full_${split}.txt; echo >> movie_data/full_${split}.txt; done; done; done;

Read into Python

For most of what we want to do in this walkthrough we’ll only need our reviews to be in a Python list. Make sure to point open to the directory where you put the movie data.

Step 3: Clean and Preprocess

The raw text is pretty messy for these reviews so before we can do any analytics we need to clean things up. Here’s one example:

Note: Understanding and being able to use regular expressions is a prerequisite for doing any Natural Language Processing task. If you’re unfamiliar with them perhaps start here: Regex Tutorial

Note: There are a lot of different and more sophisticated ways to clean text data that would likely produce better results than what I’ve done here. I wanted part 1 of this tutorial to be as simple as possible. Also, I generally think it’s best to get baseline predictions with the simplest possible solution before spending time doing potentially unnecessary transformations.

Vectorization

In order for this data to make sense to our machine learning algorithm we’ll need to convert each review to a numeric representation, which we call vectorization.

The simplest form of this is to create one very large matrix with one column for every unique word in your corpus (where the corpus is all 50k reviews in our case). Then we transform each review into one row containing 0s and 1s, where 1 means that the word in the corpus corresponding to that column appears in that review. That being said, each row of the matrix will be very sparse (mostly zeros). This process is also known as one hot encoding.

Step 4: Build Classifier

Now that we’ve transformed our dataset into a format suitable for modeling we can start building a classifier. Logistic Regression is a good baseline model for us to use for several reasons: (1) They’re easy to interpret, (2) linear models tend to perform well on sparse datasets like this one, and (3) they learn very fast compared to other algorithms.

To keep things simple I’m only going to worry about the hyperparameter C, which adjusts the regularization.

Note: The targets/labels we use will be the same for training and testing because both datasets are structured the same, where the first 12.5k are positive and the last 12.5k are negative.

It looks like the value of C that gives us the highest accuracy is 0.05.

Train Final Model

Now that we’ve found the optimal value for C, we should train a model using the entire training set and evaluate our accuracy on the 25k test reviews.

As a sanity check, let’s look at the 5 most discriminating words for both positive and negative reviews. We’ll do this by looking at the largest and smallest coefficients, respectively.

And there it is. A very simple classifier with pretty decent accuracy out of the box.