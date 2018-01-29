# Document-classification-in-python.
Implemented the Multinomial Naive Bayes algorithm to classify documents into different classes.

The details of this algorithm can be read from chapter 13 of the book "Introduction to Information
Retrieval" by Manning et al. This chapter can be downloaded from:
http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf

I will be using the 20 newsgroups dataset, which is available for download
from here: http://qwone.com/~jason/20Newsgroups/
I will use the "20 Newsgroups sorted by date" version. The direct link for this dataset is:
http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
This dataset contains folders for training and test portions, with a sub-folder for different
classes in each portion. For example, in the train portion, there is a sub-folder for computer
graphics class, titled "comp.graphics" and a similar sub-folder exists in the test portion. To
simplify storage and memory requirements, I have selected  5 classes out of these 20 to
use for training and test portions. As always, you have to train your algorithm using the
training portion and test it using the test portion.

