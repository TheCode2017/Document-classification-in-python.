# Document-classification-in-python.
Implemented the Multinomial Naive Bayes algorithm to classify documents into different classes.


In this part, you will implement the naïve Bayes algorithm for text classification tasks. The
version of naïve Bayes that you will implement is called the multinomial naïve Bayes (MNB). The
details of this algorithm can be read from chapter 13 of the book "Introduction to Information
Retrieval" by Manning et al. This chapter can be downloaded from:
http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf
Read the introduction and sections 13.1 and 13.2 carefully. The MNB model is presented in
Figure 13.2 Note that the algorithm uses add-one Laplace smoothing. Make sure that you do all
the calculations in log-scale to avoid underflow as indicated in equation 13.4.
To test your algorithm, you will use the 20 newsgroups dataset, which is available for download
from here: http://qwone.com/~jason/20Newsgroups/
You will use the "20 Newsgroups sorted by date" version. The direct link for this dataset is:
http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
This dataset contains folders for training and test portions, with a sub-folder for different
classes in each portion. For example, in the train portion, there is a sub-folder for computer
graphics class, titled "comp.graphics" and a similar sub-folder exists in the test portion. To
simplify storage and memory requirements, you can select any 5 classes out of these 20 to
use for training and test portions. As always, you have to train your algorithm using the
training portion and test it using the test portion.

