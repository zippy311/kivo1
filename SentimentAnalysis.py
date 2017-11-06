import ast, sys
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#from nltk.tokenize import word_tokenize


if __name__ == "__main__":

    #*****

    n_instances = 100
    subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
    obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
    #print(len(subj_docs), len(obj_docs))
    #print(subj_docs[0])
    train_subj_docs = subj_docs[:80]
    test_subj_docs = subj_docs[80:100]
    train_obj_docs = obj_docs[:80]
    test_obj_docs = obj_docs[80:100]
    training_docs = train_subj_docs + train_obj_docs
    testing_docs = test_subj_docs + test_obj_docs
    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
    #print(len(unigram_feats))
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    training_set = sentim_analyzer.apply_features(training_docs)
    test_set = sentim_analyzer.apply_features(testing_docs)
    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)
    for key, value in sorted(sentim_analyzer.evaluate(test_set).items()):
        print('{0}: {1}'.format(key, value))

    #*****


    print()
    file = open('InsteonforHubReviews_10-30-17.json', 'r')
    appName = "Insteon for Hub"
    reviews = []
    numReviews = 0
    for line in file:
        dicto = ast.literal_eval(line)
        rev = dicto['Text'].replace("<span class=\"review-title\"> ", "")
        rev = rev.replace("<span class=\"review-title\">", "")
        rev = rev.replace('</span>', "")
        reviews.append(rev)
        numReviews = numReviews + 1

    print("App: ", appName, '\n')
    sid = SentimentIntensityAnalyzer()
    for review in reviews:
        i = 0
        while review[i] == ' ':
            i = i + 1
        rev = review[i:]
        try:
            print(rev)
        except UnicodeEncodeError:
            print(rev.encode(sys.stdout.encoding, errors='replace'))
        ss = sid.polarity_scores(rev)
        for k in sorted(ss):
            print('{0}: {1}, '.format(k, ss[k]), end='',)
        print('\n')

    #j = 0
    #while j < i:
    #    rev2 = reviews[j]
    #    words = word_tokenize(rev2)
        #useful_words = [word for word in words if word not in stopwords.words("english")]
        #my_dict = word_feats(useful_words)

        # u_w = [word for word in words if word not in stop]
        #try:
        #    print(rev2)
        #except UnicodeEncodeError:
            # print("Hi!!!")
        #    print(rev2.encode(sys.stdout.encoding, errors='replace'))
        #print(my_dict)
        #j = j + 1

    print("Number of Reviews: ", numReviews)
    file.close()
