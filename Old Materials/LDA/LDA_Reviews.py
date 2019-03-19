import sys
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim, bz2
from gensim import corpora
import datetime

nl="\n"
noPasses=100
print("Number of Passes: ", noPasses, nl)
fileList = []
dt = datetime.datetime.now()
dtString = "Date/Time: " + dt.strftime("%Y-%m-%d %H:%M")

outTitle = "LDA_Results_"+ dt.strftime("%Y-%m-%d_%H-%M") + ".txt"

outFile = open(outTitle, 'w')
outFile.write(dtString+nl)
noPassString="Number of Passes: " + str(noPasses)
outFile.write(noPassString+nl)

file0 = open('AmazonAlexa_PRT_11-27-17-2.txt', 'r', encoding="ANSI")
file1 = open('InsteonHub_PRT_11-27-17-2.txt', 'r', encoding="utf-8")
file2 = open('Kevo_PRT_11-27-17-2.txt', 'r', encoding="ANSI")
file3 = open('Nest_PRT_11-27-17-2.txt', 'r', encoding="utf-8")
file4 = open('PhilipsHue_PRT_11-27-17.txt', 'r', encoding="utf-8")
file5 = open('WeMo_PRT_11-27-17.txt', 'r', encoding="utf-8")
file6 = open('AllReviews_3-22-18.txt', 'r', encoding="utf-8")

fileList.append(file0)
fileList.append(file1)
fileList.append(file2)
fileList.append(file3)
fileList.append(file4)
fileList.append(file5)
fileList.append(file6)

appName = ["Amazon Alexa", "Insteon for Hub", "Kevo", "Nest", "Philips Hue", "WeMo", "All Apps"]
ind = 0
for file in fileList:
    review_text = []

    for line in file:
        rev = line.replace("<span class=\"review-title\"> ", "")
        rev = rev.replace("<span class=\"review-title\">", "")
        rev = rev.replace('</span>', "")
        try:
            #print(rev)
            review_text.append(rev)
        except UnicodeDecodeError:
            #print(rev.encode(sys.stdout.encoding, errors='replace'))
            review_text.append(rev.encode(sys.stdout.encoding, errors='replace'))

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    extra = {"app", "amazon", "alexa", "cant", "wont", "didnt", "cant", "ive",
             "get", "would", "should", "could", "even", "insteon", "kevo", "nest",
             "philips", "hue", "wemo", "set", "use", "it", "its", "doesnt", "echo", "using", "hub",
             "im", "since", "dont", "go", "dot", "want", "work", "works", "worked", "working", "way",
             "need", "well", "make", "thing", "up", "one", "bit", "able", "like", "got", "seems", "also",
             "ever", "cannot", "going", "goes", "really", "used", "anymore", "0", "1", "2", "3", "4", "5", "6",
              "7", "8", "9", "10", "all", "needs", "u", "are", "otherwise", "am", "every", "still"}
            # "work" is a word that appeared in almost every topic. While the word might be useful,
            #  it might be worthwhile to see what other words will show up if "work" is excluded

    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])

        extra_free = " ".join([i for i in stop_free.lower().split() if i not in extra])
        punc_free = ''.join(ch for ch in extra_free if ch not in exclude)
        e2 = " ".join([i for i in punc_free.lower().split() if i not in extra])
        e3 = " ".join([i for i in e2.lower().split() if i not in stop])
        #sf2 = " ".join([i for i in punc_free.split() if i not in stop])

        #punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        try:
            normalized = " ".join(lemma.lemmatize(word) for word in e3.split())
        except UnicodeDecodeError:
            print("Error")
        return normalized

    doc_clean = [clean(doc).split() for doc in review_text]

    #for d in doc_clean:
    #    print(d)

    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    #ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
    ldamodel = Lda(doc_term_matrix, num_topics=20, id2word = dictionary, update_every=1, passes=noPasses)
    print(appName[ind])
    outFile.write(appName[ind]+nl)
    LDAResult = ldamodel.print_topics(num_topics=20, num_words=10)
    for topic in LDAResult:
        try:
            print(topic)
            outFile.write(str(topic)+nl)
        except UnicodeDecodeError:
            print(topic.encode(sys.stdout.encoding, errors='replace'))
            outFile.write(str(topic.encode(sys.stdout.encoding, errors='replace'))+nl)
    ind+=1
    outFile.write(nl)
    print(nl)


file0.close()
file1.close()
file2.close()
file3.close()
file4.close()
file5.close()
file6.close()
outFile.close()