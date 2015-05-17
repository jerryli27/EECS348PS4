
    # Name:
# Date:
# Description:
#
#



import math, os, pickle, re, random,copy,struct,time,numpy

class Bayes_Classifier:

    def __init__(self,pos_dict_file_name='pos_dict',neg_dict_file_name='neg_dict',k=0):
        # Used for the Kth Cross Validation. K ranges from 0 to 9

        """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text."""
        self.pos_dict={}
        self.neg_dict={}
        self.pos_total=0
        self.neg_total=0
        # CorrelationMatrix is a matrix-like dictionary that stores the 10 synonyms of almost all words.
        # It can be used in classifier to decrease the size of corpus we need to feed to the program by adding counts to
        # not only words that actually occurred, but also its synonyms.
        # There are two places we could use CorrelationMatrix, one is when we build the dictionary. The other one is
        # when we are classifying. Now I chose to use it only in classifying. We could use it in both.
        try:
            self.correlationMatrix=self.load('correlationMatrix')
        except:
            print 'Unable to load pickle files: correlationMatrix!'
            return
        if os.path.isfile(pos_dict_file_name) and os.path.isfile(neg_dict_file_name):
            try:
                self.pos_dict=self.load(pos_dict_file_name)
                self.neg_dict=self.load(neg_dict_file_name)
            except:
                print 'Unable to load pickle files: ' + pos_dict_file_name + ' and ' + neg_dict_file_name + '!'
                return
            for key in self.pos_dict.keys():
                self.pos_total+=self.pos_dict[key]
            for key in self.neg_dict.keys():
                self.neg_total+=self.neg_dict[key]
        elif k!=0:
            self.train(pos_dict_file_name,neg_dict_file_name,k)
        else:
            self.train(pos_dict_file_name,neg_dict_file_name)

    def train(self,pos_dict_file_name,neg_dict_file_name,k=-1):
        # Used for the Kth Cross Validation. K ranges from 0 to 9
        """Trains the Naive Bayes Sentiment Classifier."""
        lFileList = []
        for fFileObj in os.walk("../movies_reviews/"):
            lFileList = fFileObj[2]
            break
        if k>=0 and k<10:
            # shuffle the list (in the same manner every time) so that there's an equal chance to get positives/negatives
            random.seed(0)
            random.shuffle(lFileList)
            lFileList=lFileList[0:k*len(lFileList)/10]+lFileList[(k+1)*len(lFileList)/10:]
        for fileName in lFileList:
            star = fileName[7]
            #str(re.match('(?:movies\-)\d+(?:\-)', fileName).group(1))
            # If bad review
            if star=='1':
                tokens=self.tokenize(self.loadFile("../movies_reviews/"+fileName))
                for token in tokens:
                    if token in self.neg_dict.keys():
                        self.neg_dict[token]+=1
                    else:
                        self.neg_dict[token]=1
                    self.neg_total+=1
            elif star=='5':
                tokens=self.tokenize(self.loadFile("../movies_reviews/"+fileName))
                for token in tokens:
                    if token in self.pos_dict.keys():
                        self.pos_dict[token]+=1
                    else:
                        self.pos_dict[token]=1
                    self.pos_total+=1
        self.save(self.pos_dict,pos_dict_file_name)
        self.save(self.neg_dict,neg_dict_file_name)
        print 'pos_dict have '+str(len(self.pos_dict))+' words'
        print 'neg_dict have '+str(len(self.neg_dict))+' words'

    def classify(self, sText):
        """Given a target string sText, this function returns the most likely document
        class to which the target string belongs (i.e., positive, negative or neutral).
        """
        tokens=self.tokenize(sText)
        # The log of likelihood of the text being positive is just log of all frequencies,
        # which is log(number_of_times_this_word_appear/total_number_of_words_in_positive).
        # We could simplify this to sum(log(number_of_times_this_word_appear))-length(text)*log(total_num_of...)

        pos_pred=-math.log(self.pos_total)*len(tokens)
        neg_pred=-math.log(self.neg_total)*len(tokens)

        # The main sentiment of a sentence usually resites after "turning" words such as "but" and "however"
        # Therefore in this improvement attempt, we only classify the text after "but" and "however"
        # First, we make a copy of the tokens that contains all lowercase words to find the index of any "but" or "after"
        tokensCopy = copy.deepcopy(tokens)
        for words in tokensCopy:
            words = words.lower()
        if "but" in tokensCopy:
            butIndex = tokensCopy.index("but")
            tokens = tokens[butIndex:]
        elif "however" in tokensCopy:
            howevIndex = tokensCopy.index("however")
            tokens = tokens[howevIndex:]

        for token in tokens:
            tokenCopy = copy.deepcopy(token)    # for avoiding calling string functions on original token
            # count the total appearance of token in all form (abc, Abc, ABC) in pos_dict and neg_dict
            posCount = 1
            negCount = 1

            if tokenCopy.lower() in self.correlationMatrix.keys():
                for pair in self.correlationMatrix[tokenCopy.lower()]:
                    pairToken=pair[0]
                    pairSimilarity=pair[1]
                    pairTokenCopy=copy.deepcopy(pairToken)
                    if pairToken in self.pos_dict.keys():
                        posCount += self.pos_dict[pairToken]
                    ##Dont need to check upper or lower because everything in correlationMatrix is lower
                    #elif pairToken.islower():
                    if pairTokenCopy.upper() in self.pos_dict.keys():
                        posCount += self.pos_dict[pairTokenCopy.upper()]
                    if pairTokenCopy.title() in self.pos_dict.keys():
                        posCount += self.pos_dict[pairTokenCopy.title()]
                    if pairToken in self.neg_dict.keys():
                        negCount += self.neg_dict[pairToken]
                    #elif pairToken.islower():
                    if pairTokenCopy.upper() in self.neg_dict.keys():
                        negCount += self.neg_dict[pairTokenCopy.upper()]
                    if pairTokenCopy.title() in self.neg_dict.keys():
                        negCount += self.neg_dict[pairTokenCopy.title()]

            # # All-cap means emphasizing, in this improvement attempt we increase the "weight" of all-capped words
            # if token.isupper():
            #     # if token appears significantly more in pos_dict than in neg_dict, increase its weight in pos_pred
            #     if float(posCount)/negCount > 2:
            #         pos_pred = pos_pred + math.log(posCount) + math.log(2)
            #         neg_pred += math.log(negCount)
            #     elif float(negCount)/posCount > 2:
            #         pos_pred += math.log(posCount)
            #         neg_pred = neg_pred + math.log(negCount) + math.log(2)
            #     else:
            #         if token in self.pos_dict.keys() or tokenCopy.title() in self.pos_dict.keys() or tokenCopy.lower() in self.pos_dict.keys():
            #             pos_pred+=math.log(posCount)
            #         if token in self.neg_dict.keys() or tokenCopy.title() in self.neg_dict.keys() or tokenCopy.lower() in self.neg_dict.keys():
            #             neg_pred+=math.log(negCount)
            # else:
            #     if token in self.pos_dict.keys() or tokenCopy.title() in self.pos_dict.keys():
            #         pos_pred+=math.log(posCount)
            #     if token in self.neg_dict.keys() or tokenCopy.title() in self.neg_dict.keys():
            #         neg_pred+=math.log(negCount)
            # #else is just *=1, which does nothing
            else:
                if token in self.pos_dict.keys():
                    posCount += self.pos_dict[token]
                if token.isupper():
                    if tokenCopy.lower() in self.pos_dict.keys():
                        posCount += self.pos_dict[tokenCopy.lower()]
                    if tokenCopy.title() in self.pos_dict.keys():
                        posCount += self.pos_dict[tokenCopy.title()]
                elif token.islower():
                    if tokenCopy.upper() in self.pos_dict.keys():
                        posCount += self.pos_dict[tokenCopy.upper()]
                    if tokenCopy.title() in self.pos_dict.keys():
                        posCount += self.pos_dict[tokenCopy.title()]
                elif token.istitle():
                    if tokenCopy.upper() in self.pos_dict.keys():
                        posCount += self.pos_dict[tokenCopy.upper()]
                    if tokenCopy.lower() in self.pos_dict.keys():
                        posCount += self.pos_dict[tokenCopy.lower()]

                if token in self.neg_dict.keys():
                    negCount += self.neg_dict[token]
                if token.isupper():
                    if tokenCopy.lower() in self.neg_dict.keys():
                        negCount += self.neg_dict[tokenCopy.lower()]
                    if tokenCopy.title() in self.neg_dict.keys():
                        negCount += self.neg_dict[tokenCopy.title()]
                elif token.islower():
                    if tokenCopy.upper() in self.neg_dict.keys():
                        negCount += self.neg_dict[tokenCopy.upper()]
                    if tokenCopy.title() in self.neg_dict.keys():
                        negCount += self.neg_dict[tokenCopy.title()]
                elif token.istitle():
                    if tokenCopy.upper() in self.neg_dict.keys():
                        negCount += self.neg_dict[tokenCopy.upper()]
                    if tokenCopy.lower() in self.neg_dict.keys():
                        negCount += self.neg_dict[tokenCopy.lower()]

            if tokenCopy.lower() in self.pos_dict.keys() or tokenCopy.upper() in self.pos_dict.keys() or tokenCopy.title() in self.pos_dict.keys():
                pos_pred+=math.log(posCount)
            if tokenCopy.lower() in self.neg_dict.keys() or tokenCopy.upper() in self.neg_dict.keys() or tokenCopy.title() in self.neg_dict.keys():
                neg_pred+=math.log(negCount)



        #Need to code a Neutral zone
        print 'pos_pred = '+str(pos_pred)
        print 'neg_pred = '+str(neg_pred)
        if pos_pred>neg_pred:
            return 'positive'
        elif pos_pred<neg_pred:
            return 'negative'
        else:
            return 'neutral'
        # My thoughts on how to improve the classification.
        # Every word plays a different role in sentiment analysis. Some are more important than others.
        # For example: the word 'AI' is not so important as 'likes'
        # By putting different weights on words and train the weights using gradient descent and cross validation
        # We could get the optimal weight for each word. The predictions will become
        # sum(WEIGHT * log(number_of_times_this_word_appear))-length(text)*log(total_num_of...)

        # Another limit on the quality of classification is Naive Bayes itself.
        # It assumes the words have no connection with each other, but in fact the meaning is largely depend on
        # Which word is followed by which. My thought is lave a n*n matrix represent which words appeared around which.
        # The limit of this matrix is that it will be super sparse given how small our corpus(training data) is.
        # So if we could make the matrix more robust by also altering the values of the synonyms of the words we've seen
        # This could maybe work.

    def loadFile(self, sFilename):
        """Given a file name, return the contents of the file as a string."""

        f = open(sFilename, "r")
        sTxt = f.read()
        f.close()
        return sTxt

    def save(self, dObj, sFilename):
        """Given an object and a file name, write the object to the file using pickle."""

        f = open(sFilename, "w")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()

    def load(self, sFilename):
        """Given a file name, load and return the object stored in the file."""

        f = open(sFilename, "r")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText):
        """Given a string of text sText, returns a list of the individual tokens that
        occur in that string (in order)."""

        lTokens = []
        sToken = ""
        for c in sText:
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
                sToken += c
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))

        if sToken != "":
            lTokens.append(sToken)

        return lTokens

def tenFoldCrossValidation(precisionRecallFileName='precisionRecallFile'):
    # Initialize Precisions and Recall values
    precision=0.0
    recall=0.0

    for i in range(0,10):
        bs=Bayes_Classifier('pos_dict'+str(i),'neg_dict'+str(i),i)
        lFileList = []
        for fFileObj in os.walk("../movies_reviews/"):
            lFileList = fFileObj[2]
            break
        # shuffle the list (in the same manner every time) so that there's an equal chance to get positives/negatives
        random.seed(0)
        random.shuffle(lFileList)
        lFileList=lFileList[i*len(lFileList)/10:(i+1)*len(lFileList)/10]
        #Initialize counters to calculate precision recall
        falsePos=0.0; falseNeg=0.0; truePos=0.0; trueNeg=0.0
        for fileName in lFileList:
            result=bs.classify(bs.loadFile("../movies_reviews/"+fileName))
            star = fileName[7]
            # If bad review
            if star=='1':
                if result=='positive':
                    falsePos+=1
                elif result=='negative':
                    trueNeg+=1
            elif star=='5':
                if result=='positive':
                    truePos+=1
                elif result=='negative':
                    falseNeg+=1
        precision+=(truePos/(truePos+falsePos)+trueNeg/(trueNeg+falseNeg))/2
        recall+=(truePos/(truePos+falseNeg)+trueNeg/(trueNeg+falsePos))/2
    precision/=10
    recall/=10
    f1=2*precision*recall/(precision+recall)
    with open(precisionRecallFileName,'a') as precisionRecallFile:
        precisionRecallFile.write(str(precision)+','+str(recall)+','+str(f1)+'\n')
        print (str(precision)+','+str(recall)+','+str(f1)+'\n')

def readCSV(fileName):
    with open(fileName, "r") as f:
        #x=struct.unpack('>q',f.read(8))
        #print x
        #x=struct.unpack('<q',f.read(8))
        #print x
        vocabSize=0
        vectorLength=0
        vocab={}
        count=0
        for line in f:
            line=line[:-1]
            line=line.split(',')
            if vocabSize==0 and vectorLength==0:
                vocabSize=int(line[0])
                vectorLength=int(line[1])
            else:
                vector=[]
                word=line[0]
                for j in range(1,vectorLength+1):
                    vector.append(float(line[j]))
                vectorSum=numpy.sqrt(numpy.dot(vector,vector))
                for j in range(0,vectorLength):
                    vector[j]/=vectorSum
                vocab[word]=vector
            count+=1
            if count>30000:
                break
    return vocabSize,vectorLength,vocab

def generateCorrelationMatrix(vocab):
    correlationMatrix={}
    count=0
    for word in vocab.keys():
        synonyms=[['',-1]]*10
        for candidate in vocab.keys():
            #The dot product seems to have some problem
            similarity=numpy.dot(vocab[word],vocab[candidate])
            for i in range(0,10):
                if synonyms[i][1]<similarity:
                    synonyms.pop()
                    synonyms.insert(i,[candidate,similarity])
                    break
        correlationMatrix[word]=synonyms
        count+=1
        if count%200==0:
            print str(float(count)/len(vocab.keys())) + '% Completed!'
    return correlationMatrix

def generateCorrelationMatrix_ver2(vocab):
    correlationMatrix=[]
    for word in vocab.keys()[:30000]:
        correlationMatrix.append(vocab[word])
    correlationMatrixTranspose=numpy.array(correlationMatrix)
    correlationMatrixTranspose=correlationMatrixTranspose.transpose()
    correlationMatrixProduct=numpy.dot(correlationMatrix,correlationMatrixTranspose)
    print 'correlationMatrixProduct calculated!'
    correlationMatrixArgSort=numpy.argsort(correlationMatrixProduct)
    print 'correlationMatrixArgSort calculated!'
    correlationMatrix={}
    count=0
    for word in vocab.keys()[:30000]:
        if correlationMatrixArgSort[count][-1]!=count:
            print 'error'
        synonyms=[]
        for i in range(0,10):
            synonyms.append([vocab.keys()[correlationMatrixArgSort[count][-i-1]],correlationMatrixProduct[count][correlationMatrixArgSort[count][-i-1]]])
        correlationMatrix[word]=synonyms
        count+=1
        if count%200==0:
            print str(float(count)/len(vocab.keys())) + '% Completed!'
    return correlationMatrix
def save(dObj, sFilename):
    """Given an object and a file name, write the object to the file using pickle."""

    f = open(sFilename, "w")
    p = pickle.Pickler(f)
    p.dump(dObj)
    f.close()

def load(sFilename):
    """Given a file name, load and return the object stored in the file."""

    f = open(sFilename, "r")
    u = pickle.Unpickler(f)
    dObj = u.load()
    f.close()
    return dObj

start = time.time()
#NOTE: to use this, put the movies_reviews folder at the previous folder. It's a huge pain for github to load 20000+ files.
tenFoldCrossValidation()


# i=0
# bs=Bayes_Classifier('pos_dict'+str(i),'neg_dict'+str(i),i)
# print bs.classify('I love my AI class')
# while (1):
#     sentence = str(input('Enter the sentence you would like to classify: '))
#     print bs.classify(sentence)

#vocabSize,vectorLength,vocab=readCSV('trained file 100 noI.csv')
#print 'read word vec file'
#correlationMatrix=generateCorrelationMatrix_ver2(vocab)
#save(correlationMatrix,'correlationMatrix')
end = time.time()
print end - start