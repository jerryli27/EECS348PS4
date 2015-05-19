
    # Name:
# Date:
# Description:
#
#



import math, os, pickle, re, random,copy,struct,time,numpy

class Bayes_Classifier:

    def __init__(self,pos_dict_file_name='pos_dict_test',neg_dict_file_name='neg_dict_test',k=-1):
        # Used for the Kth Cross Validation. K ranges from 0 to 9

        """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text."""
        self.pos_dict=collections.defaultdict(int)
        self.neg_dict=collections.defaultdict(int)
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
            pos_keys=self.pos_dict.keys()
            for key in pos_keys:
                self.pos_total+=self.pos_dict[key]
                try:
                    for pair in self.correlationMatrix[key]:
                        synonym=pair[0]
                        similarity=pair[1]
                        self.pos_dict[synonym]+=similarity
                        self.pos_total+=similarity
                except:
                    pass
            neg_keys=self.neg_dict.keys()
            for key in neg_keys:
                self.neg_total+=self.neg_dict[key]
                try:
                    for pair in self.correlationMatrix[key]:
                        synonym=pair[0]
                        similarity=pair[1]
                        self.neg_dict[synonym]+=similarity
                        self.neg_total+=similarity
                except:
                    pass
        elif k!=-1:
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

            #Negation handling. Citation: http://arxiv.org/ftp/arxiv/papers/1305/1305.6143.pdf
            # F1 value went down so not using this

            # If bad review
            if star=='1':
                tokens=self.tokenize(self.loadFile("../movies_reviews/"+fileName))
                for token in tokens:
                    self.neg_dict[token]+=1
                    self.neg_total+=1
            elif star=='5':
                tokens=self.tokenize(self.loadFile("../movies_reviews/"+fileName))
                for token in tokens:
                    self.pos_dict[token]+=1
                    self.pos_total+=1
        pos_keys=self.pos_dict.keys()
        for key in pos_keys:
            try:
                for pair in self.correlationMatrix[key]:
                    synonym=pair[0]
                    similarity=pair[1]
                    self.pos_dict[synonym]+=similarity
                    self.pos_total+=similarity
            except:
                pass
        neg_keys=self.neg_dict.keys()
        for key in neg_keys:
            try:
                for pair in self.correlationMatrix[key]:
                    synonym=pair[0]
                    similarity=pair[1]
                    self.neg_dict[synonym]+=similarity
                    self.neg_total+=similarity
            except:
                pass

        #Training
        for i in range(0,40):
            for fileName in lFileList:
                content=self.loadFile("../movies_reviews/"+fileName)
                #result=bs.classify(content)
                star = fileName[7]
                # If bad review
                if star=='1':
                    result=self.gradientDescent('negative',content)
                elif star=='5':
                    result=self.gradientDescent('positive',content)
            sys.stdout.write('\r Training'+str(i/40.0)+'% Done')

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

        pos_pred=-math.log(self.pos_total*2)*len(tokens)
        neg_pred=-math.log(self.neg_total*2)*len(tokens)
        pos_count=0.0;neg_count=0.0
        for token in tokens:
            if self.pos_dict[token]!=0:
                pos_count+=1
                pos_pred+=math.log(self.pos_dict[token]+1)
            else:
                self.pos_dict[token]+=1
                self.pos_total+=1

            if self.neg_dict[token]!=0:
                neg_count+=1
                neg_pred+=math.log(self.neg_dict[token]+1)
            else:
                self.neg_dict[token]+=1
                self.neg_total+=1
            #else is just *=1, which does nothing


            try:
                for pair in self.correlationMatrix[token]:
                    synonym=pair[0]
                    pos_pred+=math.log(elf.pos_dict[synonym])
            except:
                pass

            try:
                for pair in self.correlationMatrix[token]:
                    synonym=pair[0]
                    neg_pred+=math.log(elf.neg_dict[synonym])
            except:
                pass
        # if pos_count<=0:
        #     pos_pred-=10000
        # else:
        #     pos_pred+=math.log(pos_count/(pos_count+neg_count))
        # if neg_count<=0:
        #     neg_pred-=10000
        # else:
        #     neg_pred+=math.log(neg_count/(pos_count+neg_count))


        #Need to code a Neutral zone
        #print 'pos_pred = '+str(pos_pred)
        #print 'neg_pred = '+str(neg_pred)
        if pos_pred>neg_pred:
            return 'positive'
        elif pos_pred<neg_pred:
            return 'negative'
        else:
            return 'neutral'

    def gradientDescent(self,result,sText):
        alpha=0.5
        """Given a target string sText, this function returns the most likely document
        class to which the target string belongs (i.e., positive, negative or neutral).
        """
        tokens=self.tokenize(sText)
        # The log of likelihood of the text being positive is just log of all frequencies,
        # which is log(number_of_times_this_word_appear/total_number_of_words_in_positive).
        # We could simplify this to sum(log(number_of_times_this_word_appear))-length(text)*log(total_num_of...)
        ### Without laplace smoothing
        # pos_pred=-math.log(self.pos_total)*len(tokens)
        # neg_pred=-math.log(self.neg_total)*len(tokens)
        # for token in tokens:
        #     if token in self.pos_dict.keys():
        #         pos_pred+=math.log(self.pos_dict[token])
        #     if token in self.neg_dict.keys():
        #         neg_pred+=math.log(self.neg_dict[token])
        ### With laplace smoothing and negation detection
        #Negation handling. Citation: http://arxiv.org/ftp/arxiv/papers/1305/1305.6143.pdf Useless
        #INB-1 algorithm Citation: http://ac.els-cdn.com/S0957417411016538/1-s2.0-S0957417411016538-main.pdf?_tid=9b8ab226-fcb9-11e4-bb4d-00000aacb35e&acdnat=1431883663_3277ab12b4a88f62e0026c4140cb2ddd Useless

        pos_pred=-math.log(self.pos_total*2)*len(tokens)
        neg_pred=-math.log(self.neg_total*2)*len(tokens)
        pos_count=0.0;neg_count=0.0
        for token in tokens:
            if self.pos_dict[token]!=0:
                pos_count+=1
                pos_pred+=math.log(self.pos_dict[token]+1)
            else:
                self.pos_dict[token]+=1
                self.pos_total+=1

            if self.neg_dict[token]!=0:
                neg_count+=1
                neg_pred+=math.log(self.neg_dict[token]+1)
            else:
                self.neg_dict[token]+=1
                self.neg_total+=1
            #else is just *=1, which does nothing
            try:
                for pair in self.correlationMatrix[token]:
                    synonym=pair[0]
                    pos_pred+=math.log(elf.pos_dict[synonym])
            except:
                pass

            try:
                for pair in self.correlationMatrix[token]:
                    synonym=pair[0]
                    neg_pred+=math.log(elf.neg_dict[synonym])
            except:
                pass

        #Need to code a Neutral zone
        #print 'pos_pred = '+str(pos_pred)
        #print 'neg_pred = '+str(neg_pred)
        if pos_pred>neg_pred:
            if result=='positive':
                return 'positive'
            else:
                # It should be negative but classified as positive. According to gradient descent, the delta value should be:
                for token in tokens:
                    temp=alpha*(1/self.neg_dict[token]-len(tokens)/self.neg_total)
                    self.neg_dict[token]+=temp
                    self.neg_total+=temp
                    try:
                        for pair in self.correlationMatrix[token]:
                            synonym=pair[0]
                            temp=alpha*(1/self.neg_dict[synonym]-len(tokens)/self.neg_total)
                            self.neg_dict[synonym]+=temp
                            self.neg_total+=temp
                    except:
                        pass

                return 'positive'
        elif pos_pred<neg_pred :
            if result=='negative':
                return 'negative'
            else:
                for token in tokens:
                    temp=alpha*(1/self.pos_dict[token]-len(tokens)/self.pos_total)
                    self.pos_dict[token]+=temp
                    self.pos_total+=temp
                    try:
                        for pair in self.correlationMatrix[token]:
                            synonym=pair[0]
                            temp=alpha*(1/self.pos_dict[synonym]-len(tokens)/self.pos_total)
                            self.pos_dict[synonym]+=temp
                            self.pos_total+=temp
                    except:
                        pass
                return 'negative'
        else:
            return 'neutral'
        # My thoughts on how to improve the classification.
        # Every word plays a different role in sentiment analysis. Some are more important than others.
        # For example: the word 'AI' is not so important as 'likes'
        # By putting different weights on words and train the weights using gradient descent and cross validation
        # We could get the optimal weight for each word. The predictions will become
        # sum(WEIGHT * log(number_of_times_this_word_appear))-length(text)*log(total_num_of...)

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
                    lTokens.append(sToken.lower())
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip().lower()))

        if sToken != "":
            lTokens.append(sToken.lower())

        return lTokens

def tenFoldCrossValidation(precisionRecallFileName='precisionRecallFile',trueFalseTableFileName='trueFalseTable'):
    # Initialize Precisions and Recall values
    precision=0.0
    recall=0.0

    for i in range(0,10):
        bs=Bayes_Classifier('pos_dict_test'+str(i),'neg_dict_test'+str(i),i)
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

        # Validating
        counter=0.0
        for fileName in lFileList:
            content=bs.loadFile("../movies_reviews/"+fileName)
            #result=bs.classify(content)
            star = fileName[7]
            # If bad review
            if star=='1':
                result=bs.gradientDescent('negative',content)
                if result=='positive':
                    falsePos+=1
                    #bs.analysis(result,content)
                elif result=='negative':
                    trueNeg+=1
            elif star=='5':
                result=bs.gradientDescent('positive',content)
                if result=='positive':
                    truePos+=1
                elif result=='negative':
                    falseNeg+=1
                    #bs.analysis(result,content)

            counter+=1
            if counter%30==0:
                sys.stdout.write('\r'+str(counter/len(lFileList)*100)+'% Done')
        precision+=(truePos/(truePos+falsePos)+trueNeg/(trueNeg+falseNeg))/2
        recall+=(truePos/(truePos+falseNeg)+trueNeg/(trueNeg+falsePos))/2
        with open(trueFalseTableFileName,'a') as trueFalseTableFile:
            trueFalseTableFile.write(str(truePos)+','+str(falsePos)+','+str(trueNeg)+','+str(falseNeg)+'\n')
            print (str(truePos)+','+str(falsePos)+','+str(trueNeg)+','+str(falseNeg)+'\n')
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