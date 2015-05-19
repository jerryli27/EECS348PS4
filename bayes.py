# Name: Alan Fu, Jiaming Li
# Date: 05/18/2015
# Description: The basic bayes net sentiment analysis
#
#



import math, os, pickle, re, random,sys,collections

class Bayes_Classifier:

    def __init__(self,pos_dict_file_name='pos_dict',neg_dict_file_name='neg_dict',k=-1):
        # Used for the Kth Cross Validation. K ranges from 0 to 9

        """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text."""
        self.pos_dict=collections.defaultdict(int)
        self.neg_dict=collections.defaultdict(int)
        self.pos_total=0
        self.neg_total=0
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

        # Validating
        counter=0.0
        for fileName in lFileList:
            content=bs.loadFile("../movies_reviews/"+fileName)
            result=bs.classify(content)
            star = fileName[7]
            # If bad review
            if star=='1':
                #result=bs.gradientDescent('negative',content)
                if result=='positive':
                    falsePos+=1
                    #bs.analysis(result,content)
                elif result=='negative':
                    trueNeg+=1
            elif star=='5':
                #result=bs.gradientDescent('positive',content)
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

#NOTE: to use this, put the movies_reviews folder at the previous folder. It's a huge pain for github to load 20000+ files.
tenFoldCrossValidation()
#bs=Bayes_Classifier()
#print bs.classify('I love my AI class')

#I suspect that because the positive reviews have way more words (lower -log(total) value) than negative reviews,the system is biased towards negative