# Name: Alan Fu, Jiaming Li
# Date: 05/18/2015
# Description: The basic bayes net sentiment analysis
#
#



import math, os, pickle, re, random,sys,collections

class Bayes_Classifier:

    def __init__(self,pos_dict_file_name='pos_dict_best',neg_dict_file_name='neg_dict_best',k=-1):
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


        # Training
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
        # My thoughts on how to improve the classification.
        # Every word plays a different role in sentiment analysis. Some are more important than others.
        # For example: the word 'AI' is not so important as 'likes'
        # By putting different weights on words and train the weights using gradient descent and cross validation
        # We could get the optimal weight for each word. The predictions will become
        # sum(WEIGHT * log(number_of_times_this_word_appear))-length(text)*log(total_num_of...)
    # This function will print out the log of each word if we get the answer wrong
    def analysis(self,result, sText):
        """Given a target string sText, this function returns the most likely document
        class to which the target string belongs (i.e., positive, negative or neutral).
        """
        tokens=self.tokenize(sText)
        # The log of likelihood of the text being positive is just log of all frequencies,
        # which is log(number_of_times_this_word_appear/total_number_of_words_in_positive).
        # We could simplify this to sum(log(number_of_times_this_word_appear))-length(text)*log(total_num_of...)

        pos_pred=-math.log(self.pos_total)*len(tokens)
        neg_pred=-math.log(self.neg_total)*len(tokens)
        # To view the individual term's contribution, we have to calculate
        # log(number_of_times_this_word_appear)- log(total_number_of_words_in_positive).
        pos_total_log=math.log(self.pos_total)
        neg_total_log=math.log(self.neg_total)
        # This list records the log of each token
        pos_log_list=[]
        neg_log_list=[]
        for token in tokens:
            if token in self.pos_dict.keys():
                temp=math.log(self.pos_dict[token])
                pos_pred+=temp
                # append log(number_of_times_this_word_appear)- log(total_number_of_words_in_positive).
                pos_log_list.append(math.ceil((temp-pos_total_log)*100)/100)
            else:
                # append - log(total_number_of_words_in_positive).
                # This is not mathematically correct, because we are assuming log(number_of_times_this_word_appear)=0
                # Which is somewhat equal to the effect of smoothing, taking number_of_times_this_word_appear=1
                pos_log_list.append(math.ceil((-pos_total_log)*100)/100)
            if token in self.neg_dict.keys():
                temp=math.log(self.neg_dict[token])
                neg_pred+=temp
                neg_log_list.append(math.ceil((temp-neg_total_log)*100)/100)
            else:
                neg_log_list.append(math.ceil((-neg_total_log)*100)/100)
            #else is just *=1, which does nothing
        #Need to code a Neutral zone
        print 'pos_pred = '+str(pos_pred)
        print 'neg_pred = '+str(neg_pred)

        if result=='negative':
            sys.stdout.write('False Negative:        ')
        else:
            sys.stdout.write('False Positive:        ')
        for token in tokens:
            tempString='%.12s' %token
            tempString+=' '*(12-len(tempString))
            sys.stdout.write(tempString)
        sys.stdout.write('\n')

        tempString='Pos logs:    '
        tempString+=' '*(23-len(tempString))
        sys.stdout.write(tempString)
        counter=0
        for token in tokens:
            strFormat='%.2f'%pos_log_list[counter]
            strFormat+=' '*(12-len(strFormat))
            sys.stdout.write(strFormat)
            counter+=1
        sys.stdout.write('\n')

        tempString='Neg logs:    '
        tempString+=' '*(23-len(tempString))
        sys.stdout.write(tempString)
        counter=0
        for token in tokens:
            strFormat='%.2f'%neg_log_list[counter]
            strFormat+=' '*(12-len(strFormat))
            sys.stdout.write(strFormat)
            counter+=1
        sys.stdout.write('\n')

        return

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
            if result=='positive':
                return 'positive'
            else:
                # It should be negative but classified as positive. According to gradient descent, the delta value should be:
                for token in tokens:
                    temp=alpha*(1/self.neg_dict[token]-len(tokens)/self.neg_total)
                    self.neg_dict[token]+=temp
                    self.neg_total+=temp
                return 'positive'
        elif pos_pred<neg_pred :
            if result=='negative':
                return 'negative'
            else:
                for token in tokens:
                    temp=alpha*(1/self.pos_dict[token]-len(tokens)/self.pos_total)
                    self.pos_dict[token]+=temp
                    self.pos_total+=temp
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
        bs=Bayes_Classifier('pos_dict_best'+str(i),'neg_dict_best'+str(i),i)
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

#NOTE: to use this, put the movies_reviews folder at the previous folder. It's a huge pain for github to load 20000+ files.
tenFoldCrossValidation()
#bs=Bayes_Classifier()
#print bs.classify('I love my AI class')

# I suspect that because the positive reviews have way more words (lower -log(total) value) than negative reviews,the system is biased towards negative