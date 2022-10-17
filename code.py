
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import ChannelAccount
import nltk
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

import nltk
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
 
ps = PorterStemmer()
show_details=True 
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
from cosine_ import *

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')

class EchoBot(ActivityHandler):
    async def on_members_added_activity(
        self, members_added: [ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome!")

    async def on_message_activity(self, turn_context: TurnContext):
        intents = json.loads(open('intents.json').read())
        words = pickle.load(open('words.pkl','rb'))
        classes = pickle.load(open('classes.pkl','rb'))
        msg=turn_context.activity.text
        final_answer=[]
        msg1=msg
        
        from rake_nltk import Rake
        rake_nltk_var = Rake()
        text = msg1
        rake_nltk_var.extract_keywords_from_text(text)
        sentence_words = rake_nltk_var.get_ranked_phrases()
        str1 = " "
        x=str1.join(sentence_words)
        li = list(x.split(" "))
    
        keywords=[]
        tags=[]
        res_=[]
        len_keywords=li
        #START HERE AND CHECK IF CONSIDERING THIS EVEN IS THERE IS NO VALUE '' AND IF THERE ARE KEY WORDS
        #ADD FILTER HERE
        try:
            len_keywords.remove('')
        except:
            ValueError()
        print("after",len_keywords)
        print("",len(li))
        #Determining the length of keywords to predict the tag
        cos_loop_1=[]
        if  len(len_keywords)!=0 and len(len_keywords)==1: # tricky part detailed explanation is as follows,  the return function of a keyword is a list and  even is a list is empty the value of the empty list is 1 check this p=[''] len(p) so it is madatory to have a if loop where the list returns back value such that the
        #the value that is returned is not len key_words not 0
            #print(key_words(msg1))
            
            
            #finding the keywords if no key words found it finds reponse without the keywords #jumps to else loop
            for i in range(len(li)):
                    msg2=li[i]
                    print("message is key words found",msg2)
                    keywords.append(msg2)
                                
                    sentence_words = nltk.word_tokenize(msg2)
                    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
                    
                    # bag of words - matrix of N words, vocabulary matrix
                    bag1 = [0]*len(words)
                    #print("bow",bag)
                    for s in sentence_words:
                        #print("s",s)
                        for i,w in enumerate(words):
                            #print("w is",w)
                            if w == s:
                                # assign 1 if current word is in the vocabulary position
                                bag1[i] = 1
                                if show_details:
                                    #prstint ("found in bag:" % w)
                                    print("")
                    #print("bow result is",np.array(bag))
                    p1=np.array(bag1)
                    print("p is",p1)
                    res = model.predict(np.array([p1]))[0]
                    print("predict_class",res)
                    ERROR_THRESHOLD = 0.2
                    #print("ERROR THRESHOLD IS ", ERROR_THRESHOLD)
                    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
                    #print("ERROR THRESHOLD IS after", ERROR_THRESHOLD)
                    #print("Results",results)
                    # sort by strength of probability
                    results.sort(key=lambda x: x[1], reverse=True)
                    return_list = []
                    for r in results:
                        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
                    #print("predict class return list",return_list
                    ints=return_list
                    tags.append(ints)
                    tag = ints[0]['intent']
                    print("tag is",tag)
                    list_of_intents = intents['intents']
                    #print("list of intents", list_of_intents)
                    for i in list_of_intents:
                        #print(tag)
                        if(i['tag']== tag):
                            #print(i['patterns'])
                            #print(i['patterns'][0])
                            cosine_=[]
                            pat_=[]
                            result_=[]
                            print('hghgnnh',len(i['patterns']))
                            for u in range (len(i['patterns'])):
                                text1=msg
                                text2=i['patterns'][u]
                                value=cosine(text1,text2)
                                cosine_.append(value)
                                pat_.append(u)
                                result1=sorted(zip(cosine_,pat_), reverse=True)[:2]
                                print('gfnhgmjhmjdnhgnhjbvcdfghjkl')
                            print(result1,i['patterns'][result1[0][1]],i['responses'][result1[0][1]],i['patterns'][result1[1][ 1]],i['responses'][result1[1][1]])
                            #if loop to be added and test this example "can you smell"
                            result = i['patterns'][result1[0][1]],i['responses'][result1[0][1]]
                            break
                        else:
                            result = "You must ask the right questions"
                    res=result


                    res_.append(res)
                    print(res[0])
                    cos=cosine(msg1, res[0])
                    cos_loop_1.append(cos)
                
                    if((cos)!=0):
                        print("LOOP ONE IF",cos)
                        print("keywords are",res[1])
                        bot_reply=res[1]
                        final_answer.append(bot_reply)
                    else:
                        print("LOOP ONE ELSE")
                        sentence_words = nltk.word_tokenize(msg1)
                        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

                        # bag of words - matrix of N words, vocabulary matrix
                        bag2 = [0]*len(words)
                        #print("bow",bag)
                        for s in sentence_words:
                            #print("s",s)
                            for i,w in enumerate(words):
                                #print("w is",w)
                                if w == s:
                                    # assign 1 if current word is in the vocabulary position
                                    bag2[i] = 1
                                    if show_details:
                                        #prstint ("found in bag: " % w)
                                        print("")
                        #print("bow result is",np.array(bag))
                        p2=np.array(bag2)
                        print("p is",p2)
                        res = model.predict(np.array([p2]))[0]
                        print("predict_class",res)
                        ERROR_THRESHOLD = 0.2
                        #print("ERROR THRESHOLD IS ", ERROR_THRESHOLD)
                        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
                        #print("ERROR THRESHOLD IS after", ERROR_THRESHOLD)
                        #print("Results",results)
                        # sort by strength of probability
                        results.sort(key=lambda x: x[1], reverse=True)
                        return_list = []
                        for r in results:
                            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
                        #print("predict class return list",return_list
                        ints=return_list
                        tag = ints[0]['intent']
                        print("tag is",tag)
                        list_of_intents = intents['intents']
                        #print("list of intents", list_of_intents)
                        for i in list_of_intents:
                            #print(tag)
                            if(i['tag']== tag):
                                #print(i['patterns'])
                                #print(i['patterns'][0])
                                cosine_=[]
                                pat_=[]
                                result_=[]
                                print('hghgnnh',len(i['patterns']))
                                for u in range (len(i['patterns'])):
                                    text1=msg
                                    text2=i['patterns'][u]
                                    value=cosine(text1,text2)
                                    cosine_.append(value)
                                    pat_.append(u)
                                    result1=sorted(zip(cosine_,pat_), reverse=True)[:2]
                                    print('gfnhgmjhmjdnhgnhjbvcdfghjkl')
                                print(result1,i['patterns'][result1[0][1]],i['responses'][result1[0][1]],i['patterns'][result1[1][ 1]],i['responses'][result1[1][1]])
                                #if loop to be added and test this example "can you smell"
                                result = i['patterns'][result1[0][1]],i['responses'][result1[0][1]]
                                break
                            else:
                                result = "You must ask the right questions"
                        res=result


                        print("keywords count 0",res[1])
                        bot_reply=res[1]
                        final_answer.append(bot_reply)
                        
        elif len(len_keywords)!=0 and len(len_keywords)>1:
            #print(key_words(msg1))
            print("YES")
            
            #finding the keywords if no key words found it finds reponse without the keywords #jumps to else loop
                
            try:
                from rake_nltk import Rake
                rake_nltk_var = Rake()
                text = msg1
                rake_nltk_var.extract_keywords_from_text(text)
                sentence_words = rake_nltk_var.get_ranked_phrases()
                str1 = " "
                x=str1.join(sentence_words)
                li = list(x.split(" "))
            except:
                IndexError()
            for i in range(len(li)):
                    msg2=li[i]
                    print("message is key words found",msg2)
                    keywords.append(msg2)
                    sentence_words = nltk.word_tokenize(msg2)
                    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

                    # bag of words - matrix of N words, vocabulary matrix
                    bag3 = [0]*len(words)
                    #print("bow",bag)
                    for s in sentence_words:
                        #print("s",s)
                        for i,w in enumerate(words):
                            #print("w is",w)
                            if w == s:
                                # assign 1 if current word is in the vocabulary position
                                bag3[i] = 1
                                if show_details:
                                    #prstint ("found in bag: %s" % w)
                                    print("")
                    #print("bow result is",np.array(bag))
                    p3=np.array(bag3)
                    print("p is",p3)
                    res = model.predict(np.array([p3]))[0]
                    print("predict_class",res)
                    ERROR_THRESHOLD = 0.2
                    #print("ERROR THRESHOLD IS ", ERROR_THRESHOLD)
                    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
                    #print("ERROR THRESHOLD IS after", ERROR_THRESHOLD)
                    #print("Results",results)
                    # sort by strength of probability
                    results.sort(key=lambda x: x[1], reverse=True)
                    return_list = []
                    for r in results:
                        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
                    #print("predict class return list",return_list
                    ints=return_list
                    tags.append(ints)
                    tag = ints[0]['intent']
                    print("tag is",tag)
                    list_of_intents = intents['intents']
                    #print("list of intents", list_of_intents)
                    for i in list_of_intents:
                        #print(tag)
                        if(i['tag']== tag):
                            #print(i['patterns'])
                            #print(i['patterns'][0])
                            cosine_=[]
                            pat_=[]
                            result_=[]
                            print('hghgnnh',len(i['patterns']))
                            for u in range (len(i['patterns'])):
                                text1=msg
                                text2=i['patterns'][u]
                                value=cosine(text1,text2)
                                cosine_.append(value)
                                pat_.append(u)
                                result1=sorted(zip(cosine_,pat_), reverse=True)[:2]
                                print('gfnhgmjhmjdnhgnhjbvcdfghjkl')
                            print(result1,i['patterns'][result1[0][1]],i['responses'][result1[0][1]],i['patterns'][result1[1][ 1]],i['responses'][result1[1][1]])
                            #if loop to be added and test this example "can you smell"
                            result = i['patterns'][result1[0][1]],i['responses'][result1[0][1]]
                            break
                        else:
                            result = "You must ask the right questions"
                    res=result,result1[0]
                    res=res[0]

                    res_.append(res)
                    print("keywords are",res_)
                    print(len(keywords))
                    if(len(len_keywords)>=1):
                        cosine_list=[]
                        index=[]
                        for i in range(len(keywords)):
                            print(msg1)
                            print(res_[i][0])
                            letters_only = re.sub("[^a-zA-Z]", " ", str(res_[i][0]))
                            cosine_score=cosine(msg1, letters_only)
                            print("For this keyword the score is",res_[i],cosine_score)
                            cosine_list.append([cosine_score,i])
                        cosine_list.sort()
                        print(cosine_list)
                        highestcosine=cosine_list[-1]
                        #print(highestcosine)
                        highestcosinescore=highestcosine[0]
                        print("f;ksd",highestcosinescore)
                        highestindex=highestcosine[1]
                        print(highestindex)
                        print("The similar sentence is",res_[highestindex])
                        sentence_w = nltk.word_tokenize(res_[highestindex][0])
                        sentence_w = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

                        words=sentence_w
                        print("after clean up sentence the words are",words,keywords)
                        
                        compare=list(set(words).intersection(keywords))
                        print(type(compare))
                                        
                        sentence_words = nltk.word_tokenize(msg2)
                        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
                        
                        #  bag of words - matrix of N words, vocabulary matrix
                        bag4 = [0]*len(words)
                        #print("bow",bag)
                        for s in sentence_words:
                            #print("s ",s)
                            for i,w in enumerate(words):
                                #print("w is",w)
                                if w == s:
                                    # assign 1 if current word is in the vocabulary position
                                    bag4[i] = 1
                                    if show_details:
                                        #prstint ("found in bag: " % w)
                                        print("")
                        #print("bow result is",np.array(bag))
                        p4=np.array(bag4)
                        print("p is",p4)
                        res = model.predict(np.array([p4]))[0]
                        print("predict_class",res)
                        ERROR_THRESHOLD = 0.2
                        #print("ERROR THRESHOLD IS ", ERROR_THRESHOLD)
                        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
                        #print("ERROR THRESHOLD IS after", ERROR_THRESHOLD)
                        #print("Results",results)
                        # sort by strength of probability
                        results.sort(key=lambda x: x[1], reverse=True)
                        return_list = []
                        for r in results:
                            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
                        #print("predict class return list",return_list
                        ints_compare=return_list
                        
                        print(compare,ints_compare,ints)
                        if(ints_compare[0]['intent']==ints[0]['intent']):
                            print("This sentence is a best match",res_[highestindex][0])
                            result= res_[highestindex][1]
                            print("The result ",result,len(result))
                            bot_reply=result
                            final_answer.append(bot_reply)
                        #Last resort to the answers to be provided to the users
                        else:
                            sentence_words = nltk.word_tokenize(msg1)
                            sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

                            # bag of words - matrix of N words, vocabulary matrix
                            bag5 = [0]*len(words)
                            #print("bow",bag)
                            for s in sentence_words:
                                #print("s",s)
                                for i,w in enumerate(words):
                                    #print("w is",w)
                                    if w == s:
                                        # assign 1 if current word is in the vocabulary position
                                        bag5[i] = 1
                                        if show_details:
                                            #prstint ("found in bag: %s" % w)
                                            print("")
                            #print("bow result is",np.array(bag))
                            p5=np.array(bag5)
                            print("p is",p5)
                            res = model.predict(np.array([p5]))[0]
                            print("predict_class",res)
                            ERROR_THRESHOLD = 0.2
                            #print("ERROR THRESHOLD IS ", ERROR_THRESHOLD)
                            results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
                            #print("ERROR THRESHOLD IS after", ERROR_THRESHOLD)
                            #print("Results",results)
                            # sort by strength of probability
                            results.sort(key=lambda x: x[1], reverse=True)
                            return_list = []
                            for r in results:
                                return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
                            #print("predict class return list",return_list
                            ints1=return_list
                            tag = ints[0]['intent']
                            print("tag is",tag)
                            list_of_intents = intents_json['intents']
                            #print("list of intents", list_of_intents)
                            for i in list_of_intents:
                                #print(tag)
                                if(i['tag']== tag):
                                    #print(i['patterns'])
                                    #print(i['patterns'][0])
                                    cosine_=[]
                                    pat_=[]
                                    result_=[]
                                    print('hghgnnh',len(i['patterns']))
                                    for u in range (len(i['patterns'])):
                                        text1=msg
                                        text2=i['patterns'][u]
                                        value=cosine(text1,text2)
                                        cosine_.append(value)
                                        pat_.append(u)
                                        result1=sorted(zip(cosine_,pat_), reverse=True)[:2]
                                        
                                    print(result1,i['patterns'][result1[0][1]],i['responses'][result1[0][1]],i['patterns'][result1[1][ 1]],i['responses'][result1[1][1]])
                                    #if loop to be added and test this example "can you smell"
                                    result = i['patterns'][result1[0][1]],i['responses'][result1[0][1]]
                                    break
                                else:
                                    result = "You must ask the right questions"
                            res1=result,result1[0]
                            res1=res1[0]


                            print(ints,ints1)
                            bot_reply=res1[1]
                            if(ints1[0]['intent']==ints[0]['intent']):
                                    bot_reply="Please ask the right questions, I think you are asking somthing related to topic",ints[0]['intent'].split(",")[0:4],"if this is your question",res1[0],"then my answer is",res1[1],"Please ask buisiness related queries and if I am wrong please contact Aishwarya","\N{grinning face}"
                                    final_answer.append(bot_reply)
        else:
            
            sentence_words = nltk.word_tokenize(msg1)
            sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

            # bag of words - matrix of N words, vocabulary matrix
            bag6 = [0]*len(words)
            #print("bow",bag)
            for s in sentence_words:
                #print("s",s)
                for i,w in enumerate(words):
                    #print("w is",w)
                    if w == s:
                        # assign 1 if current word is in the vocabulary position
                        bag6[i] = 1
                        if show_details:
                            #prstint ("found in bag: %s" % w)
                            print("")
            #print("bow result is",np.array(bag))
            p6=np.array(bag6)
            print("p is",p6)
            res = model.predict(np.array([p6]))[0]
            print("predict_class",res)
            ERROR_THRESHOLD = 0.2
            #print("ERROR THRESHOLD IS ", ERROR_THRESHOLD)
            results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
            #print("ERROR THRESHOLD IS after", ERROR_THRESHOLD)
            #print("Results",results)
            # sort by strength of probability
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = []
            for r in results:
                return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
            #print("predict class return list",return_list
            ints=return_list
            tag = ints[0]['intent']
            print("tag is",tag)
            list_of_intents = intents['intents']
            #print("list of intents", list_of_intents)
            for i in list_of_intents:
                #print(tag)
                if(i['tag']== tag):
                    #print(i['patterns'])
                    #print(i['patterns'][0])
                    cosine_=[]
                    pat_=[]
                    result_=[]
                    print('hghgnnh',len(i['patterns']))
                    for u in range (len(i['patterns'])):
                        text1=msg
                        text2=i['patterns'][u]
                        value=cosine(text1,text2)
                        cosine_.append(value)
                        pat_.append(u)
                        result1=sorted(zip(cosine_,pat_), reverse=True)[:2]
                        
                    print(result1,i['patterns'][result1[0][1]],i['responses'][result1[0][1]],i['patterns'][result1[1][ 1]],i['responses'][result1[1][1]])
                    #if loop to be added and test this example "can you smell"
                    result = i['patterns'][result1[0][1]],i['responses'][result1[0][1]]
                    print(result)
                    break
                else:
                    result = "You must ask the right questions"
            res=result[1]


            print("keywords count 0",res)
            bot_reply=res
            final_answer.append(bot_reply)
            
        

        
        return await turn_context.send_activity(
            MessageFactory.text(f"{final_answer[0]}")
        )
