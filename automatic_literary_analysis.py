#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import spacy
nlp = spacy.load('en')
import nltk
from nltk.corpus import cmudict
import numpy as np
from collections import Counter
import itertools 
from nltk.stem.wordnet import WordNetLemmatizer



class Literary_analysis():
    
    def __init__(self, book, url):
        self.book = book 
        self.url = url 
        
        
    def treat_text(self):
        raw_text =  open(self.url).read()
        text=nlp(raw_text)
        return text, raw_text
    
    def aliteration(cls):
        
        text, raw_text = cls.treat_text()
        sentence = [sent for sent in text.sents]
        word_tokenised=[]
        for sent in sentence: 
            word_tokenised.append([w.lemma_ for w in sent])
            
        cmu = cmudict.dict() 
        
        #This creates a list that hyphenises the text following the cmu dictionnary           
        hyphenised_sentence_list= []
        for i in range(len(word_tokenised)):
            hyphenised_local_list = []
            for y in word_tokenised[i]:
                try:
                    word_cmu = cmu[y]
                    word_cmu = word_cmu[0]
                    hyphenised_local_list.append(word_cmu)
                except KeyError:
                    pass            
            hyphenised_sentence_list.append(hyphenised_local_list)
            
        for i in range(len(hyphenised_sentence_list)):
            sent=hyphenised_sentence_list[i]
            for j in range(len(sent)):
                w=sent[j]
                for k in range(len(w)):
                    syllable=w[k]
                    if syllable[-1].isdigit():
                        w[k]=syllable[0:(len(syllable)-1)]
                sent[j]=w
            hyphenised_sentence_list[i]=sent
        
        sentences_having_aliterations = []
            
        for i in range(len(hyphenised_sentence_list)):
                sent = hyphenised_sentence_list[i]
                sent = list(itertools.chain.from_iterable(sent))
                dic=Counter(sent)
                l=[i for i in dic.values() if i>1]
                s=np.sum(np.array(l))
                normalised_aliteration_frequency= s/float(len(hyphenised_sentence_list))
                sentences_having_aliterations.append(normalised_aliteration_frequency)
    
        
        mean_sentences_having_aliterations = np.mean(sentences_having_aliterations)
        var_sentences_having_aliterations = np.var(sentences_having_aliterations)
        cls.mean_sentences_aliteration = mean_sentences_having_aliterations
        cls.var_sentences_aliteration = var_sentences_having_aliterations
        
        
    def language_diversity(cls):
        text, raw_text = cls.treat_text()
        word_list = [word for line in raw_text for word in line.split()]
        total_word = len(word_list)
        individual_words = Counter(word_list)
        len_individual_word= len(individual_words)
        lexical_diversity = float(len_individual_word)/float(total_word)
        cls.lexical_diversity = lexical_diversity

    def size_sentence_average(cls):
        text, raw_text = cls.treat_text()
        sentence_length=[len(sent) for sent in text.sents]
        average = np.mean(sentence_length)
        variance = np.var(sentence_length)
        cls.size_sentence_mean = average
        cls.size_sentence_var = variance
        return(average)
    
    def syllable_count(cls):
        text, raw_text = cls.treat_text()

        sentence = [sent for sent in text.sents]
        #tokenised each sentence
        word_tokenised=[]
        for sent in sentence: 
            word_tokenised.append([w.lemma_ for w in sent])
            
        #REMOVE PUNCTUATION FROM THE LIST    
        punctuation = ['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']
        list_punctuation = []
        for x in  punctuation: 
            list_punctuation.append(x)
        
        list_word_no_punct = []
        for w in word_tokenised:
            for y in w: 
                if y in list_punctuation: 
                     pass 
                else:
                    list_word_no_punct.append(y)
            
            
        
        #cmu is a dictionary that matches words with a decomposition in syllable
        cmu = cmudict.dict() 
        
        #This creates a list that hyphenises the text following the cmu dictionary           
        hyphenised_sentence_list= []
        for i in range(len(word_tokenised)):
            hyphenised_local_list = []
            for y in word_tokenised[i]:
                try:
                    word_cmu = cmu[y]
                    word_cmu = word_cmu[0]
                    hyphenised_local_list.append(word_cmu)
                except KeyError:
                    pass            
            hyphenised_sentence_list.append(hyphenised_local_list)
        
        count = []  
        for i in range(len(hyphenised_sentence_list)):
            sent=hyphenised_sentence_list[i]
            word_intonation = []   
            for j in range(len(sent)):
                word=sent[j]
                
                for k in range(len(word)):
                    syllable=word[k]      
                    if syllable[-1].isdigit():
                        word_intonation.append(syllable)
            length_word_intonation= len(word_intonation)
            count.append(length_word_intonation)
        
        number_of_words = float(len(list_word_no_punct))
        number_of_syllables= np.sum(count)
        avg_syllable = float(number_of_syllables)/number_of_words
        cls.avg_syllable  = avg_syllable
        return(number_of_words,avg_syllable)
    
    
    def flesh_reading_test(cls):

        average_sent_len= cls.size_sentence_average()
        total_word_count= cls.syllable_count()[0]
        word_per_sentence = total_word_count/ average_sent_len
        frt = 206.835 - np.multiply(1.05,word_per_sentence) - np.multiply(84.6, cls.avg_syllable)
        cls.flesh =  frt



    def pos(cls):
        text = cls.treat_text()[0]
        text_length = len(text)

        verb = [w.lemma_ for w in text if w.pos_=='VERB']
        punct = [w.lemma_ for w in text if w.pos_=='PUNCT']
        symbol = [w.lemma_ for w in text if w.pos_=='SYM']
        adjective = [w.lemma_ for w in text if w.pos_=='ADJ']
        conjunction = [w.lemma_ for w in text if w.pos_=='CONJ']
        number = [w.lemma_ for w in text if w.pos_=='NUM']
        determinent = [w.lemma_ for w in text if w.pos_=='DET']
        adverb = [w.lemma_ for w in text if w.pos_=='ADV']
        foreign = [w.lemma_ for w in text if w.pos_=='X']
        preposition = [w.lemma_ for w in text if w.pos_=='ADP']
        noun = [w.lemma_ for w in text if w.pos_=='NOUN']
        proper_noun = [w.lemma_ for w in text if w.pos_=='PROPN']
        particle = [w.lemma_ for w in text if w.pos_=='PART']
        space = [w.lemma_ for w in text if w.pos_=='SPACE']
        interjection = [w.lemma_ for w in text if w.pos_=='INTJ']
        
        
        verb_ratio= float(len(verb))/float(text_length)
        cls.verb_ratio = verb_ratio
        punct_ratio= float(len(punct))/float(text_length)
        cls.punct_ratio= punct_ratio 
        symbol_ratio= float(len(symbol))/float(text_length)
        cls.symbol_ratio = symbol_ratio
        adjective_ratio= float(len(adjective))/float(text_length)   
        cls.adjective_ratio =adjective_ratio
        conjunction_ratio= float(len(conjunction))/float(text_length)
        cls.conjunction_ratio =conjunction_ratio
        number_ratio=  float(len(number))/float(text_length)
        cls.number_ratio =number_ratio
        determinent_ratio= float(len(determinent))/float(text_length)
        cls.determinent_ratio=determinent_ratio
        adverb_ratio= float(len(adverb))/float(text_length)
        cls.adverb_ratio=adverb_ratio
        foreign_ratio= float(len(foreign))/float(text_length)
        cls.foreign_ratio= foreign_ratio
        preposition_ratio= float(len(preposition))/float(text_length)
        cls.preposition_ratio=preposition_ratio
        noun_ratio= float(len(noun))/float(text_length)
        cls.noun_ratio=noun_ratio
        proper_noun_ratio= float(len(proper_noun))/float(text_length)
        cls.proper_noun_ratio=proper_noun_ratio
        particle_ratio= float(len(particle))/float(text_length)
        cls.particle_ratio= particle_ratio
        space_ratio= float(len(space))/float(text_length)
        cls.space_ratio=space_ratio
        interjection_ratio= float(len(interjection))/float(text_length)
        cls.interjection_ratio= interjection_ratio
        



                
    def compound(cls, sentence):             
        list_cc = ['and','or','for','nor','but','yet','so', ';']
        index= []
        for word in sentence: 
            if word.pos_ == 'VERB' and word.dep_ not in ('aux', 'ROOT'): 
                #print(word)
                index.append(word.i)
                    
        for i in index:
            window = 6
            for y in range(i-window, i):
                if y > 0:
                #print(sent_1[y])
                    try: 
                        if sentence[y].lemma_ in list_cc:
                            return(True)
                    except IndexError:
                        pass
                  
            for y in range(i, i+window):
                if y < len(sentence):
                    try: 
                        if sentence[y].lemma_ in list_cc:
                            return(True)
                    except:
                        pass
    
    
    
    def complexe(cls, sentence):
        list_ad=['acl','relcl','advcl','ccomp','xcomp','csubj']
        for word in sentence: 
            if word.pos_ == 'VERB' and word.dep_ in list_ad: 
                return(True)
    
    
    
    def complexe_compound(cls, sentence):
        if cls.complexe(sentence) == True and cls.compound(sentence)==True:
            return(True)
    
    
    def simple(cls, sentence):
        if cls.complexe(sentence) != True and cls.compound(sentence)!= True:
            count = 0 
            for word in sentence: 
                if word.pos_ == "VERB":
    
                    count+= 1
                if count >= 1: 
                    return(True)  
                    
                    
    def ratio_clause(cls):
        text = cls.treat_text()[0]
        sentences = [sent for sent in text.sents]
        sent_length = float(len(sentences))
        simple_clause = 0
        complex_compound_clause = 0
        complex_clause = 0 
        compound_clause = 0
        for sentence in sentences: 
            if cls.simple(sentence) == True: 
                simple_clause += 1
            if cls.complexe_compound(sentence)==True: 
                complex_compound_clause += 1
            if cls.complexe(sentence) == True: 
                complex_clause += 1
            if cls.compound(sentence) == True: 
                compound_clause += 1 
            
            
        cls.simple_clause_ratio = float(simple_clause)/sent_length
        cls.complex_compound_clause_ratio = float(complex_compound_clause)/sent_length
        cls.complex_clause_ratio = float(complex_clause)/sent_length
        cls.compound_clause_ratio = float(compound_clause)/sent_length
        
        
        
        
    def un_and_usual_words(cls):
        text = cls.treat_text()[0]
        lmtzr = WordNetLemmatizer()
        text_vocab = set(w.lemma_ for w in text if w.is_alpha)
        english_vocab = set([lmtzr.lemmatize(w) for w in nltk.corpus.words.words()])
        unusual = text_vocab - english_vocab#does the intersection
        unusual = sorted(unusual)
        set_usual = sorted (text_vocab)
        frac_unusual= len(unusual)/float(len(set_usual))
        cls.frac_unusual = frac_unusual





    def gradable(cls):
        text = cls.treat_text()[0]
        list_adj = [(w.i) for w in text if w.pos_ == 'ADJ']
        list_vereally=[(w.i) for w in text if w.lemma_ in ('very', 'really')]    	    
        counter_vereally=0
        for i in range(len(list_vereally)):
            post_window=1#
            l_post=[w for w in text if w.i>=list_vereally[i] and w.i<=list_vereally[i]+post_window]
            l_post_grammar=[w.pos_ for w in l_post]
    
            if 'ADJ' in l_post_grammar:
                counter_vereally+=1    	    
        fraction_gradable = counter_vereally/float(len( list_adj))

        cls.fraction_gradable = fraction_gradable




    def tunits(cls): 
        text = cls.treat_text()[0]
        count=1
	    
        for word in text: 
            if word.dep_ == 'conj' or word.text == ";":  
                count += 1
                
            if word.text ==';':
                count +=1
	            
	    
        sentence_length=[len(sent) for sent in text.sents]
        sum_words = sum(sentence_length)
        t_score= float(sum_words)/float(count) 
        cls.t_score = t_score

	        
    def repetition(cls, search_window = 20):
    
        text = cls.treat_text()[0]
        list_pos_of_importance=['ADJ','NOUN']#['ADV','ADJ','ADP','NOUN']
        
        n_text=len(text)
        i_down=0;i_up=min(n_text-1,search_window)
	    
        l_current=[w.lemma_ for w in text[i_down:i_up] if w.pos_ in list_pos_of_importance and len(w)>3]
        count_initial=Counter(l_current)
        list_word_rep_ini=[w for w in count_initial.keys() if count_initial.get(w)>1]
        list_word_rep=[w.lemma_ for w in text[i_down:i_up] if w.lemma_ in list_word_rep_ini]

        while i_down< i_up:
            l_current=[w.lemma_ for w in text[i_down:i_up] if w.pos_ in list_pos_of_importance and len(w)>3]
            if i_up<len(text)-1:
                w_new=text[i_up+1]
                if w_new.pos_ in list_pos_of_importance and len(w_new)>3:
                    l_occ=[w for w in l_current if w==w_new.lemma_]
                    if len(l_occ)!=0:
                        list_word_rep.append(w_new.lemma_)
                i_up+=1;i_down+=1
            else:
                i_down=len(text)-1
               
        repetition_res=Counter(list_word_rep)
        frac_rep=sum(repetition_res.values())/float(len([w for w in text if w.pos_ in list_pos_of_importance]))
	    
        cls.frac_rep = frac_rep

    def passive_markers(cls):
        text = cls.treat_text()[0]
        passive_dep = ['csubjpass','auxpass','nsubjpass']
        count_sent = 0 
        list_sentence=[sent for sent in text.sents]
        for sent in list_sentence:
            passive = False
            for word in sent:
                if passive is False :
                    if word.dep_ in passive_dep:
                        passive = True
                        count_sent += 1
                        
        frac_pass=count_sent/float(len(list_sentence))
        cls.passive_voice = frac_pass
               
             
    def oneforall(cls):
        cls.repetition()
        cls.tunits()
        cls.gradable()
        cls.pos()
        cls.un_and_usual_words()
        cls.ratio_clause()
        cls.language_diversity()
        cls.flesh_reading_test()
        cls.size_sentence_average()
        cls.aliteration()
        cls.passive_markers()
        
        
        
        
        
        
        
    