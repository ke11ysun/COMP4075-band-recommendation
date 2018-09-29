from random import *
import fnmatch
import os
import re
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir, path
import time
import numpy
from shutil import copyfile
import shutil
import math
import csv
from heapq import nlargest
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

import audioBasicIO as aio
import audioFeatureExtraction as afe
import scipy.spatial.distance as dist

def parseINT(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
def preprocess_text(text, stemming):
    text=text.replace('\n',' ').lower()
    tknzr = TweetTokenizer()
    lemmatizer = WordNetLemmatizer()
    enStopW=stopwords.words('english')
    text = re.sub(r"http\S+","",text)
    if stemming:
        text=' '.join(lemmatizer.lemmatize(token).encode('utf-8') for token in tknzr.tokenize(text) if token not in string.punctuation and token not in enStopW and not token.isdigit())
    else:
        # print type(text)
        text=' '.join(token for token in tknzr.tokenize(text) if token not in string.punctuation and token not in enStopW and not token.isdigit())
    text = re.sub("[^A-Za-z\s]", "" ,text)
    text=' '.join(token.encode('utf-8') for token in text.split(' ') if len(token)>2)
    # print text+'\n\n'
    return text    
    
def term_freq(word_list): # words in a document
    word_dict = {}
    for w in word_list:
        if w in word_dict:
            word_dict[w] += 1
        else:
            word_dict[w] = 1
        # calculate #words in total and then calculate the frequency
    word_num = float(sum(word_dict.values()))
    for w in word_dict.keys():
        word_dict[w] /= word_num
    return word_dict

def inv_doc_freq(term_set, doc_name2word_list):
    doc_num = len(doc_name2word_list)
    idf_dict = {}
    # term in all doc
    for w in term_set:
        doc_count = 0
        #find the appear frenquency among all documents
        for word_list in doc_name2word_list:
            if w in word_list:
                doc_count += 1
        if doc_count != 0:
            idf_dict[w] = math.log(float(doc_num)/float(doc_count))
        else:
            idf_dict[w] = math.log(float(doc_num)/float(100))       #wrong preprocessing -> ignore

    return idf_dict
    

def main(): 

    start_time = time.time()
    # all bands are grouped by style
    style_list = [['arch_enemy', 'at_the_gates', 'children_of_bodom', 'wintersun', 'in_flames', 'fleshgod_apocalypse', 'opeth', 'katatonia', 'insomnium', 'swallow_the_sun', 'ghost', 'enslaved', 'dark_tranquillity', 'dark_lunacy', 'trivium', 'paradise_lost', 'behemoth'
                   ], ['while_she_sleeps', 'underoath', 'motionless_in_white', 'memphis_may_fire', 'killswitch_engage', 'caliban', 'bullet_for_my_valentine', 'bring_me_the_horizon', 'blessthefall', 'black_veil_brides', 'august_burns_red'
                   ], ['whitechapel', 'veil_of_maya', 'thy_art_is_murder', 'the_black_dahlia_murder', 'suicide_silence', 'rings_of_saturn', 'job_for_a_cowboy', 'infant_annhilator', 'decapitated', 'carnifex'
                   ], ['anthrax', 'iron_maiden', 'judas_priest', 'megadeth', 'metallica', 'sepultura', 'slayer', 'death', 'testament'
                   ], ['leprous', 'ihsahn', 'between_the_buried_and_me', 'baroness', 'gojira', 'code_orange', 'symphony_x', 'dream_theater'
                   ], ['acid_bath', 'isis', 'neurosis', 'mastodon', 'today_is_the_day', 'torche', 'cult_of_luna'
                   ], ['the_faceless', 'protest_the_hero', 'meshuggah', 'born_of_osiris', 'periphery'
                   ], ['xandria', 'within_temptation', 'epica', 'rhapsody_of_fire', 'haggard', 'eluveitie'
                   ], ['amon_amarth', 'amorphis', 'blind_guardian', 'eluveitie', 'finntroll', 'enslaved'
                   ], ['nile', 'napalm_death', 'deicide', 'carcass', 'cannibal_corpse'
                   ], ['venom', 'mayhem', 'immortal', 'darkthrone', 'burzum']]

    # randomly pick 5 bands from a random style
    random_style = int(uniform(0,11))
    random_bands = sample(style_list[random_style], k=5)
    print 'bands picked'
    print random_bands
    print '\n'
    
    all_songs_from_file = []
    for each_band in random_bands:      # get all songs from selected band
        # path = "C:\Users\Nathan\Desktop\WI project all data\COMP4075_PROJECT\project_songs\songs_" + each_band
        path = "/Users/sunjingxuan/Desktop/WI_project_all_data/COMP4075_PROJECT/project_songs/songs_" + each_band
        all_songs_from_file.append([f for f in os.listdir(path) if fnmatch.fnmatch(f, '*.mp3')])

    all_songs = []
    for each_song_list in all_songs_from_file:
        for each_song in each_song_list:
            all_songs.append(each_song)
    
    # preprocessing for song names -> different format
    processed_all_song_name = []
    for each_song_name in all_songs:
        tmp_song_name = ''
        inBracket = False
        inSquare = False
        for index, each_word in enumerate(each_song_name):
            if each_word == '(':        #for removing (xxx)
                inBracket = True
            elif each_word == ')':
                inBracket = False
            if each_word == '[':        #for removing (xxx)
                inSquare = True
            elif each_word == ']':
                inSquare = False
                
            if inBracket == False and inSquare == False:
                if each_word == '.':    #if the end of the name-> break
                    break  
                if each_word is not '-' and each_word is not ')' and \
                each_word is not ']':
                    tmp_song_name += each_word
                elif each_word == '-':
                    tmp_song_name = ''    #remove band name
        processed_all_song_name.append(tmp_song_name)
    for idx, each_name in enumerate(processed_all_song_name):
        if each_name != '':
            if each_name[0] == ' ':     #remove the starting " "
                tmp = ''
                for index, each_char in enumerate(each_name):
                    if index != 0:
                        tmp += each_char
                processed_all_song_name[idx] = tmp
    
    for idx, each_name in enumerate(processed_all_song_name):
        if each_name != '':
            if each_name[len(each_name)-1] == ' ':     #remove the ending " "
                tmp = ''
                for index, each_char in enumerate(each_name):
                    if index != len(each_name)-1:
                        tmp += each_char
                processed_all_song_name[idx] = tmp

    # print 'processed all song names'
    # print len(processed_all_song_name)
    # print '\n'
    ##########################################finish preprocessing###########################################
    
    each_song = []        #contains all candidate songs
    all_lyrics = ''
    songs_have_lyrics = []
    # load the lyrics file
    for each_band in random_bands:
        each_file = 'lyrics_' + each_band + '.txt'
        
        # based on the mp3 list, pick those songs that its lyrics can be found
        with open(each_file, 'r') as each_lyrics_file:
            lyrics = each_lyrics_file.readlines()
            for index_for_items, data_items in enumerate(lyrics):       #for each line
                for index, each_character in enumerate(data_items):     #for each word
                    if parseINT(each_character):        #if the first word is an int-> song
                        if data_items[index+1] == '.':  
                            sep = data_items.split()
                            each_song_name = ''
                            for idx, each_part in enumerate(sep):
                                if idx is not 0 and idx is not len(sep) -1:
                                    each_song_name += each_part+' '     #get song names
                                elif idx is len(sep) -1:
                                    each_song_name += each_part
                            each_song.append(each_song_name)
                            for processed_song in processed_all_song_name:
                                if processed_song in each_song_name and processed_song not in songs_have_lyrics:
                                    songs_have_lyrics.append(processed_song)
                                    break
                                
        each_lyrics_file.close()
        
    # print 'songs that have lyrics'
    # print len(songs_have_lyrics)
    # print '\n'
    
    selected_playlist = sample(songs_have_lyrics, k=10)
    print 'selected playlist'
    print selected_playlist
    print '\n'
    #####################################finish picking songs########################################## 
    songs_picked = ''
    
    for each_band in random_bands:
        each_file = 'lyrics_' + each_band + '.txt'
        
        with open(each_file, 'r') as each_lyrics_file:
            lyrics = each_lyrics_file.readlines()
            start_copy = False;
            for index_for_items, data_items in enumerate(lyrics):       #for each line
                for index, each_character in enumerate(data_items):     #for each word
                    if parseINT(each_character):        #if the first word is an int-> song
                        if data_items[index+1] == '.':  
                            sep = data_items.split()
                            each_song_name = ''
                            for idx, each_part in enumerate(sep):
                                if idx is not 0 and idx is not len(sep) -1:
                                    each_song_name += each_part+' '     #get song names
                                elif idx is len(sep) -1:
                                    each_song_name += each_part
                                    
                            for selected_song in selected_playlist:
                                if selected_song in each_song_name and selected_song not in songs_picked:
                                    songs_picked += (selected_song + " ")
                                    start_copy = True
                                    break
                                else:
                                    start_copy = False            
                if start_copy == True:
                    all_lyrics += data_items
        each_lyrics_file.close()
    #####################################finish getting lyrics########################################## 
    preprocessed_lyrics = preprocess_text(all_lyrics, False)
    preprocessed_lyrics = preprocessed_lyrics.split(" ")
    
    '''
    ###############################preprocessing for text of all bands###################################
    
    path = "C:\Users\Nathan\Desktop\WI project all data\COMP4075_PROJECT\project_lyrics"
    all_songs_from_file.append([f for f in os.listdir(path) if fnmatch.fnmatch(f, '*.mp3')])
    
    lyricLists=[]
    lyricsFiles = [f for f in os.listdir(path) if fnmatch.fnmatch(f, '*.txt')]
    for lyricsFile in lyricsFiles:
        if lyricsFile!= "all_words.txt" and lyricsFile != 'all_top_k_words.txt':
            with open(lyricsFile,'r') as rf:
                text=''
                for row in rf:
                    text=text+row
                lyricLists.append(preprocess_text(text, False))             #add all bands' songs
            rf.close()
        
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, sublinear_tf=True)
    wordvec = vectorizer.fit_transform(lyricLists)
    wordvec_matrix=wordvec.toarray()
    #print len(wordvec_matrix[0])
    numpy.save('tfidf_model', wordvec_matrix)                           #output training model
    
    all_words = ''
    for i, feature in enumerate(vectorizer.get_feature_names()):        #get the word sequence of the vector
        all_words +=  feature + ','
    with open('all_words.txt', 'w') as output:
        output.write(all_words)                                         #output all words (with correct sequence)
    output.close()
    

    with open('all_words.txt', 'r') as input_file:        #load in all words in all bands
        tmp_words = input_file.read()
    all_words = tmp_words.split(",")
    input_file.close()
    
    # find top-k words for each vector
    all_top_k_word = []
    for each_vector in wordvec_matrix:
        synonym = []
        top_k_words_value = nlargest(50, each_vector)
        top_k_index = []

        for each_value in top_k_words_value:
            for index, each_cal in enumerate(each_vector):
                if each_value == each_cal and index not in top_k_index:
                    top_k_index.append(index)                   #find the corresponding index in all_words
                    break
        
        for index, each_index in enumerate(top_k_index):
            for syn in wordnet.synsets(all_words[each_index]):
                for l in syn.lemmas():
                    synonym.append(l.name().encode('ascii', 'ignore'))
            synonym.append(all_words[each_index])
            
        synonym = list(set(synonym))        #remove duplicates
        print len(synonym)
        string_format = ''
        for idx, each_syn in enumerate(synonym):
            if idx != len(synonym)-1:
                string_format += each_syn + ' '
            else:
                string_format += each_syn

        all_top_k_word.append(string_format)
    
    #cast list to string and output
    output_top_k = ''
    for index, each_string in enumerate(all_top_k_word):
        if index != len(all_top_k_word) -1:
           output_top_k += each_string + ','
        else:
           output_top_k += each_string 
    
    with open('all_top_k_words.txt', 'w') as output_k:
        output_k.write(output_top_k)                                         #output all words (with correct sequence)
    output_k.close()
    #print output_top_k
    '''

    
    
    '''
    ############################################preprocessing for IDF vector###############################################
    with open('all_words.txt', 'r') as input_file:                   #load in all words in all bands
        tmp_words = input_file.read()
    all_words = tmp_words.split(",")
    input_file.close()
    
    # precalculating the idf dict
    lyricLists=[]
    path = "C:\Users\Nathan\Desktop\WI project all data\COMP4075_PROJECT\project_lyrics"
    lyricsFiles = [f for f in os.listdir(path) if fnmatch.fnmatch(f, '*.txt')]
    for lyricsFile in lyricsFiles:
        if lyricsFile != 'all_words.txt' and lyricsFile != 'all_top_k_words.txt':
            with open(lyricsFile,'r') as rf:
                text=''
                for row in rf:
                    text=text+row
                lyricLists.append(preprocess_text(text, False))     #add all bands' songs
            rf.close()
    idf_list = inv_doc_freq(all_words, lyricLists)
    with open('idf_file.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in idf_list.items():
            writer.writerow([key, value])
    csv_file.close()
    '''
         
    with open('idf_file.csv', 'rb') as csv_file:          #load in preprocessed idf file
        reader = csv.reader(csv_file)
        idf_list = dict(reader)
    csv_file.close()
    
    with open('all_words.txt', 'r') as input_file:        #load in all words in all bands
        tmp_words = input_file.read()
    all_words = tmp_words.split(",")
    input_file.close()
    
    preprocessed_lyrics = set(preprocessed_lyrics)        #remove duplicate word (faster)
    
    tf_list = term_freq(preprocessed_lyrics)    
    
    
    # remove words that are filtered out in TFIDF vectorizer
    removed_words = list(set(preprocessed_lyrics).difference(set(all_words)))
    preprocessed_lyrics = list(set(preprocessed_lyrics) - set(removed_words))
    
    
    # get the playlist TF-IDF vector
    playlist_vector = []        #the playlist TF-IDF vector
    for each_word in all_words:
        exist = False
        for each_lyrics in preprocessed_lyrics:
            if each_word == each_lyrics: 
                tfidf_value = float(tf_list[each_word]) * float(idf_list[each_word])
                playlist_vector.append(tfidf_value)
                exist = True
                break
        if exist == False:
            playlist_vector.append(0)
    
    # get top-k word in the playlist        
    top_k_words_value = nlargest(50, playlist_vector)       
    top_k_index = []    
    playlist_top_k = []
    
    for each_value in top_k_words_value:
        for index, each_cal in enumerate(playlist_vector):
            if each_value == each_cal and index not in top_k_index:
                top_k_index.append(index)                   #find the corresponding index in all_words
                break
        
    for each_index in top_k_index:
        playlist_top_k.append(all_words[each_index])            #get the actual word
        
    synonyms = []
    for each_word in playlist_top_k:
        for syn in wordnet.synsets(each_word):
            for l in syn.lemmas():
                synonyms.append(l.name().encode('ascii', 'ignore'))
    playlist_top_k += synonyms              #concadenate two list 
    playlist_top_k =  list(set(playlist_top_k))

    print "playlist keywords length (including synonyms)"
    print len(playlist_top_k)
    print "\n"
    
    # load in all bands' top-k word
    with open('all_top_k_words.txt', 'r') as top_k_input_file:        #load in all words in all bands
        tmp = top_k_input_file.read()
    all_bands_top_k = tmp.split(",")
    top_k_input_file.close()
    
    # calculate intersect keyword between user's playlist and each_band
    all_lyrics_similarity = []
    for each_string in all_bands_top_k:
        each_band_top_k = each_string.split(" ")
        all_lyrics_similarity.append(len(list(set(playlist_top_k).intersection(set(each_band_top_k)))))
    
    # print 'lyrics similarity'
    # print all_lyrics_similarity
    # print '\n'
    
    '''

    '''
    
    all_band_names = [['acid_bath'],['amon_amarth'],['amorphis'],['anthrax'],['arch_enemy'],['at_the_gates'],['august_burns_red'],['avatar'],['avenged_sevenfold'],['baroness'],\
                      ['behemoth'],['between_the_buried_and_me'],['black_sabbath'],['black_veil_brides'],['blessthefall'],['blind_guardian'],['born_of_osiris'],['breakdown_of_sanity'],['bring_me_the_horizon'],['bullet_for_my_valentine'],\
                      ['burzum'],['caliban'],['cannibal_corpse'],['carcass'],['carnifex'],['children_of_bodom'],['code_orange'],['converge'],['cradle_of_filth'],['cult_of_luna'],\
                      ['dark_lunacy'],['dark_tranquillity'],['darkthrone'],['death'],['decapitated'],['deftones'],['deicide'],['dimmu_borgir'],['dream_theater'],['eluveitie'],\
                      ['enslaved'],['epica'],['fear_factory'],['finntroll'],['fleshgod_apocalypse'],['ghost'],['gojira'],['haggard'],['ihsahn'],['immortal'],\
                      ['in_flames'],['infant_annhilator'],['insomnium'],['iron_maiden'],['isis'],['job_for_a_cowboy'],['judas_priest'],['katatonia'],['killswitch_engage'],['leprous'],\
                      ['mastodon'],['mayhem'],['megadeth'],['memphis_may_fire'],['meshuggah'],['metallica'],['motionless_in_white'],['motley_crue'],['mr_bungle'],['napalm_death'],\
                      ['neurosis'],['nile'],['opeth'],['ozzy_osbourne'],['paradise_lost'],['periphery'],['protest_the_hero'],['rhapsody_of_fire'],['rings_of_saturn'],['sepultura'],\
                      ['slayer'],['slipknot'],['suicide_silence'],['swallow_the_sun'],['symphony_x'],['testament'],['the_black_dahlia_murder'],['the_faceless'],['thy_art_is_murder'],['today_is_the_day'],\
                      ['torche'],['trivium'],['underoath'],['veil_of_maya'],['venom'],['while_she_sleeps'],['whitechapel'],['wintersun'],['within_temptation'],['xandria']]
    
    
    # for getting the top-k most similar band (lyrics)
    most_similar_band_by_lyrics = []  
    top_k_similar_values = nlargest(5, all_lyrics_similarity)
    
    similar_index = []
    for each_value in top_k_similar_values:
        for index, sim in enumerate(all_lyrics_similarity):
            if each_value == sim and index not in similar_index:
                similar_index.append(index)
    
    for each_index in similar_index:
        most_similar_band_by_lyrics.append(all_band_names[each_index])
    print 'top-5 bands for lyrics'
    print most_similar_band_by_lyrics
    print '\n'
    
    
    # min-max normalization
    normalized_lyrics_similarity = [] 
    max_lyrics_similarity = max(all_lyrics_similarity)
    min_lyrics_similarity = min(all_lyrics_similarity)
    for each_score in all_lyrics_similarity:
        normalized_lyrics_similarity.append(float((each_score - min_lyrics_similarity))/float((max_lyrics_similarity - min_lyrics_similarity)))
    #print normalized_lyrics_similarity
    

    
    
    # create a new folder and copy all playlist songs into the folder for audio feature extraction
    all_song_dir = []
    all_song_path_name = []

    for each_band in random_bands:
        all_song_dir.append("/Users/sunjingxuan/Desktop/WI_project_all_data/COMP4075_PROJECT/project_songs/songs_" + each_band)
    for each_dir in all_song_dir:
        for each_song in os.listdir(each_dir):
            all_song_path_name.append(each_dir+"/"+each_song)
    
    all_path_needed = []
    for each_selected_song in selected_playlist:
            for each_song_path in all_song_path_name:
                if each_selected_song in each_song_path :
                    all_path_needed.append(each_song_path)
                    break
    
    play_list_directory = "/Users/sunjingxuan/Desktop/user_playlist"
    if not os.path.exists(play_list_directory):   #create a tmp directory to store playlist 
        os.makedirs(play_list_directory)          #for calling audio feature function
    else:
        shutil.rmtree(play_list_directory) 
        os.makedirs(play_list_directory)
        
    for each_path in all_path_needed:

        shutil.copy(each_path, play_list_directory)
    #############################finish outputing songs to a file###############################
    
    # load in preextracted audio feature for each band
    [allMtFeatures, wavFilesList2] = afe.dirWavFeatureExtraction(play_list_directory, 1.0, 1.0, 0.050, 0.050, False)
    allMtFeatures = allMtFeatures.flatten()
    all_bands_audio_feature = numpy.load("audio_feature_new_sequence.npy")  #get all bands features
    
    #calculate audio similarity 
    all_audio_similarity = []
    for one_band_audio_feature in all_bands_audio_feature:
        one_band_audio_feature = one_band_audio_feature.flatten()
        one_audio_similarity = dist.cosine(one_band_audio_feature, allMtFeatures)
        all_audio_similarity.append(one_audio_similarity)
    
    # print 'audio similarity'
    # print all_audio_similarity
    # print '\n'
    
    # for getting the top-k most similar band (audio)
    most_similar_band_by_audio = []  
    top_k_similar_values = nlargest(5, all_audio_similarity)
    
    similar_index = []
    for each_value in top_k_similar_values:
        for index, sim in enumerate(all_audio_similarity):
            if each_value == sim and index not in similar_index:
                similar_index.append(index)
    
    for each_index in similar_index:
        most_similar_band_by_audio.append(all_band_names[each_index])
    print 'top-5 bands for audio'
    print most_similar_band_by_audio
    print '\n'
    
    
    # min-max normalization
    normalized_audio_similarity = [] 
    max_audio_similarity = max(all_audio_similarity)
    min_audio_similarity = min(all_audio_similarity)
    for each_score in all_audio_similarity:
        normalized_audio_similarity.append(float((each_score - min_audio_similarity))/float((max_audio_similarity - min_audio_similarity)))
    # print normalized_audio_similarity
    
    
    # merge and get overall similarity 
    overall_weighted_similarity = []
    weight = 0.7
    for index, each_sim in enumerate(normalized_lyrics_similarity):
        overall_weighted_similarity.append((float(weight) * float(normalized_lyrics_similarity[index])) + (float((1-weight)) * float(normalized_audio_similarity[index])))
    
    # get final recommendation
    final_recommendation = []  
    top_k_similar_values = nlargest(5, overall_weighted_similarity)
    
    similar_index = []
    for each_value in top_k_similar_values:
        for index, sim in enumerate(overall_weighted_similarity):
            if each_value == sim and index not in similar_index:
                similar_index.append(index)
    
    for each_index in similar_index:
        final_recommendation.append(all_band_names[each_index])
    print 'Final recommendation'
    print final_recommendation
    print '\n'
    
    # check if the style is correct
    num_correct = 0
    correct_list = []
    for each_recommendation in final_recommendation:
        for each_band in style_list[random_style]:
            if each_band == each_recommendation[0]:
                num_correct += 1
                correct_list.append(each_recommendation)
    
    print "number of correct recommendation"
    print num_correct
    print '\n'

    print 'recomendation list'
    print correct_list
    
    print 'time used'
    print time.time() - start_time
    print '\n'
    
    
    
if __name__ == "__main__":
    main()




