import contractions
import csv
import emoji
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models import Phrases
from gensim.models import LdaMulticore
from gensim.models import LsiModel
from gensim.models.phrases import Phraser
import joblib
from multiprocessing import Pool, cpu_count
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
import os
import pandas as pd
import pickle
import re
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import warnings

#---------------------------------------------------------


# Stopwörter einmalig laden
try:
    find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
    
stop_words = set(stopwords.words("english"))

# Benutzerdefinierte Stopwörter für den spezifischen Anwendungsfall
# Diese Wörter wurden basierend auf Tests als wenig informativ eingestuft
manual_stopwords = {"still", "even", "never", "back", "movie", "letter", "house", "people", "situation", "triple", "choice",
                    "numerous", "option", "adrian", "collection", "notice", "told", "goodlow", "mbps", "dollar", "hold",
                    "additional", "reason", "finally","second", "minute", "hour", "annual", "year", "time", "week", "month",
                    "monthly", "january", "february", "march", "april", "may", "june", "july", "august", "september",
                    "october", "november", "december", "morning", "monday", "tuesday", "wednesday", "thursday", "friday",
                    "saturday", "sunday", "michael", "charles", "anything", "marshall", "rachael", "terrance", "someone",
                    "however", "today", "also", "thing", "always", "next", "first", "much", "many", "really", "ever", "nothing", 
                    "later", "last", "several", "line", "past", "everything"}

# Kombinierte Stopwörter aus Standard- und benutzerdefinierten Stopwörtern
combined_stopwords = stop_words | manual_stopwords

# Regex-Muster einmalig kompilieren für bessere Performance
email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b') # E-Mail-Muster
url_pattern = re.compile(r'\bhttps?:\/\/\S+|www\.\S+\b') # URL-Muster
sonderzeichen_pattern = re.compile(r'[:.!?,|<>*`{}§\'\"\\/$&%#+;_()=-]') # Sonderzeichen
zahlen_pattern = re.compile(r'[0-9]+') # Zahlen
noise_pattern = re.compile(r'\b[a-z]{,2}\b') # Sehr kurze Wörter (max. 2 Zeichen)

# NLP-Tools initialisieren
tokenizer = TreebankWordTokenizer() # Tokenizer für die Textsegmentierung
lemmatizer = WordNetLemmatizer() # Lemmatizer für die Grundformreduktion
vectorizer = TfidfVectorizer() # TF-IDF-Vektorisierung für Feature-Extraktion

# Warnungen ignorieren
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Erlaubte POS-Tags
# JJ: Adjektive, NNP: Eigennamen, RB: Adverbien, NN/NNS: Nomen (Singular/Plural)
ALLOWED_TAGS = {"JJ", "NNP", "RB", "NN", "NNS"}


#---------------------Funktionen---------------------#

def process_document(document):  
    """
    Textvorverarbeitung mit mehreren Reinigungsschritten.
    
    Args:
        document: Rohes Dokument zum Verarbeiten
        
    Returns:
        Liste von vorverarbeiteten Tokens
    """
    # Überprüfen, ob Dokument ein Text ist
    if not isinstance(document, str) or not document:
        return []
    
    document = document.lower() # Alles in Kleinbuchstaben umwandeln
    document = url_pattern.sub('', document) # URLs entfernen
    document = email_pattern.sub('', document) # E-Mail-Adressen entfernen
    document = emoji.replace_emoji(document, replace='') # Emojis entfernen
    document = zahlen_pattern.sub('', document) # Zahlen entfernen
    document = contractions.fix(document) # Kurzformen erweitern (z.B. "don't" -> "do not")
    document = sonderzeichen_pattern.sub('', document) # Sonderzeichen entfernen
    document = noise_pattern.sub('', document) # Sehr kurze Wörter entfernen
    
    # Text in einzelne Wörter zerlegen (Tokenisierung)
    tokenized_document = tokenizer.tokenize(document)
    
    # Wörter auf ihre Grundform reduzieren (Lemmatisierung)
    lematized_doc = [lemmatizer.lemmatize(token) for token in tokenized_document]
    
    # POS-Tagging und Filterung
    tagged_doc = pos_tag(lematized_doc)
    final_document = []
    for token, tag in tagged_doc:
        # Nur relevante Wörter behalten: nicht in Stopwörtern, länger als 3 Zeichen und mit erlaubter Wortart
        if token not in combined_stopwords and len(token) > 3 and tag in ALLOWED_TAGS:
            final_document.append(token)
    
    return final_document


def tokenisieren(text):
    """
    Einfache Tokenisierung eines Textes ohne weitere Verarbeitung.
    
    Args:
        text: Zu tokenisierender Text
        
    Returns:
        Liste von Tokens
    """
    
    return tokenizer.tokenize(text)


def ngram_processing(tokenized_text_corpus):
    """
    Erstellt Bigramme und Trigramme aus einem tokenisierten Textkorpus.
    
    Args:
        tokenized_text_corpus: Liste von tokenisierten Dokumenten
        
    Returns:
        Liste der tokenisierten Dokumente mit erkannten Bigrammen und Trigrammen
    """
    # Bigramm-Modell mit Mindestzahl an Vorkommen erstellen
    bigram = Phrases(tokenized_text_corpus, min_count=0.04)
    bigram_model = Phraser(bigram) 
    
    # Trigramm-Modell basierend auf den Bigrammen erstellen
    trigram = Phrases(bigram[tokenized_text_corpus], min_count=0.1)
    trigram_model = Phraser(trigram)

    # Bigramme und Trigramme auf alle Dokumente anwenden
    return [trigram_model[bigram_model[text]] for text in tokenized_text_corpus]


def compute_coherence_value(processed_corpus, dictionary, vectorization_method, model=None, topic_keywords=None):
    """
    Berechnet den Kohärenzwert für ein Modell.
    
    Args:
        processed_corpus: Vorverarbeitete Textdokumente
        dictionary: Gensim-Wörterbuch
        vectorization_method: Verwendete Word-Embedding-Methode ('TF-IDF' oder 'BOW')
        model: Das trainierte LDA-Modell (für BOW)
        topic_keywords: Liste der Keywords pro Topic (für TF-IDF)
        
    Returns:
        float: Kohärenzwert des Modells
    """
    if vectorization_method == "TF-IDF":
        coherencemodel = CoherenceModel(topics=topic_keywords, texts=processed_corpus, dictionary=dictionary, coherence='c_v')
    else:
        coherencemodel = CoherenceModel(model=model, texts=processed_corpus, dictionary=dictionary, coherence='c_v')
    
    return coherencemodel.get_coherence()


def train_lda_model_tfidf(tfidf_matrix, num_topics, passes, alpha, eta):
    """
    Trainiert ein LDA-Modell mit TF-IDF-Vektorisierung und den angegebenen Parametern.
    
    Args:
        tfidf_matrix: Die TF-IDF-Matrix der Dokumente
        num_topics: Anzahl der zu extrahierenden Themen
        passes: Anzahl der Trainingsdurchläufe
        alpha: Dokument-Thema Priorverteilung
        eta: Thema-Wort Priorverteilung
        
    Returns:
        tuple: (trainiertes LDA-Modell, Trainingszeit in Sekunden)
    """
    
    start_time = time.time()
    
    # Scikit-learn LDA-Modell mit angegebenen Parametern erstellen
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        max_iter=passes,
        doc_topic_prior=alpha,
        topic_word_prior=eta,
        n_jobs=min(cpu_count()-1, 5)  # Multiprocessing
    )
    
    # Modell trainieren
    trained_lda_model = lda_model.fit(tfidf_matrix)
    training_time = time.time() - start_time
    
    return trained_lda_model, training_time 


def train_lda_model_bow(corpus, dictionary, num_topics, passes, chunksize, alpha, eta):
    """
    Trainiert ein LDA-Modell mit Bag-of-Words-Vektorisierung und den angegebenen Parametern.
    
    Args:
        corpus: Das BOW-Korpus
        dictionary: Das Gensim-Wörterbuch
        num_topics: Anzahl der zu extrahierenden Themen
        passes: Anzahl der Trainingsdurchläufe
        chunksize: Größe der Batches für Training
        alpha: Dokument-Thema Priorverteilung
        eta: Thema-Wort Priorverteilung
        
    Returns:
        tuple: (trainiertes LDA-Modell, Trainingszeit in Sekunden)
    """
    
    start_time = time.time()
    
    # Gensim LdaMulticore-Modell mit angegebenen Parametern erstellen
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        alpha=alpha,
        eta=eta,
        eval_every=None,  # Evaluierung während des Trainings deaktivieren für Geschwindigkeit
        chunksize=chunksize,
        workers=min(cpu_count()-1, 5)  # Multiprocessing
    )
    
    training_time = time.time() - start_time
    
    return lda_model, training_time


def train_lsa_model_tfidf(corpus, num_topics):
    """
    Trainiert ein LSA-Modell mit TF-IDF-Vektorisierung.
    
    Args:
        corpus: Die TF-IDF-Matrix der Dokumente
        num_topics: Anzahl der zu extrahierenden Themen
        
    Returns:
        tuple: (trainiertes LSA-Modell, Trainingszeit in Sekunden)
    """
    
    start_time = time.time()
    
    # Scikit-learn TruncatedSVD für LSA mit TF-IDF verwenden
    svd_model = TruncatedSVD(n_components=num_topics, algorithm='randomized', n_iter=10, random_state=42)
    svd_model.fit_transform(corpus)
    
    training_time = time.time() - start_time
    
    return svd_model, training_time


def train_lsa_model_bow(corpus, dictionary, num_topics):
    """
    Trainiert ein LSA-Modell mit Bag-of-Words-Vektorisierung.
    
    Args:
        corpus: Das BOW-Korpus
        dictionary: Das Gensim-Wörterbuch
        num_topics: Anzahl der zu extrahierenden Themen
        
    Returns:
        tuple: (trainiertes LSA-Modell, Trainingszeit in Sekunden)
    """
    
    start_time = time.time()
    
    # Gensim LsiModel für LSA mit BOW verwenden
    lsa_model = LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    
    training_time = time.time() - start_time
    
    return lsa_model, training_time 


def run_lda_parameter_search(dictionary, corpus, processed_corpus, output_folder, vectorization_method):
    """
    Führt eine Parametersuche für das LDA-Modell durch.
    
    Args:
        dictionary: Das Gensim-Wörterbuch
        corpus: Das vektorisierte Korpus
        processed_corpus: Die vorverarbeiteten Dokumente
        output_folder: Ordner zum Speichern der Ergebnisse
        vectorization_method: Verwendete Word-Embedding-Methode ('TF-IDF' oder 'BOW')
        
    Returns:
        dict: Die besten gefundenen Parameter
    """
    
    # Reduzierter Parameterraum für effizientere Suche
    topics_list = [6, 8, 10, 12, 14] # Zu testende Themenanzahlen
    passes_list = [20] # Anzahl der Trainingsdurchläufe
    alpha_list = [0.01, 0.1, 0.5] # Zu testende Alpha Werte
    eta_list = [0.01, 0.1, 0.5] # Zu testende Beta Werte
    chunksize = 5000
    
    # Ergebnisse für den Vergleich speichern
    results = []
    
    # Gesamtfortschritt verfolgen
    total_combinations = len(topics_list) * len(passes_list) * len(alpha_list) * len(eta_list)
    current_combination = 0
    
    print(f"Running parameter search with {total_combinations} combinations")
    
    # Kombinationen aller Parameter durchlaufen
    for num_topics in topics_list:
        for passes in passes_list:
            for alpha in alpha_list:
                for eta in eta_list:
                    current_combination += 1
                    print(f"Progress: {current_combination}/{total_combinations} combinations ({current_combination/total_combinations*100:.1f}%)")
                    print(f"Training model with: Topics={num_topics}, Passes={passes}, Alpha={alpha}, Eta={eta}")
                    
                    if vectorization_method == "TF-IDF":
                        # TF-IDF-basiertes LDA-Modell trainieren und Zeit messen
                        lda_model, training_time = train_lda_model_tfidf(
                            corpus, num_topics, passes, alpha, eta
                        )
                        
                        # Top-Begriffe für jedes Thema extrahieren und anzeigen
                        feature_names = vectorizer.get_feature_names_out()
                        print("Top words per topic:")
                        topic_keywords = []
                        for topic_idx, topic in enumerate(lda_model.components_):
                            top_term_indices = topic.argsort()[-10:]
                            top_terms = [feature_names[i] for i in top_term_indices]
                            topic_keywords.append(top_terms)
                            print(f"Topic {topic_idx+1}: {', '.join(top_terms)}")
                            
                        # Kohärenzwert berechnen
                        coherence_value = compute_coherence_value(processed_corpus, dictionary, vectorization_method, topic_keywords=topic_keywords)
                        print(f"Coherence: {coherence_value:.4f} (training time: {training_time:.1f}s)")                        
                        print("-" * 50)                        
                    else:
                        # BOW-basiertes LDA-Modell trainieren und Zeit messen
                        lda_model, training_time = train_lda_model_bow(
                            corpus, dictionary, num_topics, passes, chunksize, alpha, eta
                        )
                        
                        # Kohärenzwert berechnen
                        coherence_value = compute_coherence_value(processed_corpus, dictionary, vectorization_method, model=lda_model)
                        
                        # Top-Begriffe für jedes Thema anzeigen
                        print(f"Coherence: {coherence_value:.4f} (training time: {training_time:.1f}s)")
                        print("Top words per topic:")
                        for idx, topic in lda_model.print_topics():
                            print(f"Topic {idx+1}: {topic}")
                        print("-" * 50)           
                     
                    # Ergebnisse für diese Parameterkombination speichern    
                    results.append({
                        'num_topics': num_topics,
                        'passes': passes,
                        'alpha': str(alpha),
                        'eta': str(eta),
                        'coherence': coherence_value,
                        'training_time': training_time
                    })
    
    # Ergebnisse in DataFrame konvertieren und speichern
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_folder}/parameter_search_results_lda_{vectorization_method}.csv", index=False)

    
    # Beste Modellparameter zurückgeben (höchster Kohärenzwert)
    best_result = results_df.loc[results_df['coherence'].idxmax()]
    print(f"\nBest model parameters:")
    print(f"Topics: {best_result['num_topics']}")
    print(f"Passes: {best_result['passes']}")
    print(f"Alpha: {best_result['alpha']}")
    print(f"Eta: {best_result['eta']}")
    print(f"Coherence: {best_result['coherence']:.4f}")
    
    return best_result.to_dict()


def run_lsa_parameter_search(dictionary, corpus, processed_corpus, output_folder, vectorization_method): 
    """
    Führt eine Parametersuche für das LSA-Modell durch, um die optimale Anzahl an Topics zu finden.
    
    Args:
        dictionary: Das Gensim-Wörterbuch
        corpus: Das vektorisierte Textkorpus
        processed_corpus: Die vorverarbeiteten Textdokumente
        output_folder: Ordner zum Speichern der Ergebnisse
        vectorization_method: Verwendete Word-Embedding-Methode ('TF-IDF' oder 'BOW')
        
    Returns:
        dict: Die besten gefundenen Parameter
    """ 
    # Topic-Werte, die getestet werden sollen
    topics_list = [6, 8, 10, 12, 14]

    # Ergebnisse für den Vergleich speichern
    results = []
    
    # Gesamtfortschritt verfolgen
    total_combinations = len(topics_list)
    current_combination = 0
    
    print(f"Running parameter search with {total_combinations} combinations")
    
    # Für jede Anzahl von Topics ein Modell trainieren und evaluieren
    for num_topics in topics_list:
        current_combination += 1
        print(f"Progress: {current_combination}/{total_combinations} combinations ({current_combination/total_combinations*100:.1f}%)")
        print(f"Training model with: Topics={num_topics}")
        
        if vectorization_method == "TF-IDF":
            # TF-IDF-basiertes LSA-Modell trainieren und Zeit messen
            svd_model, training_time = train_lsa_model_tfidf(corpus, num_topics)
            
            # Top-Begriffe für jedes Thema extrahieren
            terms = vectorizer.get_feature_names_out()
            topic_keywords = []
            for topic_idx, topic in enumerate(svd_model.components_):
                top_term_indices = topic.argsort()[-10:] # Die 10 wichtigsten Begriffe
                top_terms = [terms[i] for i in top_term_indices]
                topic_keywords.append(top_terms)
                print(f"Topic {topic_idx+1}: {', '.join(top_terms)}")
                
            # Kohärenzwert für TF-IDF-Modell berechnen    
            coherence_value = compute_coherence_value(processed_corpus, dictionary, 'TF-IDF', topic_keywords=topic_keywords)
        
        else:
            # Bag-of-Words-basiertes LSA-Modell trainieren und Zeit messen
            lsa_model, training_time = train_lsa_model_bow(corpus, dictionary, num_topics)
            
            # Top-Begriffe für jedes Thema ausgeben
            for topic_idx, topic in lsa_model.show_topics(num_topics=num_topics, num_words=10, formatted=False):
                top_terms = [term for term, _ in topic]
                print(f"Topic {topic_idx+1}: {', '.join(top_terms)}")
                
            # Kohärenzwert für BoW-Modell berechnen
            coherence_value = compute_coherence_value(processed_corpus, dictionary, 'BOW', model=lsa_model)
        
        # Ergebnisse speichern
        results.append({
            'num_topics': num_topics,
            'coherence': coherence_value,
            'training_time': training_time
        })
        
        # Kohärenzwert und Trainingszeit ausgeben
        print(f"Coherence: {coherence_value:.4f} (training time: {training_time:.1f}s)")
        print("-" * 50)
        
    # Ergebnisse in DataFrame konvertieren und speichern
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_folder}/parameter_search_results_lsa_{vectorization_method}.csv", index=False)
    
    # Beste Modellparameter zurückgeben (höchster Kohärenzwert)
    best_result = results_df.loc[results_df['coherence'].idxmax()]
    print(f"\nBest model parameters:")
    print(f"Topics: {best_result['num_topics']}")
    print(f"Coherence: {best_result['coherence']:.4f}")
    
    return best_result.to_dict()


def train_best_lda_model(corpus, dictionary, best_params, output_folder, vectorization_method):
    """
    Trainiert das finale LDA-Modell mit den besten gefundenen Parametern und speichert es.
    
    Args:
        corpus: Das vektorisierte Textkorpus
        dictionary: Das Gensim-Wörterbuch
        best_params: Die besten Parameter aus der Parametersuche
        output_folder: Ordner zum Speichern der Ergebnisse
        vectorization_method: Verwendete Word-Embedding-Methode ('TF-IDF' oder 'BOW')
        
    Returns:
        Das trainierte LDA-Modell
    """
    
    print(f"\nTraining final {topic_modeling_method} model with best parameters...")
    
    if vectorization_method == "TF-IDF":
        # TF-IDF-basiertes LDA-Modell mit optimalen Parametern trainieren
        lda_model, _ = train_lda_model_tfidf(
            corpus, 
            int(best_params['num_topics']), 
            int(best_params['passes']), 
            best_params['alpha'] if best_params['alpha'] in ['symmetric', 'asymmetric'] else float(best_params['alpha']),
            best_params['eta'] if best_params['eta'] in ['symmetric', 'asymmetric'] else float(best_params['eta'])
        )
        
        # Feature-Namen abrufen und Top-Begriffe für jedes Thema ausgeben
        feature_names = vectorizer.get_feature_names_out()
        
        print("\nTopics in final model:")
        for topic_idx, topic in enumerate(lda_model.components_):
            top_term_indices = topic.argsort()[-10:]
            top_terms = [feature_names[i] for i in top_term_indices]
            print(f"Topic {topic_idx+1}: {', '.join(top_terms)}")
        
        # Modell speichern        
        joblib.dump(lda_model, f"{output_folder}/best_lda_model_tfidf_topics_{int(best_params['num_topics'])}.pkl")
        
    else:
        # Bag-of-Words-basiertes LDA-Modell mit optimalen Parametern trainieren
        lda_model, _ = train_lda_model_bow(
            corpus, 
            dictionary, 
            int(best_params['num_topics']), 
            int(best_params['passes']), 
            5000,
            best_params['alpha'] if best_params['alpha'] in ['symmetric', 'asymmetric'] else float(best_params['alpha']),
            best_params['eta'] if best_params['eta'] in ['symmetric', 'asymmetric'] else float(best_params['eta'])
        )
        
        # Modell speichern
        lda_model.save(f"{output_folder}/best_lda_model")

        # Themen im finalen Modell ausgeben
        print("\nTopics in final model:")
        for idx, topic in lda_model.print_topics():
            print(f"🔹 Topic {idx+1}:\n {topic}\n")
        print("-" * 50)
    
    return lda_model


def train_best_lsa_model(corpus, dictionary, best_params, output_folder, vectorization_method):
    """
    Trainiert das finale LSA-Modell mit den besten gefundenen Parametern und speichert es.
    
    Args:
        corpus: Das Textkorpus in vektorisierter Form
        dictionary: Das Gensim-Wörterbuch
        best_params: Die besten Parameter aus der Parametersuche
        output_folder: Ordner zum Speichern der Ergebnisse
        vectorization_method: Verwendete Word-Embedding-Methode ('TF-IDF' oder 'BOW')
        
    Returns:
        Das trainierte LSA-Modell
    """
    
    print(f"\nTraining final model with best parameters...")
    
    # Für TF-IDF-basierte LSA
    if vectorization_method == "TF-IDF":
        # LSA-Modell mit optimaler Themenanzahl berechnen
        lsa_model, _ = train_lsa_model_tfidf(
            corpus, 
            int(best_params['num_topics'])
        )
        
        # Top-Begriffe für jedes Thema anzeigen
        terms = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lsa_model.components_):
            top_term_indices = topic.argsort()[-10:]  # Die 10 wichtigsten Wörter extrahieren
            top_terms = [terms[i] for i in top_term_indices]
            print(f"Topic {topic_idx+1}: {', '.join(top_terms)}")
                
        # Modell speichern
        joblib.dump(lsa_model, f"{output_folder}/best_lsa_model_tfidf_topics_{int(best_params['num_topics'])}.pkl")
    
    # Für Bag-of-Words-basierte LSA
    else:
        num_topics = int(best_params['num_topics'])
        
        # LSA-Modell mit optimaler Themenanzahl berechnen
        lsa_model, _ = train_lsa_model_bow(
            corpus, 
            dictionary, 
            num_topics
        )
        
        # Modell speichern
        lsa_model.save(f"{output_folder}/best_model_topics_{num_topics}.pkl")
        
        # Top-Begriffe für jedes Thema anzeigen
        for topic_idx, topic in lsa_model.show_topics(num_topics=num_topics, num_words=10, formatted=False):
            top_terms = [term for term, _ in topic]
            print(f"Topic {topic_idx+1}: {', '.join(top_terms)}")
    
    
    return lsa_model


def get_user_preferences():
    """
    Interaktive Abfrage der Benutzereinstellungen für Topic-Modeling und Word-Embedding.
    
    Returns:
        tuple: (topic_modeling_method, vectorization_method)
            - topic_modeling_method: Die gewählte Topic-Modeling-Methode ('LDA' oder 'LSA')
            - vectorization_method: Die gewählte Word-Embedding-Methode ('TF-IDF' oder 'BOW')
    """
    topic_modeling_method = None
    vectorization_method = None
    
    
    # Abfrage der Topic-Modeling-Methode
    while not topic_modeling_method:
        print("Which Topic Modeling method do you want?")
        u_input = input("Type 'LDA' or 'LSA': ")
        if u_input == 'LDA':
            topic_modeling_method = 'LDA'
        elif u_input == 'LSA':
            topic_modeling_method = 'LSA'
        else:
            print("Wrong Input. Try again...")  
    
    # Abfrage der Word-Embedding-Methode
    while not vectorization_method:
        print("Which word embedding method do you want?")
        u_input = input("Type 'TF-IDF' or 'BOW': ")
        if u_input == 'TF-IDF':
            vectorization_method = 'TF-IDF'
        elif u_input == 'BOW':
            vectorization_method = 'BOW'
        else:
            print("Wrong Input. Try again...")
    
    return topic_modeling_method, vectorization_method
         
    
    
#-----------------------------------------------#      
#--------------|Start des Skripts|--------------#
#-----------------------------------------------#

if __name__ == "__main__":
    start_time = time.time() # Startzeit für Laufzeitmessung
    
    # Benutzereinstellungen für Topic-Modeling und Word-Embedding Methoden abfragen
    topic_modeling_method, vectorization_method = get_user_preferences()
    
    # Ausgabeordner basierend auf der gewählten Topic-Modeling und Word-Embedding Methode erstellen
    output_folder = ""
    if topic_modeling_method == "LDA":
        output_folder += ("lda_results")
    else:
        output_folder += ("lsa_results")
        
    if vectorization_method == "TF-IDF":
        output_folder += ("/tfidf")
    else:
        output_folder += ("/bow")


    # Prüfen ob der Ausgabeordner existiert, falls nicht wird er erstellt
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)        
    
    # Pfad zur Cache-Datei für vorverarbeiteten Korpus definieren
    cache_file = "data/preprocessed_corpus.pkl"
    
    # Überprüfen ob bereits ein vorverarbeiteter Korpus im Cache existiert
    if os.path.exists(cache_file):
        print("Loading preprocessed corpus from cache...")
        with open(cache_file, 'rb') as f:
            processed_corpus = pickle.load(f)
    else:
        print("Processing corpus from CSV...")
        # Dokumente aus CSV-Datei einlesen
        documents = []
        with open('data/comcast_consumeraffairs_complaints.csv', 'r', encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader) # Kopfzeile überspringen
            for document in csv_reader:
                documents.append(document[3]) # Text aus vierter Spalte extrahieren
        
        # Dokumente parallel verarbeiten
        print(f"Processing {len(documents)} documents with {min(cpu_count()-1, 5)} processes...")
        with Pool(processes=min(cpu_count()-1, 5)) as pool:
            preprocessed_corpus = pool.map(process_document, documents)
        
        # Leere Dokumente aus dem Korpus herausfiltern
        preprocessed_corpus = [doc for doc in preprocessed_corpus if doc]
        
        # N-Gramme (Bigramme und Trigramme) erzeugen
        print("Creating bigrams and trigrams...")
        processed_corpus = ngram_processing(preprocessed_corpus)
        
        # Verarbeiteten Korpus für spätere Verwendung im Cache speichern
        with open(cache_file, 'wb') as f:
            pickle.dump(processed_corpus, f)
    
    print(f"Corpus size: {len(processed_corpus)} documents")       
    
    # Wörterbuch und Korpus für Topic-Modeling erstellen
    print("Creating dictionary and corpus...")
    dictionary = Dictionary(processed_corpus)
    # Extreme Wörter filtern (zu selten oder zu häufig)
    dictionary.filter_extremes(no_below=0.01, no_above=0.7)
    print("-" * 50)
    
    # Korpusrepräsentation basierend auf der gewählten Word-Embedding-Methode erstellen
    if vectorization_method == 'BOW':
        # Bag-of-Words Darstellung
        corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    else:
        # TF-IDF Darstellung
        processed_corpus_string = [" ".join(doc) for doc in processed_corpus]
        corpus = vectorizer.fit_transform(processed_corpus_string)

        
    # Parameter-Suche und Training des besten Modells durchführen
    if topic_modeling_method == "LDA":
        # Für LDA: Parametersuche durchführen und bestes Modell trainieren
        best_params = run_lda_parameter_search(dictionary, corpus, processed_corpus, output_folder, vectorization_method)
        train_best_lda_model(corpus, dictionary, best_params, output_folder, vectorization_method)
    else:
        # Für LSA: Parametersuche durchführen und bestes Modell trainieren
        best_params = run_lsa_parameter_search(dictionary, corpus, processed_corpus, output_folder, vectorization_method)
        train_best_lsa_model(corpus, dictionary, best_params, output_folder, vectorization_method)

    # Gesamtlaufzeit berechnen und ausgeben
    total_time = (time.time() - start_time) / 60
    print(f"\nTotal execution time: {total_time:.2f} minutes")
