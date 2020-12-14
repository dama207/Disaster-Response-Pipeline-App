import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble  import BaggingClassifier
import pickle


def load_data(database_filepath):
    '''Load dataframe and returns X, Y, category_names
    Args:
       database_filepath :  Path file data.
    
    Returns:
        X: Dataframe with features to train model.
        Y: Target features.
        category_names: Name of categories.
    
    '''
    
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath )
    df = pd.read_sql('SELECT * FROM categorias_base', engine)
    X = df.message.values
    Y = df.iloc[:, 4:39]
    Y['related'].replace(2, 1, inplace=True)
    category_names =  Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    '''clean text (remove whitespace and punctuation) , tokenize and lemmatize text 
    Args:
       text : text to prepare.
    Returns:
       clean_tokens:  normalized, tokenized and lemmatized text.
    '''
    
    text = re.sub(r'[^\w\s]','',text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''train model(RandomForestClassifier)
    Returns:
        model: best model to classifier data. 
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0),
        'tfidf__use_idf': (True, False)
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''evaluate model with Accuracy, precision score, recall score and f1 score
    Args:
        model: model to evaluate.
        X_test: features to evaluate model.
        Y_test: Real values of target feature.
        category_names: Name of categories.
    '''
    
    y_pred = model.predict(X_test)
    
    print("Accuracy")
    print((y_pred == Y_test).mean())
    
    y_pred_df = pd.DataFrame(y_pred, columns=Y_test.columns)
    evaluation = {}
    for column in Y_test.columns:
        evaluation[column] = []
        evaluation[column].append(precision_score(Y_test[column], y_pred_df[column]))
        evaluation[column].append(recall_score(Y_test[column], y_pred_df[column]))
        evaluation[column].append(f1_score(Y_test[column], y_pred_df[column]))
    scores=pd.DataFrame(evaluation)
  
    print(scores)
    
    pass


def save_model(model, model_filepath):
    '''save model in a pkl file
    Args:
       model: model to save.
       model_filepath: Name to save model   
    '''

    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()