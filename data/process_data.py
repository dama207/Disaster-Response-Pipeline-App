import sys
import pandas as pd
import re 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''Load data of messages and categories and marge them on only dataframe
    Args:
        messages_filepath : Path file messages.
        categories_filepath : Path file categories.
        
    Returns:
    
        df: Dataframe with data merged
        
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id')
    
    return df


def clean_data(df):
    
    '''Clean dataframe and build a feature for each category, convert text in categories in 1 or 0 for each category, delete duplicates
    Args:
        df: Dataframe
    Returns:
        df: Dataframe clean with column for each category with binary values
    
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.join(df.categories.str.split(';',expand=True))
    categories=categories.drop(columns= ['categories','message','original','genre'])
    
    # rename the columns of `categories`
    columns_name = ['id']+list(categories.iloc[0,1:37].apply(lambda x: str(x).replace("-0", '')).apply(lambda x: str(x).replace("-1", '')))
    categories.columns=columns_name
    
    categories.set_index('id', inplace=True)
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : int((re.sub("\D", "",x))))
        # replace 2 to 1 
        categories[column].replace(2, 1, inplace=True)
    
    categories.reset_index(inplace=True)
    # drop the original categories column from `df`
    df.drop(columns= ['categories'], inplace =True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, left_on='id', right_on='id')
    # drop duplicates
    df=df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    
    '''save data into base
    Args:
       df : dataframe
       database_filename : name database   
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('categorias_base', engine, index=False)
    
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()