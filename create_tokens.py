import pandas as pd
import numpy as np
import en_core_sci_lg
import de_core_news_lg
import os
import datetime
from pandarallel import pandarallel
import warnings

warnings.filterwarnings('ignore')
pandarallel.initialize(use_memory_fs=False)

# client = Elasticsearch([{'host': 'localhost'}, {'port': 9200}])

nlp_german = de_core_news_lg.load(exclude=["parser", "ner", "tok2vec", "textcat"])

nlp_sci = en_core_sci_lg.load(exclude=["parser", "ner", "tok2vec", "textcat"])


# UNIVERSAL

def is_token_allowed_german(token):
    '''
         Only allow valid tokens which are not stop words
         and punctuation symbols.
    '''
    if not token or not token.text.strip() or token.is_stop or token.is_punct:
        return False
    return True


def preprocesstoken_german(token):
    # Reduce token to its lowercase lemma form
    return token.lemma_.strip().lower()


# def tokenize_german(x):
#     try:
#         return str([preprocesstoken_german(token) for token in nlp_german(x) if is_token_allowed_german(token)])
#     except:
#         return str([])


def tokenize_string_german(x):
    return ",".join([preprocesstoken_german(token) for token in nlp_german(x) if is_token_allowed_german(token)])


def tokenize_german_numpy(x):
    return np.array(
        [[",".join([preprocesstoken_german(token) for token in nlp_german(i) if is_token_allowed_german(token)])] for i
         in x], dtype=str).reshape((len(x),))


def tokenize_sci_numpy(x):
    return np.array(
        [[",".join([preprocesstoken_sci(token) for token in nlp_sci(i) if is_token_allowed_sci(token)])] for i in x],
        dtype=str).reshape((len(x),))


# SCIENTIFIC

def is_token_allowed_sci(token):
    '''
         Only allow valid tokens which are not stop words
         and punctuation symbols.
    '''
    if not token or not token.text.strip() or token.is_stop or token.is_punct:
        return False
    return True


def preprocesstoken_sci(token):
    # Reduce token to its lowercase lemma form
    return token.lemma_.strip().lower()


# def tokenize_sci(x):
#     try:
#         return str([preprocesstoken_sci(token) for token in nlp_sci(x) if is_token_allowed_sci(token)])
#     except:
#         return str([])


def tokenize_string_sci(x):
    return ",".join([preprocesstoken_sci(token) for token in nlp_sci(x) if is_token_allowed_sci(token)])


# def tokenize_string_sci(x):
#     string = ""
#     for token in nlp_sci(x):
#         if is_token_allowed_sci(token):
#             string = string + "," + preprocesstoken_sci(token)
#     return string


def prettify(x):
    if type(x) == list:
        x = x[0]
        return x.translate(x.maketrans("", "", "[]'""))
    elif type(x) == str:
        return x.translate(x.maketrans("", "", "[]'""))
    else:
        return ""


# df = pd.read_csv("all_filtered_linux_test.csv", delimiter=",")
# %%
def main(f):
    df_chunks = pd.read_json(path_or_buf=f"./data/livivo/documents/{f}", lines=True,
                            chunksize=1000000,
                            encoding="utf-8")  # increase Chunksize to atleast 100 000 on the VM
    # %%
    for i, df in enumerate(df_chunks):
        print(datetime.datetime.now(), "  ", i)
        cols = ['DBRECORDID', 'TITLE', 'ABSTRACT', 'LANGUAGE', 'MESH', 'CHEM', 'KEYWORDS']
        for c in cols:
        if c not in df.columns:
            df[c] = ""

        df = df.loc[:, cols]
        df.fillna('', inplace=True)

        df.loc[:, 'TITLE'] = df['TITLE'].parallel_apply(prettify)
        df.loc[:, 'ABSTRACT'] = df['ABSTRACT'].parallel_apply(prettify)
        df.loc[:, 'LANGUAGE'] = df['LANGUAGE'].parallel_apply(prettify)

        df.loc[:, 'MESH'] = df['MESH'].parallel_apply(prettify)
        df.loc[:, 'CHEM'] = df['CHEM'].parallel_apply(prettify)
        df.loc[:, 'KEYWORDS'] = df['KEYWORDS'].parallel_apply(prettify)

        # 
        german_mask = df['LANGUAGE'] == 'ger'
        else_mask = (df['LANGUAGE'] != 'ger') | (df['LANGUAGE'] == '')

        # TITLE
        df.loc[german_mask, 'TITLE_TOKENZ_GERMAN'] = df.loc[german_mask, 'TITLE'].parallel_apply(
            tokenize_string_german)
        #print("title_tokenz_german")

        df.loc[else_mask, 'TITLE_TOKENZ_SCI'] = df.loc[else_mask, 'TITLE'].parallel_apply(tokenize_string_sci)
        #print("title_tokenz_sci")

        df.drop(columns=["TITLE"], inplace=True)

        # ABSTRACT
        df.loc[german_mask, 'ABSTRACT_TOKENZ_GERMAN'] = df.loc[german_mask, 'ABSTRACT'].parallel_apply(
            tokenize_string_german)
        #print("abstract_tokenz_german")

        df.loc[else_mask, 'ABSTRACT_TOKENZ_SCI'] = df.loc[else_mask, 'ABSTRACT'].parallel_apply(
            tokenize_string_sci)
        #print("abstract_tokenz_sci")
        df.drop(columns=["ABSTRACT"], inplace=True)


        df.loc[:, 'KEYWORDS_TOKENZ'] = df['KEYWORDS'].parallel_apply(tokenize_string_sci)
        #print("keywords")
        df.drop(columns=["KEYWORDS"], inplace=True)


        df.loc[:, 'MESH_TOKENZ'] = df['MESH'].parallel_apply(tokenize_string_sci)
        #print("mesh_to")
        df.drop(columns=["MESH"], inplace=True)


        df.loc[:, 'CHEM_TOKENZ'] = df['CHEM'].parallel_apply(tokenize_string_sci)
        #print("chem_to")
        df.drop(columns=["CHEM"], inplace=True)
        
        # FOR TEST ONLY REMOVE THE BREAK!!
        # break
        # if file does not exist write header
        df.fillna("", inplace =True)
        yield df
        '''
        if not os.path.isfile('fast_concat_utf.csv'):
            df.to_csv('fast_concat_utf.csv', header=df.columns, index=False)
        else:  # else it exists so append without writing the header
            df.to_csv('fast_concat_utf.csv', mode='a', header=False, index=False)
        '''

if __name__ == '__main__':
    main()