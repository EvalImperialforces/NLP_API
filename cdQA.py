import os
from ast import literal_eval
import pandas as pd

from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline

df = pd.read_csv('esrc_pdfs.csv', converters={'paragraphs': literal_eval})

cdqa_pipeline = QAPipeline(reader='/resources/cdQA/bert_qa.joblib') # use 'distilbert_qa.joblib' for DistilBERT instead of BERT
cdqa_pipeline.fit_retriever(df=df) # should this be fit_reader???

cdqa_pipeline.dump_reader('/resources/cdQA/bert-reader.joblib')

prediction = cdqa_pipeline.predict(query, n_predictions=5)

def make_prediction(query, n_predictions):

    prediction = cdqa_pipeline.predict(query, n_predictions=n_predictions)

    return prediction


