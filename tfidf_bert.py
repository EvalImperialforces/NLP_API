from tika import parser # pip install tika
from tfidf import tfidf
import re
import torch
import json

##### Load Model #####

model_path = "drive/MyDrive/Colab Notebooks/bertQA_model"
tokenizer_path = "drive/MyDrive/Colab Notebooks/bertQA_tokenizer"

model = torch.load(model_path)
tokenizer = torch.load(tokenizer_path)

##### Split Document #####

raw = parser.from_file('resources/Enterprise Support Reporting Cockpit Cloud (1).pdf')

rx = re.compile(r'''
    ^
    (?:Section\ )?\d+\.\d+
    [\s\S]*?
    (?=^(?:Section\ )?\d+\.\d+|\Z)

    ''', re.VERBOSE | re.MULTILINE)

parts = [match.group(0) for match in rx.finditer(raw['content'])]
parts2 = [i.replace('\n', ' ') for i in parts] #remove\n

# Remove content table
part_noctable = parts2[83:]

##### BERT ####

def bert(question, text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, text)

    # Report how long the input sequence is.
    #print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    print(answer)

    #return json.loads("' + answer + '")

def bert_answers(question, no_answers):
    # Insert question - tf-idf operation performed first and top n bert answers will be returned
    pars = tfidf(question, part_noctable)
 
    # Select TF-IDF results above threshold.
    top_results = pars.loc[pars['Match Percentage'] >= 10]
    top_results = top_results['Paragraph']
  
    # Crop results based on those specified in no_answers, unless top_results in less than number specified
    if len(top_results) > no_answers:
        crop_results = top_results.head(no_answers)
    else:
        crop_results = top_results


    #print(crop_results)
    for index, row in crop_results.iteritems():
        print("Bert Answer {}".format(index+1))
        print(row)
        print(bert(question, row))


#### Test ####
question = "How do customers get licence information?"

bert_answers(question, 5)