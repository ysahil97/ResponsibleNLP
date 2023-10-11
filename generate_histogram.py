from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from tqdm import tqdm
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

import argparse



# question, answer = "Explain nuclear fusion like I am five", "Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants."
# inputs = tokenizer(answer, return_tensors='pt')
# score = rank_model(**inputs).logits[0].cpu().detach()
# print(score)


def get_rm_score(tokenizer,rank_model,str):
    inputs = tokenizer(str, return_tensors='pt')
    if len(tokenizer(str, return_tensors='pt').input_ids[0]) <= 512:
        score = rank_model(**inputs).logits[0].cpu().detach()[0]
        # print(type(score))
        return score
    else:
        return None


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-k", "--num_data", help="Number of datapoints to be included")
    argParser.add_argument("-d", "--dataset", help="Type of Dataset")
    args = argParser.parse_args()
    print("args=%s" % args)

    # print("args.name=%s" % args.num_data)
    # reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    reward_name = "facebook/roberta-hate-speech-dynabench-r4-target"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
    # Opening JSONL file
    with open('./fairscore/dataset/dataset/PANDA-annotated-100k-shard0.jsonl', 'r') as json_file:
        json_list = list(json_file) 
    print(type(json_list[1]))
    diff = []
    sentence_list = []
    hist_score = []
    if args.dataset == "HolisticBias":
        # pass
        with open('../sentences.csv') as csv_file:
            lines = len(csv_file.readlines())
        with open('../sentences.csv','r') as csvfile:
            reader = csv.reader(csvfile)
            i = int(args.num_data)
            for row in tqdm(reader,total = lines):
                # print(row)
                # sentence_list.append(row[0])
                hist_score.append(get_rm_score(tokenizer,rank_model,row[0]))
                i -= 1
                if i == 0:
                    break
    elif args.dataset == "FairScore":
        with open('./fairscore/dataset/dataset/PANDA-annotated-100k-shard0.jsonl', 'r') as json_file:
            json_list = list(json_file)  
        for jl in tqdm(json_list[:int(args.num_data)]):
            jla = json.loads(jl)
            sc = get_rm_score(tokenizer,rank_model,jla["rewrite"])
            if sc:
                hist_score.append(sc)
    #     diff.append(get_rm_score(tokenizer,rank_model,jla["original"]).item() - get_rm_score(tokenizer,rank_model,jla["rewrite"]).item())
    # for sentence in sentence_list:
    #     hist_score.append(get_rm_score(tokenizer,rank_model,sentence))
    # print(min(hist_score),max(hist_score))
    # print(hist_score)
    x = np.arange(len(hist_score))

    plt.hist(
        hist_score
    )
    plt.show()