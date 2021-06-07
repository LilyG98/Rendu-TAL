#général: 
import spacy
import fr_core_news_md
import pandas as pd
import numpy as np
import re

#Importation du pretrained model en français
nlp = spacy.load("fr_core_news_md")

#conversion du df en JSONL compatible avec Doccano 
import codecs
import pandas
import json 

#pour l'entraînementde fr_core_news_md avec les nouvelles annotations
from pathlib import Path
from spacy.util import minibatch, compounding 
from spacy.training import Example

#Visualisation du cross validation : 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
np.random.seed(1338)


def list_ents_by_label(doc,label):
    """ 
    input : 1) une variable doc (type = spacy.tokens.doc.Doc)
            2) le nom du label en format str. Ex : 'LOC'
    
    output : la liste des entités d'un doc pour le label souhaité"""
    list_entbylabel = list(set([e.text for e in doc.ents if e.label_ == label]))
    return list_entbylabel

def dicoKfolds_to_doccano(dico):
    output_pathmodel = "./DOCCANO/doccano-input/kfolds/"
    keys_list = list(dico)
    for n in range(len(dico)):
        key = keys_list[n]
        output_path = output_pathmodel+keys_list[n]+".jsonl"
        
        for text in dico[keys_list[n]]:   # remplace 'full_text'  le nom de la colonne qui a les textes
            
            doc = nlp(str(text))
            labels = []
            if len(doc.ents) > 0:
                for ent in doc.ents:
                    labels.append([ent.start_char,
                                  ent.end_char,
                                  ent.label_])
            if len(labels) > 0:
                sentence = {'text': text, 'labels': labels}
            else:
                sentence = {'text': text}
            
            json_string = json.dumps(sentence, ensure_ascii=False)
            with codecs.open(output_path, 'a', encoding='utf8') as f:
                f.write(json_string)
                f.write('\n')
        print("File",keys_list[n], "created!")
              
              
def pandas_to_doccano(df, nlp, output_path):
    """ Conversion du df en JSONL compatible avec Doccano 
    input : 1 - le dataframe avec l'ensemble des titres d'article à annoter
            2 - la pipeline nlp voulue
            3 - l'output_path pour sotcker le fichier qui sera ensuite importé dans Doccano
    
    output : un document .jsonl enregistré dans le output_path"""
    
    with codecs.open(output_path, 'w', encoding='utf8') as f:
    
        for text in df.article_titre:   # remplace 'full_text'  le nom de la colonne qui a les textes

            doc = nlp(text)
            labels = []
            if len(doc.ents) > 0:
                for ent in doc.ents:
                    labels.append([ent.start_char,
                                  ent.end_char,
                                  ent.label_])
            if len(labels) > 0:
                sentence = {'text': text, 'labels': labels}
            else:
                sentence = {'text': text}
            json_string = json.dumps(sentence, ensure_ascii=False)
            f.write(json_string)
            f.write('\n')
    print('File created!')

    
def visualize_groups(classes, groups, name):
    # Visualize dataset groups
    fig, ax = plt.subplots()
    ax.scatter(range(len(groups)),  [.5] * len(groups), c=groups, marker='_',
               lw=50, cmap=cmap_data)
    ax.scatter(range(len(groups)),  [3.5] * len(groups), c=classes, marker='_',
               lw=50, cmap=cmap_data)
    ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
           yticklabels=['Data\ngroup', 'Data\nclass'], xlabel="Sample index")


def train_data(json_doc_path):
    """ Permet de transformer le document jsonl en liste avec les annotations.
        input: le path vers le jsonl annoté sur doccano
        output : une liste transcrivant les annotations du JSONL"""
    TRAIN_DATA = []

    with codecs.open(json_doc_path, "r", encoding="utf8") as f:
        lines = f.readlines()
    #     print(lines)
        for line in lines:
            line = json.loads(line)
    #         print(line)
            if "labels" in line:
                line["entities"] = line.pop("labels")
            else:
                line["entities"] = []
            ents = []
            for entity in line["entities"]:
                ents.append((entity[0], entity[1], entity[2]))
            TRAIN_DATA.append({"text": line["text"], "entities": ents})
            #json_string = json.dumps({"text": line["text"], "entities": ents}, ensure_ascii=False)
    f.close()
    return TRAIN_DATA


def train_nlp (nom_output_nlp,TRAIN_DATA):
    """ Entraîner le modèle ner par défaut de Spacy (ici fr_core_news_md). Modifie la chaîne de départ par itération dunouveau modèle ner.
    input : 1 - le nom de l'output que l'on souhaite donner au nom du dir qui contiendra la pipeline nlp entraînée
            2 - la liste TRAIN_DATA définit avec la fonction train_data
            
    output : Pas d'output à proprement parler mais modification intrinsèque du ner par défaut de Spacy"""
    
    #Etape1 : ajout des entités annotées dans le ner de nlp
    
    ner = nlp.get_pipe("ner") #On s'occupe uniquement de la pipe ner de nlp.

    for annotations in TRAIN_DATA:
    #print(annotations)
        for ent in annotations["entities"]:
            ner.add_label(ent[2])
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    #Etape 2 : entraînement par itération du nlp avec les annotations de TRAIN_DATA
    
    with nlp.disable_pipes(*other_pipes):  #On s'occupe uniquement de la pipe ner de nlp.
        
        for itn in range(3):
            print("iteration: "+str(itn))
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                examples = []
                for ba in batch:
                    examples.append(Example.from_dict(nlp.make_doc(ba["text"]), ba))
                    nlp.update(examples)        
    print("training is finished")
    
    output_dir = Path("./Notebook_data/Training_Doccano/trained_nlp/"+nom_output_nlp)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
    

def test_set_new_NLP(newNLP_output_dir,test_set):
    print("Loading from", newNLP_output_dir)
    nlp_test = spacy.load(newNLP_output_dir)
    
    doc_test = nlp_test(test_set)
    for ent in doc_test.ents:
        print(ent.label_, ent.text)
        #if("Variétés" in ent.text):
        #    print(ent.label_, ent.text)



        
        
        
        
        
        
        
        
        
        
        
        
        