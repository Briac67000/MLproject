"""
Prepare data
"""

import pandas as pd
import re 
import csv
import PyPDF2
import os
#from sklearn.model_selection import train_test_split
#import torch
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def read_excel_df(file):
    """
    read an excel file in argument and returnss a dataframe containing the data file
    """
    return pd.read_excel(file)# read the file

def export_df_to_excel(df,filepath):
    """
    arguments to pass : df is a dataframe, filepath is the name of the excel file that will be created from the dataframe provided
    """
    df.to_excel(filepath, index=False)
    
def clean_text(text):
    """
    arguments: text is a string
    cleans the text by deleting "\n" and removing specific german characters 
    returns the text cleaned
    """
    # Remplace \n et \n\n par un espace
    text = re.sub(r'\n\n|\n', ' ', text)
    # Supprime les points-virgules
    text = re.sub(r';', '', text)
    # Remplace les caractères allemands
    text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss').replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue')
    return text

def create_new_column_datesandtexts(df):
    """
    arguments: df is a dataframe 
    creates a new column in a dataframe df which will contain for each row the dates and the texts of each patient; 
    the data frame passed in argument should have for third and fourth columns the dates and texts of each patient
    returns the dataframe passed in argument with the new column "newtext"
    """
    df['newtext']=[0]*len(df) # create a new column 'newtext' which will contain dates AND texts
    
    for i in range(len(df)):
        df.loc[i,'newtext'] = ' ' + str(df.loc[i][2]) + ' ' +  str(df.loc[i][3]) # adding dates to texts in the 'newtext' column
    return df

def join_rows_same_id_in_new_df(df):
    """
    arguments: df is a dataframe
    creates a new dataframe containing all the texts refering to one patient, by merging them 
    the dataframe passed in argument should have a column "CAS_BK" which corresponds to the ID patients, and a column "newtext" which corresponds to the dates+texts of each patient
    returns the new dataframe 
    """
    df_grouped=pd.DataFrame()
    
    df_grouped = df.groupby("CAS_BK")["newtext"].apply(lambda x: " ".join(x)).reset_index()
    return df_grouped

def read_extractpages_pdf(file): 
    """
    arguments: file is the name of a pdf file in the working directory
    read a pdf file and extract all its pages in a string 
    return the extracted text as a string
    """
    pdfFileObj = open(file, 'rb')
    #print(file)
    try:
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        #print("read ok")
        num_pages = len(pdfReader.pages)
        Text = ''
        for k in range(num_pages):
            pageObj = pdfReader.pages[k]
            Text = Text + pageObj.extract_text()
    except:
        print(str(file) + " is not read")
        return(None)


    return Text
    # ne pas oublier de clean le text à la fin

def extract_chars_between_words_old(text, start_word, end_word):
    """
    arguments: text is the text of interest which is a string, start_word is a string corresponding to the beginning of the extraction string process, end_word is a string correspond to the end of the extraction string process
    extract all the characters between two words passed in argument from a text passed in argument 
    returns the string text extracted
    """
    start_index = text.find(start_word)
    if start_index == -1:
        return ""

    # Index of the end word
    end_index = text.find(end_word, start_index + len(start_word))
    if end_index == -1:
        return ""

    # extracting caracters between first and end words
    extracted_chars = ""
    for char in text[start_index + len(start_word):end_index]:
        extracted_chars += char

    return extracted_chars

def extract_chars_between_words(text, start_words, end_words):
    """
    arguments: text is the text of interest which is a string, start_word is a string corresponding to the beginning of the extraction string process, end_word is a string correspond to the end of the extraction string process
    extract all the characters between two words passed in argument from a text passed in argument
    returns the string text extracted
    """
    start_index = -1
    end_index = -1
    for start_word in start_words:
        if text.find(start_word) != -1:
            start_index = text.find(start_word)
            break
    if start_index == -1:
        return ""
    for end_word in end_words:
        if text.find(end_word, start_index + len(start_word)) != -1:
            end_index = text.find(end_word, start_index + len(start_word))
            break
    if end_index == -1:
        return ""
    # extracting caracters between first and end words
    extracted_chars = ""
    for char in text[start_index + len(start_word):end_index]:
        extracted_chars += char

    return extracted_chars

def create_list_pdf_names(df):
    """
    arguments: a dataframe containing the pdf names of interest
    creates a list of the pdf names of interest
    returns a list 
    """
    a = list(df['DATEI'])
    return a

def merge_table_id_text(df,id,text): 
    """
    arguments: df is a dataframe, id is a integer, text is a string
    the dataframe passed in argument must have two columns "ID" and "TEXT" 
    add the id of the pdf names passed in argument in the column ID of the dataframe 
    and the text corresponding to this id in the column TEXT
    returns the dataframe updated
    """
    df =pd.concat([df,pd.DataFrame({"ID": [id],"TEXT": [text]})])
    return df

def delete_postaladresses_phone_number(text,pattern):
    newtext=re.sub(pattern,"",text)
    return newtext

def extract_names(file):
    pattern=r'[^-\d.]+'
    names=re.findall(pattern,file)
    return ' '.join(names)

def replace_elements(text,list,word_replacement):
    words = re.split(r'\s|-|,', text)
    final_words=[]
    for word in words:
        if word in list:
            final_words.append(word_replacement)
        else:
            final_words.append(word)
        final_text=" ".join(final_words)
    return final_text

def main():
    os.chdir("C:\\Users\\moriceb\\Desktop\\MLproject")
    #Creation of an excel file which will contain the inputs    
    df_input=read_excel_df('C:\\Users\\moriceb\\Desktop\\MLproject\\extract_for_exercise.xlsx')
    a=create_new_column_datesandtexts(df_input)
    df_input_unique_id=join_rows_same_id_in_new_df(a)
    df_input_unique_id = df_input_unique_id.rename(columns={'CAS_BK': 'ID'})
    df_input_unique_id = df_input_unique_id.rename(columns={'newtext': 'TEXT'})
    for k in range(len(df_input_unique_id)):
        df_input_unique_id.loc[k,'TEXT']=clean_text(df_input_unique_id.loc[k,'TEXT'])
    export_df_to_excel(df_input_unique_id, 'C:\\Users\\moriceb\\Desktop\\MLproject\\input_texts.xlsx')
    
    df_matching_table = read_excel_df('C:\\Users\\moriceb\\Desktop\\MLproject\\matching_table.xlsx')
    df_summary = pd.DataFrame(columns=['ID', 'TEXT'])
    pdf_names = create_list_pdf_names(df_matching_table)
    directory_to_rename = "C:\\Users\\moriceb\\Desktop\\MLproject"  # directory containing all the pdf files
    
    regex_tel_french=r"\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())"
    regex_tel_french_international1=r"(\+33)(\s|""|.|-|())\d{1}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())"
    regex_tel_french_international2=r"(00)(\s|""|.|-)\d{2}(\s|()|""|.|-|())\d{1}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())"
    regex_tel_swiss_international1=r"(\+|"")\d{2}(\s|""|.|-|())(\s|""|.|-|())\d{2}(\s|""|.|-|())(\s|""|.|-|())\d{3}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())"
    regex_tel_swiss_international2=r"(00)(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{3}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())"
    regex_tel_swiss_national=r"\d{3}(\s|""|.|-|())\d{3}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())"
    regex_tel_german_international1=r"(00)(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())"
    regex_tel_german_international2=r"(\+|"")\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())\d{2}(\s|""|.|-|())"
    regex_adresse1 = r"\b([a-zA-ZüöäàáéèÜÖÄß]+)(Graben|graben|Stadt|stadt|-strasse|-Strasse|Str.|str.|wiesendamm|Wiesendamm|Ring|ring|strasse|weg|gasse|Strasse|Weg|Gasse|Platz|platz|Allee|allee)\s+(\d{1,}|\d{1,}[A-Za-z]|\d+\/\d+)\s+(\d{4})\s+([a-zA-ZüàöéäèáÜÖÄß\s]+)"
    regex_adresse2 = r"(Wiesendamm|wiesendamm|Stadt|stadt|Graben|graben|-strasse|-Strasse|Str.|str.|Chaussee|Chaussée|chaussée|chaussee|Promenade|promenade|Ring|ring|Chemin|chemin|Avenue|avenue|Impasse|impasse|Esplanade|esplanade|Quai|quai|Cour|cour|Passage|passage|Fbg|fbg|clos|Clos|Rue|rue|Bvd|bvd|Boulevard|boulevard|Faubourg|faubourg|allée|Allée|Hameaux|hameaux|hameau|Hameau|Allee|allee|Route|route|strasse|weg|gasse|Strasse|Weg|Gasse|Platz|platz)([a-zA-ZüöäàáéèÜÖÄß\s]+)\s+(\d{1,}|\d{1,}[A-Za-z]|\d+\/\d+)\s+(\d{4})\s+([a-zA-ZüàöéäèáÜÖÄß\s]+)"
    regex_all = regex_tel_french + '|'+ regex_tel_french_international1 + '|' + regex_tel_french_international2 + '|' + regex_tel_swiss_international1 + '|' + regex_tel_swiss_international2 + '|' + regex_tel_swiss_national + '|' + regex_tel_german_international1 + '|' + regex_tel_german_international2 + '|' + regex_adresse1  + '|' + regex_adresse2
  

    for filename in os.listdir(directory_to_rename):
        id = os.path.splitext(filename)[0] + str('.pdf')  # Extraction of the name of the file
        if id in pdf_names:  # a est la sortie de create_list_pdf_names
            names = extract_names(os.path.splitext(filename)[0])
            list_names = names.split()
            Text = read_extractpages_pdf(id)

            if Text is not None:
                Text =clean_text(Text)
                Text =extract_chars_between_words(Text, ["Verla uf:","Ver lauf:","V erlauf:","Ve rlauf:","Verl auf:","Verlau f:","Verlauf:","Verlauf :", "verlauf:","verlauf :","ver lauf:","v erlauf:","ve rlauf:","verl auf:","verlau f:","verlauf:"], [ "Procedere:", "Proce dere:","Proc edere:","Pro cedere:","Pr ocedere:","P rocedere:", "Proced ere:","Procede re:","Proceder e:","Procedere :","Mit freundlichen Gruessen","Freundliche Gruesse"]) # Hospitalisationsverlauf: / Text on 2 pages ? /
                Text=delete_postaladresses_phone_number(Text,regex_all)

                #if Text1 !=Text2 :
                    #words_text1=set(Text1.split())
                    #words_text2=set(Text2.split())
                    #unique_words_text1= words_text1 - words_text2
                    #print(unique_words_text1)
                Text=replace_elements(Text,list_names,"SUPERMAN")
                df_summary = merge_table_id_text(df_summary, int(
                    df_matching_table[df_matching_table['DATEI'] == id]['Fallidentifikator (FID)'].iloc[0]),Text)  # pour avoir l'id correspondant à un fichier pdf grâce à la matching table, on peut utiliser cette commande au lieu d'un dictionnaire: df3[df3['DATEI']==id]['Fallidentifikator (FID)'], df3 la matching table;
    export_df_to_excel(df_summary, 'C:\\Users\\moriceb\\Desktop\\MLproject\\summary_reports2.xlsx')

if __name__ == "__main__":
    main()

#df_summary =pd.concat([df_summary,pd.DataFrame({"ID": [87759460,88064502,88080535,88091908,88102007],"TEXT": ['bonjour','fgf','gfgg','fdbghfdg','dfggr']})])

# #deleting the lines in the input dataframe that don't have summaries in the groundtruth dataframe
# id_set = list(set(df_summary['ID']).intersection(df_input_unique_id['ID']))

# #df_input_unique_id = df_input_unique_id[df_input_unique_id['ID'].isin(id_set)]
# df_summary=df_summary[df_summary['ID'].isin(id_set)]
# # creating, for each dataframe INPUT and GROUNDTRUTH, two dataframes train and test //
# # each train (test) dataframe, i.e input_train_df (input_test_df) and summary_train_df (summary_test_df), has the same id as the other one train ( test )  dataframe
# id_train, id_test = train_test_split(list(id_set), test_size=0.2)
# df_input_unique_id_train =df_input_unique_id[df_input_unique_id['ID'].isin(id_train)]
# df_summary_train = df_summary[df_summary['ID'].isin(id_train)]
# df_input_unique_id_test =df_input_unique_id[df_input_unique_id['ID'].isin(id_test)]
# df_summary_test = df_summary[df_summary['ID'].isin(id_test)]
#
## reorganize the dataframe df_summary_train(df_summary_test) using the order of the dataframe df_input_unique_id_train(df_input_unique_id_test)
#df_summary_train = df_summary_train.set_index('ID').reindex(df_input_unique_id_train['ID']).reset_index()
#df_summary_test = df_summary_test.set_index('ID').reindex(df_input_unique_id_test['ID']).reset_index()
#
# tokenizer = AutoTokenizer.from_pretrained("Einmalumdiewelt/T5-Base_GNAD")
#
#
# model = AutoModelForSeq2SeqLM.from_pretrained("Einmalumdiewelt/T5-Base_GNAD")
#
# input_text = df_input_unique_id.iloc[15][1]
#
# # Encodage des textes d'entrée et de sortie du décodeur
# inputs = tokenizer.encode_plus(input_text, return_tensors="pt", max_length=512,  padding="longest")
#
#
# encoder_inputs = inputs['input_ids']
#
# # Prédictions du modèle en utilisant encoder_inputs
# outputs = model.generate(input_ids=encoder_inputs, max_length=150, num_beams=4, early_stopping=True)
# translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("Summary:", translated_text)


