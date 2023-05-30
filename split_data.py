from preprocessing import *
from sklearn.model_selection import train_test_split
import pandas as pd
import re 
import csv
import PyPDF2
import os

df_summary=read_excel_df('C:\\Users\\moriceb\\Desktop\\MLproject\\summary_reports2.xlsx')
df_input_unique_id=read_excel_df('C:\\Users\\moriceb\\Desktop\\MLproject\\input_texts.xlsx')

# # deleting the lines in the input dataframe that don't have summaries in the groundtruth dataframe
id_set = list(set(df_summary['ID']).intersection(df_input_unique_id['ID']))
df_input_unique_id = df_input_unique_id[df_input_unique_id['ID'].isin(id_set)]
df_summary=df_summary[df_summary['ID'].isin(id_set)]
# # creating, for each dataframe INPUT and GROUNDTRUTH, two dataframes train and test //
# # each train (test) dataframe, i.e input_train_df (input_test_df) and summary_train_df (summary_test_df), has the same id as the other one train ( test )  dataframe
id_train, id_test = train_test_split(list(id_set), test_size=0.2)
df_input_unique_id_train =df_input_unique_id[df_input_unique_id['ID'].isin(id_train)]
df_summary_train = df_summary[df_summary['ID'].isin(id_train)]
df_input_unique_id_test =df_input_unique_id[df_input_unique_id['ID'].isin(id_test)]
df_summary_test = df_summary[df_summary['ID'].isin(id_test)]

df_summary_train = df_summary_train.set_index('ID').reindex(df_input_unique_id_train['ID']).reset_index()
df_summary_test = df_summary_test.set_index('ID').reindex(df_input_unique_id_test['ID']).reset_index()

export_df_to_excel(df_summary_train, 'C:\\Users\\moriceb\\Desktop\\MLproject\\summary_train_group.xlsx')
export_df_to_excel(df_summary_test, 'C:\\Users\\moriceb\\Desktop\\MLproject\\summary_test_group.xlsx')
export_df_to_excel(df_input_unique_id_train, 'C:\\Users\\moriceb\\Desktop\\MLproject\\input_train_group.xlsx')
export_df_to_excel(df_input_unique_id_test, 'C:\\Users\\moriceb\\Desktop\\MLproject\\input_test_group.xlsx')
 
if __name__ == "__main__":
    main()