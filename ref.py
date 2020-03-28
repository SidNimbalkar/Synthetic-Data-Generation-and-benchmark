import pandas as pd
import numpy as np
import streamlit as st
import json
import requests
from dataset.process_dataset import *
from dataset.load_dataset import *
from benchmark.benchmark import *

path = st.text_input('Provide path to fetch data')

if path:
    df = get_data(path)
    st.subheader('Data')
    st.dataframe(df)
    continuous_columns,categorical_columns,ordinal_columns,columns = get_columns(df)
    continuous_columns = st.multiselect("Continuous Columns (update if required)", df.columns.tolist(),default=continuous_columns)
    categorical_columns = st.multiselect("Categorical Columns (update if required)", df.columns.tolist(), default=categorical_columns)
    ordinal_columns = st.multiselect("Ordinal Columns", df.columns.tolist(), default=ordinal_columns)
    column_type = get_column_type_list(continuous_columns,categorical_columns,ordinal_columns,columns,df)
    meta = get_metadata(column_type,df)
    tdata = project_table(df, meta)
    filename = st.text_input("Provide a file name to save metadata and npz file")
    if filename:
        prep_npz_file(tdata,filename,df)
        problem_type = st.radio("Problem Type", ['binary_classification', 'multiclass_classification', 'regression','gaussian_likelihood','bayesian_likelihood'])  
        if problem_type :
            prep_meta_file(meta,problem_type,filename)
            synthesizer = st.multiselect("Choose one or more synthesizers",['clbn','ctgan','identity','independent','medgan','tablegan','tvae','uniform','veegan'])
            if synthesizer:
                req = {'data': str(filename), 'synthesizer': synthesizer}
                url = 'http://127.0.0.1:5000/benchmark'
                response = (requests.post(url, json=req)).text
                print(response)
                res_dic = json.loads(response)
                for i in synthesizer:
                    synthetic_data = res_dic['synthetic'][str(i)]
                    benchmark_pd = pd.DataFrame(res_dic['score'][str(i)])
                    st.subheader('Synthetic Data'+ str(i))
                    st.dataframe(synthetic_data)
                    st.subheader('Benchmark Scores'+ str(i))
                    st.dataframe(benchmark_pd)
            else: 
                st.write('Something went wrong')	
    else:
        st.write('Please provide a name for the files')
    #synthesizer = st.multiselect("Choose a synthesizer",['clbn','ctgan','identity','independent','medgan','tablegan','tvae','uniform','veegan'])


else:
    st.write('Path not found or entered')