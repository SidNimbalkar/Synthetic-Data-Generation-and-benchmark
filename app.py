# Using flask to make an api
from flask import Flask, jsonify, request
import json
import pandas as pd
import pickle
import os
from dataset.load_dataset import *
from benchmark.evaluate import *
from syn.identity import IdentitySynthesizer
from syn.clbn import CLBNSynthesizer
from syn.ctgan import CTGANSynthesizer
from syn.independent import IndependentSynthesizer
from syn.medgan import MedganSynthesizer
from syn.privbn import PrivBNSynthesizer
from syn.tablegan import TableganSynthesizer
from syn.tvae import TVAESynthesizer
from syn.uniform import UniformSynthesizer
from syn.veegan import VEEGANSynthesizer
from benchmark.benchmark import *


# creating a Flask app
app = Flask(__name__)

# on the terminal type: curl http://127.0.0.1:5000/
@app.route('/', methods = ['GET','POST'])
def home():
    if(request.method == 'POST' or request.method == 'GET'):

        data = "hello world"
        return jsonify({'data': data})


@app.route('/benchmark', methods = ['POST', 'GET']) #/result route, allowed request methods; POST, and GET
def predict():
    dataset = ''
    synthesizer = ''
    if request.method == 'POST':
        synthesizer_l = []

        text = request.get_data()#as_text=True
        text_dic = json.loads(text)
        #text_dic=text
        script_dir = os.path.dirname(__file__)
        dictionary = {}
        dictionary['score'] = {}
        dictionary['synthetic'] = {}
        dataset = text_dic['data']
        synthesizer_l = text_dic['synthesizer']
        for synthesizer in synthesizer_l:
            print (synthesizer)
            if synthesizer.lower() == 'identity':
                #model = pickle.load(open('identitymodel.pickle','rb'))
                DEFAULT_DATASETS = [dataset]
                data, categorical_columns, ordinal_columns = load_dataset(dataset)
                synthesizer = IdentitySynthesizer()
                synthesizer.fit(data, categorical_columns, ordinal_columns)
                sampled = synthesizer.sample(3)
                #dictionary.update(benchmark(synthesizer.fit_sample,  datasets=DEFAULT_DATASETS, repeat = 1).to_dict())
                dictionary['score']['identity'] = benchmark(synthesizer.fit_sample, str(dataset), repeat = 3).to_dict('records')
                dictionary['synthetic']['identity'] = sampled.tolist()

                #return (dictionary)

            elif synthesizer.lower() == 'independent':
                #model = pickle.load(open('independentmodel.pickle','rb'))
                DEFAULT_DATASETS = [dataset]
                data, categorical_columns, ordinal_columns = load_dataset(dataset)
                synthesizer = IndependentSynthesizer()
                synthesizer.fit(data, categorical_columns, ordinal_columns)
                sampled = synthesizer.sample(3)
                #dictionary.update(benchmark(synthesizer.fit_sample,  datasets=DEFAULT_DATASETS, repeat = 1).to_dict())
                dictionary['score']['independent'] = benchmark(synthesizer.fit_sample,  str(dataset), repeat = 3).to_dict('records')
                dictionary['synthetic']['independent'] = sampled.tolist()

                #return (dictionary)

            elif synthesizer.lower() == 'clbn':

                #model = pickle.load(open('clbnmodel.pickle','rb'))
                DEFAULT_DATASETS = [dataset]
                data, categorical_columns, ordinal_columns = load_dataset(dataset)
                synthesizer = CLBNSynthesizer()
                synthesizer.fit(data, categorical_columns, ordinal_columns)
                sampled = synthesizer.sample(3)
                #dictionary = benchmark(synthesizer.fit_sample,  datasets=DEFAULT_DATASETS, repeat = 1).to_dict()
                dictionary['score']['clbn'] = benchmark(synthesizer.fit_sample,  str(dataset), repeat = 3).to_dict('records')
                dictionary['synthetic']['clbn'] = sampled.tolist()

                #return (dictionary)

            elif synthesizer.lower() == 'ctgan':

                #model = pickle.load(open('ctganmodel.pickle','rb'))
                DEFAULT_DATASETS = [dataset]
                data, categorical_columns, ordinal_columns = load_dataset(dataset)
                synthesizer = CTGANSynthesizer()
                synthesizer.fit(data, categorical_columns, ordinal_columns)
                sampled = synthesizer.sample(3)
                #dictionary = benchmark(synthesizer.fit_sample,  datasets=DEFAULT_DATASETS, repeat = 1).to_dict()
                dictionary['score']['ctgan'] = benchmark(synthesizer.fit_sample,  str(dataset), repeat = 3).to_dict('records')
                dictionary['synthetic']['ctgan'] = sampled.tolist()

                #return (dictionary)

            elif synthesizer.lower() == 'medgan':

                #model = pickle.load(open('medganmodel.pickle','rb'))
                DEFAULT_DATASETS = [dataset]
                data, categorical_columns, ordinal_columns = load_dataset(dataset)
                synthesizer = MedganSynthesizer()
                synthesizer.fit(data, categorical_columns, ordinal_columns)
                sampled = synthesizer.sample(3)
                #dictionary = benchmark(synthesizer.fit_sample,  datasets=DEFAULT_DATASETS, repeat = 1).to_dict()
                dictionary['score']['medgan'] = benchmark(synthesizer.fit_sample,  str(dataset), repeat = 3).to_dict('records')
                dictionary['synthetic']['medgan'] = sampled.tolist()

                #return (dictionary)

            elif synthesizer.lower() == 'tablegan':

                #model = pickle.load(open('tablegansmodel.pickle','rb'))
                DEFAULT_DATASETS = [dataset]
                data, categorical_columns, ordinal_columns = load_dataset(dataset)
                synthesizer = TableganSynthesizer()
                synthesizer.fit(data, categorical_columns, ordinal_columns)
                sampled = synthesizer.sample(3)
                #dictionary = benchmark(synthesizer.fit_sample,  datasets=DEFAULT_DATASETS, repeat = 1).to_dict()
                dictionary['score']['tablegan'] = benchmark(synthesizer.fit_sample,  str(dataset), repeat = 3).to_dict('records')
                dictionary['synthetic']['tablegan'] = sampled.tolist()

                #return (dictionary)

            elif synthesizer.lower() == 'tvae':

                #model = pickle.load(open('tvaemodel.pickle','rb'))
                DEFAULT_DATASETS = [dataset]
                data, categorical_columns, ordinal_columns = load_dataset(dataset)
                synthesizer = TVAESynthesizer()
                synthesizer.fit(data, categorical_columns, ordinal_columns)
                sampled = synthesizer.sample(3)
                #dictionary = benchmark(synthesizer.fit_sample,  datasets=DEFAULT_DATASETS, repeat = 1).to_dict()
                dictionary['score']['tvae'] = benchmark(synthesizer.fit_sample,  str(dataset), repeat = 3).to_dict('records')
                dictionary['synthetic']['tvae'] = sampled.tolist()

                #return (dictionary)

            elif synthesizer.lower() == 'uniform':

                #model = pickle.load(open('uniformmodel.pickle','rb'))
                DEFAULT_DATASETS = [dataset]
                data, categorical_columns, ordinal_columns = load_dataset(dataset)
                synthesizer = UniformSynthesizer()
                synthesizer.fit(data, categorical_columns, ordinal_columns)
                sampled = synthesizer.sample(3)
                #dictionary = benchmark(synthesizer.fit_sample,  datasets=DEFAULT_DATASETS, repeat = 1).to_dict()
                dictionary['score']['uniform'] = benchmark(synthesizer.fit_sample,  str(dataset), repeat = 3).to_dict('records')
                dictionary['synthetic']['uniform'] = sampled.tolist()

                #return (dictionary)

            elif synthesizer.lower() == 'veegan':

                #model = pickle.load(open('veeganmodel.pickle','rb'))
                DEFAULT_DATASETS = [dataset]
                data, categorical_columns, ordinal_columns = load_dataset(dataset)
                synthesizer = VEEGANSynthesizer()
                synthesizer.fit(data, categorical_columns, ordinal_columns)
                sampled = synthesizer.sample(3)
                #dictionary = benchmark(synthesizer.fit_sample,  datasets=DEFAULT_DATASETS, repeat = 1).to_dict()
                dictionary['score']['veegan'] = benchmark(synthesizer.fit_sample,  str(dataset), repeat = 3).to_dict('records')
                dictionary['synthetic']['veegan'] = sampled.tolist()

                #return (dictionary)

            else:
                return ("Invalid Input")

        return (dictionary)


# driver function
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,host='0.0.0.0',port=port)
