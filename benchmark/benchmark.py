import pandas as pd
from dataset.load_dataset import *
from benchmark.evaluate import *

def benchmark(synthesizer,name,repeat=3):
    results = list()

    train, test, meta, categoricals, ordinals = load_dataset(name, benchmark=True)

    for iteration in range(repeat):	
        synthesized = synthesizer(train, categoricals, ordinals)
        scores = evaluate(train, test, synthesized, meta)
        scores['dataset'] = name
        scores['iter'] = iteration
        results.append(scores)

    return pd.concat(results)