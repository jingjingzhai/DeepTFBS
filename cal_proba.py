def test_classifier(seqtest, models):
    """function to test the vector-k-mer model, it uses the w2v output (models) and bayes inversion
    Parameters
    ----------
    seqtest: `pd.DataFrame`
        A list of regions, and each region is a list of sentences
    models: `list`
        Each list is a gensim.models.Word2Vec (each potential class)

    Returns
    -------
    prob: `pd.DataFrame`
    	A dataframe with five columns
    	"category_0": `float` The probability for the first model
    	"category_1": `float The probability for the second model
    	"true_category": `int`, 0 for the first model or 1 for the second model
    	"predict": `int`, 0 for the first model or 1 for the second model
    	"predict_proba": `float The probability for the winner model
    """
    import pandas as pd
    import numpy as np
    docs = [r['range_sentences'] for r in seqtest]
    docs_cats = pd.Series([r['range_category'] for r in seqtest])
    sentlist = [s for d in docs for s in d]
    llhd = np.array([m.score(sentlist, len(sentlist)) for m in models])
    lhd = np.exp(llhd - llhd.max(axis=0))
    prob = pd.DataFrame((lhd / lhd.sum(axis=0)).transpose())
    prob["seq"] = [i for i, d in enumerate(docs) for s in d]
    prob = prob.groupby("seq").mean()
    prob['true_category'] = docs_cats.values
    prob['predict'] = np.where(prob[1] <= prob[0], 0, 1)
    prob['predict_proba'] = np.where(prob[1] <= prob[0], prob[0], prob[1])
    prob.columns = ["category_0", "category_1", "true_category", "predict", "predict_proba"]
    return prob

import gensim
import logging
import numpy as np
import os
import pandas as pd
model1 = gensim.models.Word2Vec.load("/home/malab14/research/codon_usage/00word2vec_results/puccinia/puccunia_model.h5")
model2 = gensim.models.Word2Vec.load("/home/malab14/research/codon_usage/00word2vec_results/arabidopsis/arabidopsis_model.h5")

models = [model1, model2]
seqtest = pd.read_table("~/Desktop/tmp.txt", header=None)
seqtest = seqtest.T
for i in seqtest:
    print i
