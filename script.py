import pandas as pd

import _pickle as cPickle


# load it again
with open('my_dumped_classifier.pkl', 'rb') as fid:
    text_clf = cPickle.load(fid)


# print('ok')
def predict_from_sentence(sentence):
    sent = pd.Series(sentence)
    prediction = text_clf.predict(sent)
    if prediction[0] == 0:
        return 'заблокировать'
    else:
        return 'разрешить'

print(predict_from_sentence('Привет, как дела?'))
print(predict_from_sentence('Ненавижу тебя, гад'))