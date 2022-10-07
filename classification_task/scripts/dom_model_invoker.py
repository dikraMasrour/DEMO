import pickle


FILE_DOM = '..\classification_task\models\logr_domain_classification_OS_90.sav'
DOM_MODEL = pickle.load(open(FILE_DOM, 'rb'))


def domain_classification(vect_ntb):
    return DOM_MODEL.predict(vect_ntb)[0]
    


