import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def labelize(x,label_dict):
        return label_dict[x]
   
class Data:
    @staticmethod
    def load_data(path:str):
        df=pd.read_csv(path)
        print("Data Loaded, Shape : {} ,{}".format(df.shape[0],df.shape[1]))
        return df
    @staticmethod
    def prepare_data(df:pd.DataFrame,label_dict:dict,test_size=0.2,target='label'):
        
        df[target]=df[target].apply(lambda x:labelize(x,label_dict))
        datax=df.drop(columns=[target,'filename','label'])
        datay=df[target]
        trainx,testx,trainy,testy=train_test_split(datax,datay,test_size=0.2,random_state=42)
        sc=StandardScaler()
        trainx=sc.fit_transform(trainx)
        testx=sc.transform(testx)
        return [trainx,testx,trainy,testy]