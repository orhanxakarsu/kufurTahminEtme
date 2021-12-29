import numpy as np
import pandas as pd
from  sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,GRU,Embedding,CuDNNGRU
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from  sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.models import load_model


df1 = pd.read_excel("python.xlsx")
df2 = pd.read_excel("python2.xlsx")
df3 = pd.read_excel("python3.xlsx")
df4 = pd.read_excel("python4.xlsx")
df5 = pd.read_excel("python5.xlsx")
df6 = pd.read_excel("Kitap 2.xlsx")
kufursuz_veri = pd.read_csv("hepsiburada.csv")

kufurlu_X=0
kufursuz2_X=0
dataset1=df6[df6["Skorlar"]==0]
kufurlu_X=dataset1["Cumleler"].values
dataset2=df6[df6["Skorlar"]==1]
kufursuz2_X=dataset2["Cumleler"].values
kufursuz2_X=np.append(kufursuz2_X,kufursuz2_X)
kufursuz2_X=np.append(kufursuz2_X,kufursuz2_X)
kufursuz2_y=np.ones(kufursuz2_X.size)
kufurlu_X=np.append(kufurlu_X,df1.values)
kufurlu_X=np.append(kufurlu_X,df1.values)
kufurlu_X=np.append(kufurlu_X,df2.values)
kufurlu_X=np.append(kufurlu_X,df3.values)
kufurlu_X=np.append(kufurlu_X,df4.values)
kufurlu_X=np.append(kufurlu_X,df5.values)
kufurlu_X=np.append(kufurlu_X,kufurlu_X)




kufurlu_y=np.zeros(kufurlu_X.size)

kufursuz_dataset = kufursuz_veri[kufursuz_veri["Rating"]==1]
kufursuz_dataset=kufursuz_dataset[:30000]
kufursuz_X=kufursuz_dataset["Review"].values
kufursuz_X1=np.append(kufursuz_X,kufursuz2_X)
kufursuz_y =np.ones(kufursuz_X.size)

kufursuz_y1=np.append(kufursuz_y,kufursuz2_y)

X=np.append(kufursuz_X1,kufurlu_X)
y=np.append(kufursuz_y1,kufurlu_y)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=30)


total_data=X.tolist()
X_train =X_train.tolist()
X_test=X_test.tolist()

num_words=20000
tokenizer =Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(total_data)

X_train_tokens=tokenizer.texts_to_sequences(X_train)
X_test_tokens=tokenizer.texts_to_sequences(X_test)

num_tokens=[len(tokens) for tokens in X_train_tokens + X_test_tokens]
num_tokens=np.array(num_tokens)
max_tokens= np.mean(num_tokens)+ np.std(num_tokens)*2
max_tokens=int(max_tokens)

X_train_pad =pad_sequences(X_train_tokens,maxlen=max_tokens)
X_test_pad =pad_sequences(X_test_tokens,maxlen=max_tokens)

models = Sequential()
embedding_size = 50
models.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name="embedding_layer"))


models.add(GRU(units=8,return_sequences=True))
models.add(GRU(units=8,return_sequences=False))


models.add(Dense(1,activation="sigmoid"))

models.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
               metrics=["accuracy"])

models.fit(X_train_pad,y_train,epochs=3,batch_size=200)

def tahminskoru(cumle):
    tokens=tokenizer.texts_to_sequences(cumle)
    tokens_pad = pad_sequences(tokens,maxlen=max_tokens)
    tahmin =models.predict(tokens_pad)
    print(tahmin)
    

while True:
    cumle =input("(Çıkmak için 1'e bas. \n')Bir cümle gir ->>")
    if cumle =="1":
        break
    else:
        cumle=[cumle]
        tahminskoru(cumle)
import json

tokenizer_json = tokenizer.to_json()
with open("models"+'_tokenizer.json', 'w', encoding='utf-8') as f: 
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



