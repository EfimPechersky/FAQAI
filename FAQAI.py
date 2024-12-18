print("Importing librarys")
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import tensorflow as tf
import os
print("Program started")
# Define the model repo
model_name = "DeepPavlov/rubert-base-cased"


# Download pytorch model
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
bertmodel = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
print("Bert loaded")

stop_words=[]
with open("data/stop-ru.txt","r",encoding='UTF8') as file:
  stop_words=file.read().split('\n')

def filter_text(x):
  new_text=""
  for i in x.lower().split(" "):
    if i not in stop_words:
      new_text+=i+" "
  new_text=new_text[:-1]
  if new_text[-1] in ".?":
    new_text=new_text[:-1]
  return new_text

questions=[]
with open("data/questions.txt","r",encoding='UTF8') as file:
  questions=list(map(filter_text,file.read().split('\n')))
show_answers=[]
with open("data/answers.txt","r",encoding='UTF8') as file:
  show_answers=file.read().split('\n')
answers=[]
with open("data/answers.txt","r",encoding='UTF8') as file:
  answers=list(map(filter_text,file.read().split('\n')))

print("Files loaded")
dataset=[]
for i in range(len(questions)):
  q_emb=np.array(list(map(float,list(bertmodel(**tokenizer(questions[i], return_tensors="pt")).pooler_output[0]))))
  a_emb=np.array(list(map(float,list(bertmodel(**tokenizer(answers[i], return_tensors="pt")).pooler_output[0]))))
  dataset.append([np.array(q_emb),np.array(a_emb)])
dataset=np.array(dataset)

X,Y=[],[]
for i in range(dataset.shape[0]):
  for j in range(dataset.shape[0]):
    X.append(np.concatenate([dataset[i,0,:],dataset[j,1,:]],axis=0))
    if i==j:
      Y.append(1)
    else:
      Y.append(0)
X=np.array(X)
Y=np.array(Y)
print("Dataset loaded")
if os.path.isfile("model/FAQ.keras"):
  model=tf.keras.models.load_model("model/FAQ.keras")
else:
  model=tf.keras.models.Sequential()
  model.add(tf.keras.layers.InputLayer(input_shape=(1536,)))
  model.add(tf.keras.layers.Dense(300,activation='selu'))
  model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
  es = tf.keras.callbacks.EarlyStopping(monitor='auc', mode='max', patience=10, restore_best_weights=True)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(curve='pr', name='auc')])
  model.fit(X,Y,epochs=1000, callbacks=[es], class_weight={0:1, 1:np.sqrt(Y.shape[0])})
  model.fit(X,Y,epochs=1000, class_weight={0:1, 1:np.sqrt(Y.shape[0])-1})
  model.summary()
  model.save("model/FAQ.keras")

print("Введите вопрос на русском языке или exit что-бы выключить программу")
inpq=input()
while inpq!="exit":
  q1=np.array(list(map(float,list(bertmodel(**tokenizer(filter_text(inpq), return_tensors="pt")).pooler_output[0]))))
  x_test=[]
  for i in range(dataset.shape[0]):
    x_test.append(np.concatenate([q1,dataset[i,1,:]],axis=0))
  x_test=np.array(x_test)
  ans=show_answers[np.argmax(model.predict(x_test))]
  print(ans)
  inpq=input()

