import os
import json
import pypandoc
import requests
import pandas as pd 
import numpy as np
from bs4 import BeautifulSoup
from markdown import markdown
from sentence_transformers import SentenceTransformer
import arxiv
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
# os.system("somef describe -r https://github.com/dgarijo/Widoco/ -o test.json -t 0.8")

#DEFINITIVO:
# https://github.com/facebookresearch/encodec sip 
# https://github.com/huggingface/transformers sip 
# https://github.com/WongKinYiu/yolov7 sip
# https://github.com/openai/sparse_attention sip 
# https://github.com/jadore801120/attention-is-all-you-need-pytorch sip
# https://github.com/extreme-bert/extreme-bert sip
# https://github.com/fengres/mixvoxels sip
# https://github.com/shi-labs/versatile-diffusion sip 
# https://github.com/shoufachen/diffusiondet ?

sentences = ["I ate dinner", "I love pasta"]

def cosine(u,v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def obtenerJason (url,nombre):
  orden1="somef describe -r "
  orden2=" -o "
  orden3=" -t 0.8"
  orden=orden1+url+orden2+nombre+orden3
  print (orden)
  os.system(orden)
  return(nombre)
  
def obtenerinfo (archivo):
  f = open("test.json")
  json_load = (json.loads(f.read()))
  print(json_load["name"]["excerpt"])

def transformarSentenceembeding(sentences,original):
  valores= []
  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  embeddings = model.encode(sentences)

  query = original
  query_vec = model.encode([query])[0]
  i=0
  for sent in sentences:
    sim = cosine(query_vec, model.encode([sent])[0])
    print("Sentence = repositorio",i, " ,similarity = ", sim)
    i+=1
    valores.append(sim)
  return valores


def transformarTfidf(sentences,original):
  valores=[]
  con = 0
  con2 = 0
  tfidf_vectorizer = TfidfVectorizer()
  tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
  cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
  print(cosine_sim)
  for i in sentences:
    #print (i)
    #print (original)
    print (i == original)
    if i == original:
      for j in cosine_sim[con]:
        #print("j es "+j)
        if con2 != con:
          valores.append(j)
          con2 += 1
        else:
          con2 +=1

    else:
      con +=1

  return valores

def obtainReadme(archivo):
  f = open(archivo)
  json_load = (json.loads(f.read()))
  url = json_load["readmeUrl"]["excerpt"]
  print(url)
  return url

def obtainDoi(archivo):
  f = open(archivo)
  data = (json.loads(f.read()))
  #url = json_load["citation"]["doi"]
  con=0
  for i in data["citation"]:
    for j in i:
      if(j == "doi"):
        #print(con)
        res=data["citation"][con]["doi"]
        #print(res)
    con+=1
  f.close()
  abstract=obtainAbastract(res)

  return abstract

def obtainAbastract(url):
  #https://doi.org/10.3233/SW-223135
  #https://doi.org/10.1007/978-3-319-68204-4_9
  #http://api.crossref.org/works/10.1007/978-3-319-68204-4_9
  new=url.replace("https://doi.org/",'')
  #print(new)
  direccion= "http://api.crossref.org/works/"
  direccion2=direccion+new
  response = requests.get(direccion2)
  texto=response.text
  data = (json.loads(texto))
  #print(data["message"])
  for i in data["message"]:
    print('')
  return ("nothing")

def obtainArxiv(archivo):
  f = open(archivo)
  data = (json.loads(f.read()))
  #url = json_load["citation"]["doi"]
  res = data["arxivLinks"]["excerpt"][0]
  f.close()
  text=obtainAbstractArxiv(res)
  return text

def obtainAbstractArxiv(res):
  print("viejo "+res)
  new=res.replace("https://arxiv.org/abs/",'')
  new2=new.replace("https://arxiv.org/pdf/",'')
  new3=new2.replace(".pdf.",'')
  new4=new3.replace(".pdf",'')
  print("nuevo "+new4)
  search = arxiv.Search(id_list=[new4])
  paper = next(search.results())
  text = paper.summary
  return text

urls = ["https://github.com/facebookresearch/encodec",
"https://github.com/huggingface/transformers",
"https://github.com/WongKinYiu/yolov7",
"https://github.com/openai/sparse_attention",
"https://github.com/jadore801120/attention-is-all-you-need-pytorch",
"https://github.com/extreme-bert/extreme-bert",
"https://github.com/fengres/mixvoxels",
"https://github.com/shi-labs/versatile-diffusion"]

print("-----PRUEBA COMPARACION REPOSITORIOS GITHUB-----")
print("lista de repositorios")
print(urls)

print("Elija el repositorio a comparar")
numero = input()
num=int(numero)
print("Ha escojido el repositorio "+urls[num])

i = 0
for url in urls:
  nombre = 'test'
  nombre2 = '.json'
  nombrecompleto=nombre+str(i)+nombre2
  #print(nombrecompleto)
  #obtenerJason(url,nombrecompleto)
  i+=1


urlreadme = []
for a in range (i):
  nombre = 'test'
  nombre2 = '.json'
  nombrecompleto=nombre+str(a)+nombre2
  urlreadme.append(obtainReadme(nombrecompleto))

origen="nothing"
cont = 0
readmes = []
readmestotal = []
for read in urlreadme:
  #print(read)
  response = requests.get(read)
  readmestotal.append(response.text)
  #print(response.text)
  print(cont==num)
  if cont==num:
    origen=response.text
    cont +=1
  else:
    readmes.append(response.text)
    cont +=1

print("origen es "+ origen)
valoresReadme = transformarSentenceembeding(readmes,origen)
valoresReadmeTfidf = transformarTfidf(readmestotal,origen)
print("valores Readme son ")
print(valoresReadme)
print("valores Readme tfidf son ")
print(valoresReadmeTfidf)

abstracts = []
for a in range (i):
  nombre = 'test'
  nombre2 = '.json'
  nombrecompleto=nombre+str(a)+nombre2
  print("ANALISIS DEL REPOSITORIO "+nombrecompleto)
  try:
    abstracts.append(obtainDoi(nombrecompleto))
  except:
    print("no hay doi")
    try:
      abstracts.append(obtainArxiv(nombrecompleto))
    except:
      print("no hay doi ni arxiv")
      abstracts.append("nothing")


otros = []
otrosTotal = []
cont = 0
for abstract in abstracts:
  otrosTotal.append(abstract)
  if cont==num:
    origen=abstract
    cont+=1
  else:
    otros.append(abstract)
    cont+=1

valoresAbstracts = transformarSentenceembeding(otros,origen)
valoresAbstractsTfidf = transformarTfidf(otrosTotal,origen)
print("valores Abstract son ")
print(valoresAbstracts)
print("valores Abstract tfidf son ")
print(valoresAbstractsTfidf)




#columnas df[nombre]
#filas df.iloc(indice)

df = pd.DataFrame(columns = ['Readme(sentence-embedings)','Readme(tf-idf)','Abstract(sentence-embedings)','Abstract(tf-idf)'])
df["Readme(sentence-embedings)"]=valoresReadme
df["Abstract(sentence-embedings)"]=valoresAbstracts
df["Readme(tf-idf)"]=valoresReadmeTfidf
df["Abstract(tf-idf)"]=valoresAbstractsTfidf
print(urls[num])
print(df)