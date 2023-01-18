import os
import json
import requests
import pandas as pd 
import numpy as np
from markdown import markdown
from sentence_transformers import SentenceTransformer
import arxiv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import webbrowser


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


#------- Algoritmo Sentences Embedings------------
def transformarSentenceembeding(sentences,original):
  valores= []
  if(original=="no hay"):
    for num in sentences:
      valores.append("-")
    return valores
  
  model = SentenceTransformer('all-MiniLM-L6-v2')
  i=0
  for sent in sentences:
    sim = cosine(model.encode(original), model.encode(sent))
    print("Sentence = repositorio",i, " ,similarity = ", sim)
    i+=1
    valores.append(sim)
  return valores

def cosine(u,v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

#------- Algoritmo Tfidf------------
def transformarTfidf(sentences,original):
  valores=[]
  if(original=="no hay"):
    for sent in range(len(sentences)-1):
      valores.append("-")
    return valores
  con = 0
  con2 = 0
  tfidf_vectorizer = TfidfVectorizer()
  tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
  cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
  print(cosine_sim)
  for i in sentences:
    print (i == original)
    if i == original:
      for j in cosine_sim[con]:
        if con2 != con:
          valores.append(j)
          con2 += 1
        else:
          con2 +=1
    else:
      con +=1
  return valores

#------Algoritmo de Stemming-----------
def raices (text):
  fin=""
  porter_stemmer  = PorterStemmer()
  raiz = nltk.word_tokenize(text)
  for palabra in raiz:
    w = porter_stemmer.stem(palabra)
    fin=fin+w+" "
  return fin

def obtainTittle(archivo):
  f = open(archivo)
  json_load = (json.loads(f.read()))
  url = json_load["name"]["excerpt"]
  return url

def obtainReadme(archivo):
  f = open(archivo)
  json_load = (json.loads(f.read()))
  url = json_load["readmeUrl"]["excerpt"]
  print(url)
  return url

#--------ObtainDoi (no consistente)----------
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
  textstemmed=raices(text)
  return textstemmed

def obtainTopic(archivo):
  f = open(archivo)
  data = (json.loads(f.read()))
  res = data["topics"]["excerpt"]
  junto = ""
  for palabra in res:
    junto = junto + " " + palabra
  f.close()
  return junto

def obtainDescription(archivo):
  f = open(archivo)
  data = (json.loads(f.read()))
  res = data["description"][0]["excerpt"]
  f.close()
  textstemmed = raices(res)
  return textstemmed


with open("data.json") as f:
    urls = json.load(f)

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
  obtenerJason (url,nombrecompleto)
  i+=1


nombres = []
for a in range (i):
  nombre = 'test'
  nombre2 = '.json'
  nombrecompleto=nombre+str(a)+nombre2
  nombres.append(obtainTittle(nombrecompleto))

nombresSin = []
nombreOrigen =''
co=0
for cada in nombres:
  if co==num:
    nombreOrigen=cada
    co+=1
  else:
    nombresSin.append(cada)
    co+=1
  


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
  response = requests.get(read)
  readmestotal.append(response.text)
  if cont==num:
    origen=response.text
    cont +=1
  else:
    readmes.append(response.text)
    cont +=1

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

topics = []
for a in range (i):
  nombre = 'test'
  nombre2 = '.json'
  nombrecompleto=nombre+str(a)+nombre2
  print("ANALISIS DEL REPOSITORIO "+nombrecompleto)
  try:
    topics.append(obtainTopic(nombrecompleto))
  except:
    print("no hay topic")
    topics.append("no hay")

topiclista = []
topiclistatotal = []
cont = 0
for topic in topics:
  topiclistatotal.append(topic)
  if cont==num:
    origen=topic
    cont+=1
  else:
    topiclista.append(topic)
    cont+=1

#valoresTopic = transformarSentenceembeding(topiclista,origen)
valoresTopicTfidf = transformarTfidf(topiclistatotal,origen)
#print("valores Topic son ")
#print(valoresTopic)
print("valores Topic tfidf son ")
print(valoresTopicTfidf)

description = []
for a in range (i):
  nombre = 'test'
  nombre2 = '.json'
  nombrecompleto=nombre+str(a)+nombre2
  print("ANALISIS DEL REPOSITORIO "+nombrecompleto)
  try:
    description.append(obtainDescription(nombrecompleto))
  except:
    print("no hay description")
    topics.append("no hay")

descriptions = []
descriptionsTotal = []
cont = 0
for des in description:
  descriptionsTotal.append(des)
  if cont==num:
    origen=des
    cont+=1
  else:
    descriptions.append(des)
    cont+=1


valoresDescriptions = transformarSentenceembeding(descriptions,origen)
valoresDescriptionsTfidf = transformarTfidf(descriptionsTotal,origen)
print("valores Description son ")
print(valoresDescriptions)
print("valores Description tfidf son ")
print(valoresDescriptionsTfidf)

df = pd.DataFrame(index=nombresSin,columns = ['Readme S-Embedings','Readme TfIdf','Abstract S-Embedings','Abstract TfIdf','Keywords TfIdf','Description S-Embedings','Description TfIdf'])
df["Readme S-Embedings"]=valoresReadme
df["Readme TfIdf"]=valoresReadmeTfidf
df["Abstract S-Embedings"]=valoresAbstracts
df["Abstract TfIdf"]=valoresAbstractsTfidf
df["Keywords TfIdf"]=valoresTopicTfidf
df["Description S-Embedings"]=valoresDescriptions
df["Description TfIdf"]=valoresDescriptionsTfidf
print("\n ------------- comparacion de ------------------ \n")
print(urls[num])
print("\n ------------------------------- \n")
print(df)

for a in range(i):
  nombre = 'test'
  nombre2 = '.json'
  nombrecompleto=nombre+str(a)+nombre2

html = df.to_html()
html_template = """<html>
<head>
<title>Tfg Javier Prieto</title>
</head>
<body>
<h2>Compararación de proyectos de software científico similares</h2>
  
<p>El repositorio escogido para comparar es ' """+nombreOrigen+ """ '.</p>
  
</body>
</html>
"""
#write html to file
text_file = open("index.html", "w")
text_file.write(html_template)
text_file.write(html)
text_file.close()
webbrowser.open_new_tab("index.html")


