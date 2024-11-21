import csv
import shutil
from pathlib import Path


akiec = []
bcc = []
bkl = []
df = []
mel = []
nv = []
vasc = []

with open("./datasorting/HAM10000_metadata.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    if row[2] == "akiec": 
      akiec.append(row[1])
    elif row[2] == "bcc": 
      bcc.append(row[1])
    elif row[2] == "bkl": 
      bkl.append(row[1])
    elif row[2] == "df": 
      df.append(row[1])
    elif row[2] == "mel": 
      mel.append(row[1])
    elif row[2] == "nv": 
      nv.append(row[1])
    elif row[2] == "vasc": 
      vasc.append(row[1])

akiec_test = akiec[int(len(akiec)*0.8):len(akiec)]
akiec_train = akiec[:int(len(akiec)*0.8)]
bcc_test = bcc[int(len(bcc)*0.8):len(bcc)]
bcc_train = bcc[:int(len(bcc)*0.8)]
bkl_test = bkl[int(len(bkl)*0.8):len(bkl)]
bkl_train = bkl[:int(len(bkl)*0.8)]
df_test = df[int(len(df)*0.8):len(df)]
df_train = df[:int(len(df)*0.8)]
mel_test = mel[int(len(mel)*0.8):len(mel)]
mel_train = mel[:int(len(mel)*0.8)]
nv_test = nv[int(len(nv)*0.8):len(nv)]
nv_train = nv[:int(len(nv)*0.8)]
vasc_test = nv[int(len(vasc)*0.8):len(vasc)]
vasc_train = nv[:int(len(vasc)*0.8)]


for i in mel_train: 
    pathstring = "./data/test/mel/" + i + ".jpg"
    path = Path(pathstring)
    if not path.is_file(): 
        pathstring = "./datasorting/HAM10000_images_part_2/" + i + ".jpg"
        path = Path(pathstring)
    if path.is_file(): 
        diseasepath = "./data/train/mel/" + i + ".jpg"
        shutil.move(pathstring, diseasepath)

for i in vasc_train: 
    pathstring = "./datasorting/HAM10000_images_part_1/" + i + ".jpg"
    path = Path(pathstring)
    if not path.is_file(): 
        pathstring = "./datasorting/HAM10000_images_part_2/" + i + ".jpg"
        path = Path(pathstring)
    if path.is_file(): 
        diseasepath = "./data/train/vasc/" + i + ".jpg"
        shutil.move(pathstring, diseasepath)
    
for i in vasc_test: 
    pathstring = "./datasorting/HAM10000_images_part_1/" + i + ".jpg"
    path = Path(pathstring)
    if not path.is_file(): 
        pathstring = "./datasorting/HAM10000_images_part_2/" + i + ".jpg"
        path = Path(pathstring)
    if path.is_file(): 
        diseasepath = "./data/test/vasc/" + i + ".jpg"
        shutil.move(pathstring, diseasepath)