from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
import pandas as pd
import re
import transformers
from transformers import BertModel, BertTokenizer
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F 
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__, static_folder='static')
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@127.0.0.1/berita_hoax'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:admin12345@34.45.216.14/berita_hoax'
db = SQLAlchemy(app)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRE_TRAINED_MODEL_NAME = 'indobenchmark/indobert-base-p2'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
encoder = LabelEncoder()
encoder.classes_ = np.load('bert_classes.npy', allow_pickle=True)
MAX_LEN = 512
BATCH_SIZE = 2

class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output[1])
        return self.out(output)

stopword = StopWordRemoverFactory().create_stop_word_remover()
id_sw = stopwords.words('indonesian')
adt_sw = ['salah', "klarifikasi", "disinformasi", "cekfakta", "hasil", "periksa","fakta",
          "baca","juga","tempo","penjelasan","jakarta","facebook","twitter","tik","tok",
          "tiktok","memposting","mengunggah","youtube","https","http","www","gseehttps","id", "bit","ly",
          "google","translate","mxvn","com","co","cnn","kompas","read", "formula", "cnnindonesia",
          "page", "news","permalink", "youtu", "be", "ioyc", "referensi", "index", "html", "watch", "hoaks"]
stop_words = id_sw + adt_sw
model = SentimentClassifier(2)
state_dict = torch.load('content_skripsi(8).bin', map_location=torch.device('cpu'))
if "bert.embeddings.position_ids" in state_dict:
    del state_dict["bert.embeddings.position_ids"]
model.load_state_dict(state_dict)
model.eval()

def stopwordSastrawi(text):
    return stopword.remove(text)

def addt_stop_word(sentence):
    words = word_tokenize(sentence)
    return ''.join(' '.join(w for w in words if not w.lower() in stop_words))

def all_preproc(input_data):
    text = input_data.lower()
    text = re.sub(r'[^.,a-zA-Z0-9 \n\.]',' ',text) #remove symbol
    text = re.sub(r'[\s]+', ' ', text) #menghilangkan additional whitespace
    text = re.sub(r'[^\w\s]','',text) #remove punctuation
    text = stopwordSastrawi(text)
    text = addt_stop_word(text)
    return text

def process_input(input_data, MAX_LEN):
    preprocessed_data = all_preproc(input_data)
    encoding = tokenizer.encode_plus(
      preprocessed_data,
      add_special_tokens=True,
      max_length= MAX_LEN,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
      truncation=True,
    )
    input_ids = encoding['input_ids'].flatten()
    attention_mask = encoding['attention_mask'].flatten()

    return input_ids, attention_mask

class hoax_news(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Headline = db.Column(db.Text)
    Link = db.Column(db.Text)
    Date = db.Column(db.DateTime)
    Content = db.Column(db.Text)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/Deteksi", methods=['GET', 'POST'])
def deteksi():
    if request.method == "POST":
        input_data = request.form.get("input_data")
        input_ids, attention_mask = process_input(input_data,  MAX_LEN)
        with torch.no_grad():
            output = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            y_pred = torch.max(output, dim=1)[1]
            probs = F.softmax(output, dim=1)

        ypred = encoder.inverse_transform(y_pred)[0]
        probs = probs.cpu().numpy()[0] 
        max_prob = probs.max() * 100
        max_prob = f" ({max_prob:.1f}%)"
    
        return render_template("deteksi.html", ypred=ypred, probs=max_prob)
    return render_template("deteksi.html", ypred="", probs="")

@app.route("/BeritaHoax")
def hoax():
    page = request.args.get('page', 1, type=int)
    per_page = 5
    berita_hoax = hoax_news.query.order_by(desc(hoax_news.Date)).paginate(page=page, per_page=per_page)
    return render_template("beritahoax.html", berita_hoax=berita_hoax)

if __name__ == "__main__":
    app.run(debug=True)
