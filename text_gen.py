from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle


modeloselect = input("Selecione o modelo (digite só o número): \n 1 - Fantasia \n 2 - Histórias Em Geral \n 3 - Machado de Assis \n 4 - Horror \n 5 - Ficcção Científica \n")

if modeloselect == '1':
    pastamodelo = 'fantasia'

if modeloselect == '2':
    pastamodelo = 'historiasemgeral'

if modeloselect == '3':
    pastamodelo = 'machado'

if modeloselect == '4':
    pastamodelo = 'horror'

if modeloselect == '5':
    pastamodelo = 'ficcao'

with open(pastamodelo+'/transform.pkl', 'rb') as file:
    loaded_tokenizer = pickle.load(file)

with open(pastamodelo+'/max_seq_len.pkl', 'rb') as file:
    loaded_max_seq_len = pickle.load(file)

# Load json and create model
json_file=open((pastamodelo+'/model.json'),'r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=tf.keras.models.model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights(pastamodelo+"/model.h5")
print(pastamodelo+"Loaded model from disk")

seed_text = input("Digite o texto de início:\n")
next_words = int(input("Digite o número de palavras (max 100):\n"))

while (next_words > 100):
    next_words = int(input("Digite o número de palavras (max 100):\n"))

for _ in range(next_words):
    token_list = loaded_tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=int(loaded_max_seq_len) - 1, padding='pre')
    predicted = loaded_model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in loaded_tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print("Texto gerado:\n")
print(seed_text)