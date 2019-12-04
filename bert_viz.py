from bertviz.pytorch_transformers_attn import BertForSequenceClassification, BertTokenizer
from bertviz.head_view import show
model_type = "bert_fineturn"
MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer)
}
model_class, tokenizer_class = MODEL_CLASSES["bert"]
tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
model = model_class.from_pretrained("../local/model_bert_90.65")
model.eval()
sentence_a = "How would you compare Qatar to UAE? Just wondering because they said UAE is an open city"
sentence_b = "Abu Dhabi is fab! I used to live there .I'm sure Doha will get like that soon It is dangerous to be sincere unless you are also stupid - George Bernard Shaw (1856-1950)"
show(model, model_type, tokenizer, sentence_a, sentence_b)