from bertviz.pytorch_transformers_attn import BertForSequenceClassification, BertTokenizer
import torch
from collections import defaultdict

def get_pool(model, model_type, tokenizer, sentence_a, sentence_b=None, include_queries_and_keys=False):
    if model_type not in ('bert', 'gpt2', 'xlnet', 'roberta','bert_fineturn'):
        raise ValueError("Invalid model type:", model_type)
    if not sentence_a:
        raise ValueError("Sentence A is required")
    is_sentence_pair = bool(sentence_b)
    if is_sentence_pair and model_type not in ('bert', 'roberta', 'xlnet','bert_fineturn'):
        raise ValueError(f'Model {model_type} does not support sentence pairs')
    if is_sentence_pair and model_type == 'xlnet':
        raise NotImplementedError("Sentence-pair inputs for XLNet not currently supported.")

    # Prepare inputs to model
    tokens_a = None
    tokens_b = None
    token_type_ids = None
    if not is_sentence_pair: # Single sentence
        if model_type in ('bert', 'roberta','bert_fineturn'):
            tokens_a = [tokenizer.cls_token] + tokenizer.tokenize(sentence_a) + [tokenizer.sep_token]
        elif model_type == 'xlnet':
            tokens_a = tokenizer.tokenize(sentence_a) + [tokenizer.sep_token] + [tokenizer.cls_token]
        else:
            tokens_a = tokenizer.tokenize(sentence_a)
    else:
        if model_type in ['bert','bert_fineturn']:
            tokens_a = [tokenizer.cls_token] + tokenizer.tokenize(sentence_a) + [tokenizer.sep_token]
            tokens_b = tokenizer.tokenize(sentence_b)[0:512-len(tokens_a)-1] + [tokenizer.sep_token]
            token_type_ids = torch.LongTensor([[0] * len(tokens_a) + [1] * len(tokens_b)])
        elif model_type == 'roberta':
            tokens_a = [tokenizer.cls_token] + tokenizer.tokenize(sentence_a) + [tokenizer.sep_token]
            tokens_b = [tokenizer.sep_token] + tokenizer.tokenize(sentence_b) + [tokenizer.sep_token]
            # Roberta doesn't use token type embeddings per https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/convert_roberta_checkpoint_to_pytorch.py
        else:
            tokens_b = tokenizer.tokenize(sentence_b)

    token_ids = tokenizer.convert_tokens_to_ids(tokens_a + (tokens_b if tokens_b else []))
    tokens_tensor = torch.tensor(token_ids).unsqueeze(0)
    if(tokens_b and model_type=="bert_fineturn"):
        max_length = 200
        input_ids = token_ids
        token_type_ids = [0] * len(tokens_a) + [1] * len(tokens_b)
        mask_padding_with_zero = True
        pad_token = 0
        max_length = 200
        pad_token_segment_id = 0
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        
        tokens_tensor = torch.tensor(input_ids).unsqueeze(0)
        token_type_ids = torch.tensor(token_type_ids).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)

    # Call model to get attention data
    model.eval()
    if((token_type_ids is not None) and (tokens_b is not None) and(model_type=='bert_fineturn')):
        #output = model(tokens_tensor,attention_mask=attention_mask,token_type_ids=token_type_ids)
        output = model(tokens_tensor,attention_mask=attention_mask,token_type_ids=token_type_ids)
        output = torch.nn.Softmax(dim=-1)(output[0])
    elif token_type_ids is not None:
        output = model(tokens_tensor, token_type_ids=token_type_ids)
    else:
        output = model(tokens_tensor)
    # pool_list = output[1]
    return output


model_type = "bert_fineturn"
MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer)
}
model_class, tokenizer_class = MODEL_CLASSES["bert"]
tokenizer = tokenizer_class.from_pretrained("bert-base-multilingual-cased")
model = model_class.from_pretrained("../local/bert4ecomerce_mrpc/checkpoint-900")
model.eval()

fw = open('test.pred','w',encoding="utf-8")
with open('test.tsv','r',encoding='utf-8') as lines:
    for line in lines:
        tmp = line.strip().split('\t')
        sentence_a = tmp[3]
        sentence_b = tmp[4]
        label = str(tmp[0])
        pool_list = get_pool(model, model_type, tokenizer, sentence_a, sentence_b)
        pool_list = pool_list.cpu().detach().numpy().tolist()
        fw.write("\t".join([label]+[str(pool_list[0][1])]))
        fw.write("\n")
fw.close()