import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import json
from tqdm.notebook import tqdm

from torch import cuda
device = 'cuda:0'
#device = 'cpu'

model_params = {
    "MODEL": "Salesforce/codet5-base",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 16,  # training batch size
    "VALID_BATCH_SIZE": 16,  # validation batch size
    # "TRAIN_EPOCHS": 10,  # number of training epochs\
    "TRAIN_EPOCHS" : 10,
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}

tokenizer = RobertaTokenizer.from_pretrained(model_params["MODEL"])

class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }

def load_data(path,tokenizer):
    '''
    with open(f'data/{path}.jsonl', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            obj=json.loads(line)
            # source=obj['description']+tokenizer.sep_token+obj['solutions']
            source = obj['solutions']
            
            for t in obj['test_cases']:
                sources.append(source)
                targets.append(t+tokenizer.eos_token)
            for t in obj['private_tests']:
                sources.append(source)
                targets.append(t+tokenizer.eos_token)
            if idx>50000:
                break
    '''
    
    sources=[]
    targets=[]
    
    with open(f'data/{path}.jsonl', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            # line_num += 1
            # if idx > 2500: break
            obj = json.loads(line)
            # print((obj.keys()))
            input_spec = obj['spec']
            
            source = obj['code'] + tokenizer.sep_token + input_spec
            if len(source) > model_params['MAX_SOURCE_TEXT_LENGTH']: continue

            for test_case in obj['test_case_inputs']:
                if len(test_case) > model_params['MAX_TARGET_TEXT_LENGTH'] or len(test_case) < 1: continue
                sources.append(source)
                targets.append(test_case + tokenizer.eos_token)
            '''
            for test_case in obj['public_tests']['test_case_inputs']:
                if len(test_case) > model_params['MAX_TARGET_TEXT_LENGTH'] or len(test_case) < 1: continue
                sources.append(source)
                targets.append(test_case + tokenizer.eos_token)
            
            for test_case in obj['private_tests']['input']:
                if len(test_case) > model_params['MAX_TARGET_TEXT_LENGTH'] or len(test_case) < 1: continue
                sources.append(source)
                targets.append(test_case + tokenizer.eos_token)
            '''
            # targets.append(obj['public_tests']['input'][0]+tokenizer.eos_token)

    df=pd.DataFrame()
    df['source']=sources
    df['target']=targets
    
    # df = df.sample(frac=0.025, random_state=model_params['SEED'])

    print(len(df['source']))

    return df

def train(epoch, tokenizer, model, device, loader, optimizer):
    print(epoch, "th train start")
    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    total_loss = 0
    # curr_loss = 99
    for i, data in enumerate(loader, 1):
        # print(i)
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        total_loss += loss.item()
        if i % 1000 == 0:
            aver_loss = total_loss / 1000
            total_loss = 0
            print('{:7}: {}'.format(int(i/1000), aver_loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch, "th train end")

def validate(epoch, tokenizer, model, device, loader):
    print("validate start")
    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    actuals = []
    sources = []
    with torch.no_grad():
        loss = 0
        for idx, data in enumerate(loader, 1):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=5,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            
            # loss += model(input_ids=ids, attention_mask=mask, labels=y).loss

            # if idx == 1: print(type(generated_ids))

            # if idx % 1000 == 0:
            #     print("{:10} prediction loss: {}".format(int(idx/1000), loss/1000))
            #     loss = 0

            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            source = [tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in ids]
            sources.extend(source)
            predictions.extend(preds)
            actuals.extend(target)
    print("validate end")
    return predictions, actuals, sources

def T5Trainer(
      output_dir="./outputs/", save_model_name = 'incorrect_spec_model_files'

):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = RobertaTokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)


    # Importing the raw dataset
    # dataframe = dataframe[[source_text, target_text]]
    dataframe = load_data('incorrect_pair',tokenizer)
    source_text='source'
    target_text='target'
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = YourDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )
    print('starting training')
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        predictions, actuals,sources = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals,"Source Text":sources})
        final_df.to_csv(os.path.join(output_dir, f"predictions_train{epoch}.csv"))
    # Saving the model after training
        path = os.path.join(output_dir, f"{save_model_name}{epoch}")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        
        # predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        # final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        # final_df.to_csv(os.path.join(output_dir, f"predictions_valid{epoch}.csv"))

if __name__ == '__main__':
    T5Trainer()