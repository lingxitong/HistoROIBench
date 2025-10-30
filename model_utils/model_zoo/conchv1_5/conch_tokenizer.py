import os
import torch
from typing import List, Optional, Union

import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

# https://stackoverflow.com/q/62691279
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Conch_Tokenizer:
    """
    Tokenizer for Conch and Titan
    """
    
    def __init__(self, context_length: Optional[int] = 128):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("MahmoodLab/TITAN")
        self.context_length = context_length
        
    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        input_ids = tokenize(self.tokenizer, texts)
        return input_ids

def tokenize(tokenizer, texts, max_length = 128):
    # model context length is 128, but last token is reserved for <cls>
    # so we use 127 and insert <pad> at the end as a temporary placeholder
    tokens = tokenizer.batch_encode_plus(texts, 
                                        max_length = max_length - 1,
                                        add_special_tokens=True, 
                                        return_token_type_ids=False,
                                        truncation = True,
                                        padding = 'max_length',
                                        return_tensors = 'pt')
    tokens = F.pad(tokens['input_ids'], (0, 1), value=tokenizer.pad_token_id)
    return tokens