from abc import ABC, abstractmethod

from ovis.util.constants import IMAGE_TOKEN, IGNORE_ID, IMAGE_TOKEN_ID

class ConversationFormatter(ABC):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_token = IMAGE_TOKEN
        self.image_token_id = IMAGE_TOKEN_ID
        self.ignore_id = IGNORE_ID
        self.im_end = "<|im_end|>\n"

        self.from2role = {
            "system": "<|im_start|>system\n",
            "human": "<|im_start|>user\n",
            "gpt": "<|im_start|>assistant\n",
        }
        self.default_system_prompt = "You are a helpful assistant."

    def format(self, conversations, generation_preface=None):
        gpt_token_num = len(self.tokenizer(
            self.from2role['gpt'], add_special_tokens=False).input_ids
        )

        if conversations[0]['from']!='system':
            conversations.insert(0, {
                'from': 'system', 
                'value': self.default_system_prompt
            })
        
        if generation_preface is not None:
            conversations.append({
                "from": "gpt",
                "value": generation_preface
            })
        
        prompt = ""
        input_ids = []
        labels = []
        num_conv = len(conversations)
        for i, conversation in enumerate(conversations):
            frm = conversation['from']
            role = self.from2role[frm]
            message = conversation['value']
            text = role + message

            if i < num_conv - 1 or generation_preface is None:
                text += self.im_end

            prompt+=text
            token_ids = self._tokenize_with_image_symbol(text)
            input_ids.extend(token_ids)
            label_ids = [self.ignore_id] * len(token_ids)
            if frm == "gpt" and generation_preface is None:
                label_ids[gpt_token_num:-1] = token_ids[gpt_token_num:-1]
                labels.extend(label_ids)

        assert self._tokenize_with_image_symbol(prompt) == input_ids
        assert len(input_ids) == len(labels)

        return prompt, input_ids, labels
    
    def format_query(self, query, generation_preface=""):
        prompt, input_ids, _ = self.format([{
            "from": "human",
            "value": query
        }], generation_preface=generation_preface)

        return prompt, input_ids