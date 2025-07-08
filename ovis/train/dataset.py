import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import PreTrainedTokenizer
import json
import os
import copy
from ovis.train.dataset.formatter import ConversationFormatter

from ovis.util.constants import IMAGE_TOKEN_ID, IGNORE_ID
import transformers
from typing import Dict, List, Sequence, Union
class MultiModalDataset(Dataset):
    def __init__(
        self,
        json_path,
        text_tokenizer,
        visual_tokenizer,
        **kwargs,
    ):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} not found.")
        
        self.json_path = json_path
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.text_tokenizer = text_tokenizer
        self.visual_tokenizer = visual_tokenizer
        self.img_h, self.img_w = self.visual_tokenizer.get_image_size()
        self.max_text_length = kwargs.get("max_text_length", 128)
        self.min_frames = kwargs.get("min_frames", 1)
        self.max_frames = kwargs.get("max_frames", 16)
        self.formatter = ConversationFormatter(tokenizer=self.text_tokenizer)
        
    def read_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} not found.")
        image = Image.open(image_path).convert("RGB")
        return image
    
    def preprocess_inputs(
        self, 
        text_or_conversations, 
        images,
        max_partition=9,
        generation_preface='',
        propagate_exception=True,
    ):
        # convert text to conversations
        if isinstance(text_or_conversations, str):
            conversations = [{
                "from": "human",
                "value": text_or_conversations
            }]
        elif isinstance(text_or_conversations, list):
            conversations = text_or_conversations
        
        prompt, raw_input_ids, raw_labels = self.formatter.format(
            conversations, generation_preface=generation_preface
        )

        input_ids = []
        labels = []
        pixel_values = []
        invalidate_label = False
        image_token_indices = [
            i for i, v in enumerate(raw_input_ids)
            if v == IMAGE_TOKEN_ID
        ]
        last_image_token_index = -1

        for i in range(len(image_token_indices)):
            head = 0 if i == 0 else image_token_indices[i - 1] + 1
            tail = image_token_indices[i]
            last_image_token_index = tail
            input_ids.extend(raw_input_ids[head:tail])
            labels.extend(raw_labels[head:tail])
            try:
                image = images[i]
                raw_pixel_values, image_placeholders = self.visual_tokenizer.preprocess_image(
                    image, max_partition=max_partition)
            except Exception as e:
                if propagate_exception:
                    raise e
                invalidate_label = True
                raw_pixel_values, image_placeholders = self.visual_tokenizer.mock_input()
            input_ids.extend(image_placeholders)
            labels.extend([IGNORE_ID] * len(image_placeholders))
            pixel_values.append(raw_pixel_values)
        input_ids.extend(raw_input_ids[last_image_token_index + 1:])
        labels.extend(raw_labels[last_image_token_index + 1:])

        # return tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor([IGNORE_ID] * len(labels) if invalidate_label else labels, dtype=torch.long)
        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None

        return prompt, input_ids, pixel_values, labels

    def __getitem__(self, i):
        sample = self.data[i]
        convs = copy.deepcopy(sample)
        images = None
        max_partition = sample.get("max_partition", None)
        
        if 'image' in sample:
            img_path = sample['image']
            images = [self.read_image(img_path)]

        prompt, input_ids, pixel_values, labels = self.preprocess_inputs(
            convs,
            images,
            max_partition=max_partition,
            generation_preface=None,
            return_labels=True,
            propagate_exception=False
        )

        input_ids = input_ids[:self.text_max_length]
        labels = labels[:self.text_max_length]

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )


class DataCollatorForMultimodalDataset:
    def __init__(self, text_tokenizer: transformers.PreTrainedTokenizer, device):
        self.text_tokenizer = text_tokenizer
        self.device = device

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        pixel_values, input_ids, labels, intervention_locations = tuple([instance[key] for instance in instances]
                                                for key in ("pixel_values", "input_ids", "labels", "intervention_locations"))
        input_ids = [torch.LongTensor(i) for i in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.text_tokenizer.pad_token_id).to(self.device)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id).to(self.device)
        
        labels = [torch.LongTensor(i) for i in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_ID).to(self.device)
        num_valid_label = torch.not_equal(labels, IGNORE_ID).sum().item()
        if num_valid_label == 0:
            print(
                f'[DataCollatorForMultimodalDataset] All labels in a batch are ignored, which may lead to training instability\n{input_ids=}\n{attention_mask=}\n{labels=}')
        
        pixel_values = [torch.BFloat16Tensor(i).to(self.device) for i in pixel_values]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            intervention_locations=torch.LongTensor(intervention_locations)
        )
