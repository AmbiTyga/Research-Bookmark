import copy
import datasets
import transformers
IGNORE_INDEX = -100
from pyreft import ReftDataCollator
import torch
from typing import List, Union, Dict, Sequence

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
            padding_value=IGNORE_INDEX).to(self.device)
        num_valid_label = torch.not_equal(labels, IGNORE_INDEX).sum().item()
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


# This also handles Pixel Values
def make_last_position_supervised_data_module(
    tokenizer, model, images, inputs, outputs, 
    num_interventions=1, nonstop=False, max_partition=9
):
    """Make dataset and collator for supervised fine-tuning."""

    all_base_input_ids, all_intervention_locations, all_output_ids = [], [], []
    all_pixel_values = []
    for i in range(len(inputs)):
        _image = images[i]
        _input = inputs[i]
        _output = outputs[i]
    
        base_prompt = _input
        base_input = base_prompt + _output
        if not nonstop:
            base_input += tokenizer.eos_token
    
        # tokenize 
        # Input
        base_prompt_ids = tokenizer(
            base_prompt, 
            max_length=tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)

        # Input + Output
        _, base_input_ids, pixel_values = model.preprocess_inputs(
             base_input, 
             [_image], 
             max_partition=max_partition
        )
        output_ids = copy.deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX
        
        all_base_input_ids.append(base_input_ids)
        all_intervention_locations.append([[base_prompt_length - 1]]*num_interventions)
        all_output_ids.append(output_ids)
        all_pixel_values.append(pixel_values)
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "pixel_values": all_pixel_values,
        "labels": all_output_ids,
    })
    device = torch.device("cuda") 
    data_collator_fn = DataCollatorForMultimodalDataset(
        text_tokenizer=tokenizer, device=device
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
