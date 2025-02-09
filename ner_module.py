from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


class NERProcessor:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.label_list = [
            'O', 'B-ORG', 'I-ORG', 'E-ORG', 'B-TITLE', 'I-TITLE', 'E-TITLE',
            'B-PRO', 'I-PRO', 'E-PRO', 'B-EDU', 'I-EDU', 'E-EDU', 'B-NAME',
            'I-NAME', 'E-NAME', 'B-RACE', 'I-RACE', 'E-RACE', 'B-LOC', 'I-LOC',
            'E-LOC', 'B-CONT', 'I-CONT', 'E-CONT', 'S-NAME', 'S-RACE', 'S-ORG'
        ]

    def _parse_entities(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_ids = inputs.word_ids(batch_index=0)

        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()

        entities = []
        current_entity = None

        for idx, (word_id, pred) in enumerate(zip(word_ids, predictions)):
            label = self.label_list[pred]

            if word_id is None:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'start': word_id,
                    'end': word_id,
                    'type': label[2:],
                    'label': label
                }
            elif label.startswith('I-'):
                if current_entity and current_entity['type'] == label[2:]:
                    current_entity['end'] = word_id
            elif label.startswith('E-'):
                if current_entity and current_entity['type'] == label[2:]:
                    current_entity['end'] = word_id
                    entities.append(current_entity)
                    current_entity = None
            elif label.startswith('S-'):
                entities.append({
                    'start': word_id,
                    'end': word_id,
                    'type': label[2:],
                    'label': label
                })
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return self._deduplicate_entities(entities)

    def _deduplicate_entities(self, entities):
        seen = set()
        unique_entities = []
        for ent in entities:
            identifier = (ent['start'], ent['end'], ent['type'])
            if identifier not in seen:
                seen.add(identifier)
                unique_entities.append(ent)
        return unique_entities

    def get_ner_masks(self, text, config):
        entities = self._parse_entities(text)
        char_list = list(text)
        mask_indices = set()

        for ent in entities:
            if config.get(ent['type'], False):
                for i in range(ent['start'], ent['end'] + 1):
                    if i < len(char_list):
                        mask_indices.add(i)

        return mask_indices