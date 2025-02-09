from .ner_module import NERProcessor
from .regex_module import RegexProcessor


class TextSanitizer:
    def __init__(self, model_path):
        self.ner_processor = NERProcessor(model_path)
        self.regex_processor = RegexProcessor()

    def sanitize(self, text, config):
        # 获取各模块的掩码
        ner_masks = self.ner_processor.get_ner_masks(text, config)
        regex_masks = self.regex_processor.get_regex_masks(text, config)

        # 合并掩码
        all_masks = ner_masks.union(regex_masks)

        # 执行替换
        modified_chars = [
            '*' if i in all_masks else char
            for i, char in enumerate(list(text))
        ]

        return ''.join(modified_chars)