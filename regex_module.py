import re


class RegexProcessor:
    def __init__(self):
        # 更新后的正则表达式（移除首尾锚定符以便在文本中匹配）
        self.patterns = {
            "ID": r"(?<!\d)[1-9]\d{5}(18|19|20|(3\d))\d{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])\d{3}[\dXx](?!\d)",
            # 身份证
            "MOBILE": r"(?<!\d)1(3\d|4[5-9]|5[0-35-9]|6[567]|7[0-8]|8\d|9[0-35-9])\d{8}(?!\d)",  # 手机号
            "EMAIL": r"\b[\w]+@[A-Za-z]+(\.[A-Za-z0-9]+){1,2}\b"  # 邮箱
        }

    def _find_matches(self, text, pattern):
        """
        返回匹配项的（start, end）元组列表
        包含前向否定和后向否定断言防止部分匹配
        """
        return [(m.start(), m.end() - 1) for m in re.finditer(pattern, text, flags=re.IGNORECASE)]

    def get_regex_masks(self, text, config):
        mask_indices = set()

        for key in ["ID", "MOBILE", "EMAIL"]:
            if config.get(key, False):
                # 获取所有匹配位置
                matches = self._find_matches(text, self.patterns[key])
                # 转换为字符索引
                for start, end in matches:
                    mask_indices.update(range(start, end + 1))

        return mask_indices