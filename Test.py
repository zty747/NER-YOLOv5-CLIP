from text_sanitizer.integrator import TextSanitizer

# 初始化处理器
sanitizer = TextSanitizer("models_for_ner/NERcheckpoint-360")

# 配置参数
config = {
    "LOC": True,      # NER实体
    "NAME": True,
    "ORG": True,
    "PRO": True,

    "ID": True,       # 正则实体
    "MOBILE": True,
    "EMAIL": False
}

# 测试文本 以下信息不是真的
test_text = "张腾亚在福建师范大学学习软件工程,身份证号350923200303270058,手机号13123263837,邮箱是2677925676@qq.com"

# 执行脱敏
result = sanitizer.sanitize(test_text, config)
print("原始文本:", test_text)
print("脱敏文本:", result)