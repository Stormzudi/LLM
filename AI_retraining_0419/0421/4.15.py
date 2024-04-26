"""
读者在实验下列代码的时候请对照书籍

"""

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama as OllamaLLM
#
# # 1.
# prompt = PromptTemplate.from_template("写一首关于{topic}的诗。")
# # 初始化模型
# model = OllamaLLM(model="gemma:2b")
#
# chain = prompt | model
# print(chain.invoke({"topic": "春天"}))
#
# # 2.
# # 绑定停止序列参数示例
# chain = prompt | model.bind(stop=[","])
# print(chain.invoke({"topic": "夏日"}))
#
#
#
# # 3.
# from langchain.output_parsers.regex import RegexParser
#
# # 假设的LLM调用输出
# llm_output = "今天天气晴朗，28摄氏度，东南风3-4级。"
#
# # 初始化RegexParser
# # 注意，这里的regex需要根据实际的输出格式来定制
# regex = r"天气:(?P<weather>.*?), 气温:(?P<temperature>.*?), 风力:(?P<wind>.*)"
# output_keys = ["weather", "temperature", "wind"]
#
# parser = RegexParser(regex=regex, output_keys=output_keys)
#
# # 使用RegexParser解析LLM输出
# parsed_output = parser.parse(llm_output)
#
# # 打印解析后的输出
# print(parsed_output)

# 4.

#
# from langchain_core.output_parsers import JsonOutputParser,SimpleJsonOutputParser
#
# prompt = PromptTemplate.from_template("写一首关于{topic}的诗。")
# model = OllamaLLM(model="gemma:2b")
#
# chain = prompt | model | JsonOutputParser
#
# prompt_str = "春天"
# print(chain.invoke({"topic": prompt_str}))
#
# prompt_str = "What's the weather like in New York today? Give the temperature in Fahrenheit."
# print(chain.invoke({"topic": prompt_str}))

# 5.
# 使用RunnableParallel简化链的调用



# from langchain_community.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, SimpleJsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 示例1:使用JsonOutputParser
prompt1 = PromptTemplate.from_template("写一首关于{topic}的诗。")
model1 = OllamaLLM(model="gemma:2b")
chain1 = prompt1 | model1 | JsonOutputParser()

prompt_str1 = "春天"
print(chain1.invoke({"topic": prompt_str1}))

prompt_str2 = "What's the weather like in New York today? Give the temperature in Fahrenheit."
print(chain1.invoke({"topic": prompt_str2}))

# 示例2:使用RunnableParallel简化链的调用
prompt2 = PromptTemplate.from_template("关于{topic}的内容:")
model2 = OllamaLLM(model="gemma:2b")
output_parser2 = SimpleJsonOutputParser()

parallel_map = RunnableParallel(topic=RunnablePassthrough())
chain2 = parallel_map | prompt2 | model2 | output_parser2

# 调用链,这里的输入需要是字典形式以匹配RunnableParallel的期望
print(chain2.invoke({"topic": "Write a poem about the ocean."}))















prompt = PromptTemplate.from_template("关于{topic}的内容：")
model = OllamaLLM(model="gemma:2b")
output_parser = SimpleJsonOutputParser()

# 使用RunnableParallel简化链的调用
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

parallel_map = RunnableParallel(topic=RunnablePassthrough())
chain = (
    parallel_map
    | prompt
    | model
    | output_parser
)

# 调用链，这里的输入需要是字典形式以匹配RunnableParallel的期望
print(chain.invoke({"topic": "Write a poem about the ocean."}))

