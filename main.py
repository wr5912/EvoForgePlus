# -*- coding: utf-8 -*-
"""
EvoForgePlus 主入口文件

这个文件包含了三个主要的用例场景，展示了如何使用 EvoForgePlus 框架：
1. case1: 使用复杂代理 DNA 配置（循环和分支）进行诗歌创作
2. case2: 使用进化优化器对智能体进行内环优化，提升数学问题解决能力
3. case3: 完整的进化流程，包括多代变异和最佳配置保存

该文件还配置了 MLflow 实验追踪和 DSPy 语言模型设置。

环境变量配置：
所有敏感配置（如API密钥）都从.env文件中读取，确保安全性和可移植性。
请确保项目根目录下存在.env文件，并正确配置相关环境变量。
"""

import os
import json
import dspy
from dotenv import load_dotenv
from evoforge.agent_dna_config import AgentDNAConfig
from evoforge.engine import GraphAgent
import mlflow
from evoforge.optimizer import EvoOptimizer

# 加载环境变量
# 从项目根目录的.env文件中读取配置
load_dotenv()

# 配置 MLflow 实验追踪
# 设置 MLflow 跟踪服务器的 URI（本地运行），从环境变量读取，默认为本地服务器
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# 启用 DSPy 自动日志记录
# log_compiles: 跟踪优化过程
# log_evals: 跟踪评估结果
# log_traces_from_compile: 在优化过程中跟踪程序轨迹
mlflow.dspy.autolog(
    log_compiles=True,
    log_evals=True,
    log_traces_from_compile=True
)

# 创建唯一实验名称用于区分不同运行，从环境变量读取，默认为 "EvoForgePlus"
mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "EvoForgePlus")
mlflow.set_experiment(mlflow_experiment_name)

# 配置 DSPy 语言模型
# 从环境变量读取 DeepSeek API 配置，确保敏感信息不暴露在代码中
# 注意：必须按照 LiteLLM 支持的格式指定模型名称
model = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-reasoner")
api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
api_key = os.getenv("DEEPSEEK_API_KEY")

# 验证必要的环境变量是否已设置
if not api_key:
    raise ValueError("DEEPSEEK_API_KEY 环境变量未设置。请在 .env 文件中配置您的 DeepSeek API 密钥。")


# 创建 DSPy 语言模型实例
LM = dspy.LM(
    model=model,
    api_key=api_key,
    api_base=api_base,
)

# 全局配置 DSPy 使用指定的语言模型
dspy.configure(lm=LM)

# 打印配置信息（调试用，生产环境可移除）
print(f"✅ MLflow 跟踪服务器: {mlflow_tracking_uri}")
print(f"✅ MLflow 实验名称: {mlflow_experiment_name}")
print(f"✅ DeepSeek 模型: {model}")
print(f"✅ DeepSeek API 基础地址: {api_base}")
print("✅ DeepSeek API 密钥: 已配置")

# 准备训练数据集
# 这是驱动进化的核心，用于告诉系统什么是好的表现
# 数据集包含示例问题及其正确答案，用于训练智能体解决数学问题
train_data = [
    dspy.Example(
        question="小明有3个苹果，小红给了他2个，然后他吃掉了1个，现在有几个？",
        answer="4"
    ).with_inputs("question"),

    dspy.Example(
        question="一个长方形长10米，宽5米，面积是多少？",
        answer="50"
    ).with_inputs("question"),

    dspy.Example(
        question="5的平方加上10等于多少？",
        answer="35"
    ).with_inputs("question"),
]


def evaluation_metric(example, pred, trace=None):
    """
    评估指标函数
    用于判断智能体输出的正确性

    参数:
        example: 训练数据中的示例，包含问题和正确答案
        pred: 智能体的预测结果，包含智能体生成的答案
        trace: 可选的执行轨迹（用于调试和优化分析）

    返回:
        bool: 预测是否正确的布尔值
             True: 预测答案中包含正确答案（子字符串匹配）
             False: 预测答案中不包含正确答案或处理过程中出现异常

    说明:
        这是一个简单的评估函数，使用子字符串匹配来检查预测答案是否包含正确答案。
        生产环境中可以使用更复杂的语义相似度或 LLM 评分来提高评估质量。
    """
    # 简单的精确匹配（生产环境可以使用更复杂的语义相似度或 LLM 评分）
    try:
        # 清理答案中的非数字字符，进行简单比对
        ground_truth = str(example.answer).strip()
        prediction = str(pred.answer).strip()

        # 使用简单的包含关系检查，提高容错率
        # 例如，预测答案为 "答案是 4" 而正确答案是 "4" 时，仍然算正确
        return ground_truth in prediction
    except Exception:
        # 如果处理过程中出现异常（如类型转换错误），返回 False
        return False


def case1():
    """
    用例1：使用复杂代理 DNA 配置（包含循环和分支）进行诗歌创作
    
    这个用例展示了如何使用预定义的复杂智能体 DNA 配置来创建一个具有
    循环和分支能力的智能体，用于创作特定主题的诗歌。
    
    流程：
    1. 加载复杂代理 DNA 配置（从 JSON 文件）
    2. 实例化图智能体
    3. 运行智能体处理诗歌创作任务
    4. 输出结果和执行路径
    
    注意：这个智能体包含自我批评（Critic）和精炼（Refiner）节点，
    可以迭代改进诗歌质量。
    """
    print(">>> Loading Advanced Agent DNA (Loop & Branch)...")
    with open("complex_agent_dna_config.json", "r", encoding="utf-8") as fd:
        config: dict = json.loads(fd.read())
        agent_dna_config = AgentDNAConfig(**config)

    # 实例化图智能体
    agent = GraphAgent(agent_dna_config)

    # 运行测试
    # 我们让批评家（Critic）变得极其挑剔（通过覆盖 instruction，或者依赖 config 里的 prompt）
    # 这里我们直接运行，观察它是否会触发精炼器（Refiner）

    topic = "写一首关于'程序员熬夜'的悲伤的诗"
    print(f"\n>>> Input Topic: {topic}")

    # 运行智能体
    result = agent(topic=topic)

    print("\n" + "=" * 50)
    print("🏁 FINAL RESULT")
    print("=" * 50)
    if hasattr(result, '_trace_path'):
        print(f"Final Path Taken: {' -> '.join(result._trace_path)}")

    print("-" * 20)
    print(f"Final Poem:\n{result.content}")
    print("-" * 20)

    # 如果最后一步有评论意见，打印出来
    if hasattr(result, 'critique'):
        print(f"Last Critique: {result.critique}")
        print(f"Final Decision: {getattr(result, 'decision', 'N/A')}")


def case2():
    """
    用例2：执行智能体进化的完整流程（内环优化）
    
    这个用例展示了如何使用进化优化器对智能体进行内环优化，提升其数学问题解决能力。
    
    流程：
    1. 加载基础代理 DNA 配置
    2. 实例化 0 代智能体（未经优化的初始智能体）
    3. 测试进化前的智能体表现
    4. 使用进化优化器进行内环优化
    5. 验证进化后的智能体表现
    6. 查看进化成果（学习的演示示例）
    
    这个用例展示了如何通过进化优化提升智能体在特定任务上的表现。
    """
    # --- 步骤 1: 加载 DNA（智能体配置）---
    print(">>> Loading Agent DNA...")
    with open("agent_dna_config.json", "r", encoding="utf-8") as fd:
        config: dict = json.loads(fd.read())
        agent_dna_config: AgentDNAConfig = AgentDNAConfig(**config)

    # --- 步骤 2: 实例化 0 代智能体 ---
    print(">>> Building Zero-Shot Agent...")
    agent = GraphAgent(agent_dna_config)

    # 测试进化前的 0 代智能体效果
    print("\n--- Test Before Evolution ---")
    q1 = "计算 25 乘以 4 再减去 10"
    res = agent(question=q1)
    print(f"Q: {q1}")
    print(f"Plan: {getattr(res, 'plan', 'N/A')}")
    print(f"Answer: {res.answer}")

    # --- 步骤 3: 启动进化（内环优化）---
    print("\n>>> Starting Evolution (Inner Loop)...")
    optimized_agent, cur_agent_dna_config = EvoOptimizer(agent_dna_config, train_data, evaluation_metric).evolve()

    # --- 步骤 4: 验证进化后的智能体 ---
    print("\n--- Test After Evolution ---")
    q2 = "一辆车每小时跑 80 公里，跑了 3 小时，然后倒车 20 公里，总位移是多少？"
    res_opt = optimized_agent(question=q2)

    print(f"Q: {q2}")
    print(f"Plan (Optimized): {getattr(res_opt, 'plan', 'N/A')}")
    print(f"Answer (Optimized): {res_opt.answer}")

    # --- 步骤 5: 查看进化成果 ---
    # 查看 'executor' 节点，了解 DSPy 自动添加的 Few-Shot 示例
    print("\n>>> Inspecting Evolution DNA:")
    # executor 是 ReAct 模块，可以查看其学习的演示示例
    if hasattr(optimized_agent.executor, 'demos'):
        print(f"Executor Node learned {len(optimized_agent.executor.demos)} optimal examples from dataset.")
        for idx, demo in enumerate(optimized_agent.executor.demos):
            print(f"\n[Learned Example {idx + 1}]")
            print(f"Question: {demo.question}")
            print(f"Plan: {demo.plan}")
            print(f"Answer: {demo.answer}")

    # （可选）保存优化后的智能体（DSPy 支持 save/load）
    # optimized_agent.save("optimized_agent_v1.json")


def case3():
    """
    用例3：完整的进化流程，包括多代变异和最佳配置保存
    
    这个用例展示了完整的进化优化流程，包括：
    1. 定义评估指标
    2. 加载初始配置
    3. 启动多代进化优化
    4. 保存最佳配置
    
    参数说明：
    - max_generations: 最大进化代数，控制进化过程的迭代次数
    - score_threshold: 分数阈值，达到该阈值后停止进化
    
    这个用例会生成一个经过多代优化的最佳智能体配置，并保存到文件中。
    """
    # 3. 定义指标
    def metric_func(gold, pred, trace=None):
        """
        评估指标函数（用例3专用）
        
        参数:
            gold: 标准答案（ground truth）
            pred: 预测答案
            trace: 可选的执行轨迹
            
        返回:
            bool: 预测答案是否包含正确答案
        """
        return str(gold.answer) in str(pred.answer)

    # 4. 加载初始配置
    with open("agent_dna_config.json", "r") as fd:
        config: dict = json.loads(fd.read())
        agent_dna_config: AgentDNAConfig = AgentDNAConfig(**config)

    # 5. 启动进化
    optimizer = EvoOptimizer(
        agent_dna_config=agent_dna_config,
        trainset=train_data,
        metric_func=metric_func,
        max_generations=5,  # 最多尝试变异 5 次
        score_threshold=95.0  # 达到 95 分就停止
    )

    best_agent, best_config = optimizer.evolve()

    # 6. 保存最终结果
    print("\n>>> Evolution Complete!")
    print(f"Best Config Structure: {best_config.nodes.keys()}")
    with open("best_agent_dna_config.json", "w") as f:
        json.dump(best_config.model_dump_json(), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    """
    主程序入口
    
    执行顺序：
    1. case1(): 运行复杂代理 DNA 配置的诗歌创作用例
    2. case2(): 运行智能体内环优化用例
    3. case3(): 运行完整进化流程用例
    
    每个用例都展示了 EvoForgePlus 框架的不同功能和应用场景。
    """
    case1()
    case2()
    case3()
