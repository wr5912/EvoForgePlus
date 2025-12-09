# -*- coding: utf-8 -*-
"""
EvoForgePlus 引擎模块：图智能体（GraphAgent）实现

这个模块定义了 GraphAgent 类，它基于代理 DNA 配置动态构建和执行图结构的工作流。
智能体由多个节点组成，每个节点对应一个 DSPy 模块（如 ChainOfThought、ReAct 等），
节点之间通过流规则（顺序流或分支流）连接，形成可执行的工作流。

核心功能：
1. 根据 AgentDNAConfig 动态构建智能体节点
2. 执行智能体工作流，支持顺序和分支路由
3. 限制最大执行步骤以防止无限循环
"""

from evoforge.agent_dna_config import AgentDNAConfig, BranchFlow
import dspy


class GraphAgent(dspy.Module):
    """
    图智能体类：基于 DSPy 模块的动态工作流执行器

    这个类将 AgentDNAConfig 转换为可执行的智能体工作流。它通过动态创建 DSPy 模块
    作为工作流节点，并根据流规则在节点之间进行路由，实现复杂的智能体行为。

    属性:
        agent_dna_config (AgentDNAConfig): 智能体 DNA 配置对象
        start_node (str): 起始节点名称
        max_steps (int): 最大执行步骤数，防止无限循环
    """

    def __init__(self, agent_dna_config: AgentDNAConfig):
        """
        初始化 GraphAgent 实例

        参数:
            agent_dna_config (AgentDNAConfig): 智能体 DNA 配置对象，
                包含节点定义、流规则和起始节点等信息

        初始化步骤:
            1. 保存配置并提取起始节点
            2. 设置最大执行步骤数
            3. 动态构建所有节点对应的 DSPy 模块
        """
        super().__init__()

        # --- 步骤 1: Pydantic 校验 ---
        # 保存智能体 DNA 配置对象（已经通过 Pydantic 验证）
        self.agent_dna_config: AgentDNAConfig = agent_dna_config

        # 从配置中提取起始节点名称
        self.start_node = self.agent_dna_config.start_node
        # 设置最大执行步骤数，防止无限循环
        self.max_steps = 15

        # --- 步骤 2: 动态构建节点 ---
        # 遍历配置中的所有节点，为每个节点创建对应的 DSPy 模块
        for node_name, node_cfg in self.agent_dna_config.nodes.items():
            # 根据节点配置创建 DSPy 签名（Signature）
            signature = dspy.Signature(node_cfg.signature)
            # 将节点的指令设置为签名的文档字符串
            signature.__doc__ = node_cfg.instruction

            # 根据节点类型创建相应的 DSPy 模块
            if node_cfg.type == 'ChainOfThought':
                module = dspy.ChainOfThought(signature)
            elif node_cfg.type == 'ReAct':
                # 注意：这里简化了工具加载，实际实现应从 TOOL_REGISTRY 导入工具
                # tools = [TOOL_REGISTRY[t] for t in node_cfg.tools ...]
                module = dspy.ReAct(signature, tools=[])  # 简化演示
            else:
                # 默认为 Predict 模块
                module = dspy.Predict(signature)

            # 将创建的模块动态设置为当前实例的属性，以便后续访问
            self.__setattr__(node_name, module)

    def forward(self, **kwargs):
        """
        执行智能体工作流的主方法

        参数:
            **kwargs: 输入参数，将作为初始上下文传递给第一个节点

        返回:
            dspy.Prediction: 包含最终执行结果的预测对象

        工作流程:
            1. 初始化上下文和当前节点
            2. 循环执行节点直到达到结束节点或最大步数
            3. 在每个节点执行 DSPy 模块并更新上下文
            4. 根据流规则决定下一个节点（顺序流或分支流）
            5. 返回最终上下文作为预测结果

        注意:
            - 使用 max_steps 防止无限循环
            - 分支流根据上下文变量的值决定下一个节点
            - 顺序流直接跳转到下一个节点
        """
        # 初始化上下文（复制输入参数）
        context = kwargs.copy()
        # 从起始节点开始执行
        current_node_name = self.start_node
        steps = 0

        # 主执行循环：直到到达结束节点或超过最大步数
        while current_node_name != "end" and steps < self.max_steps:
            # 获取当前节点对应的 DSPy 模块
            module = getattr(self, current_node_name)
            # 调试输出：显示当前节点和上下文
            print(f"forward::{current_node_name}: {context}")
            # 执行当前节点的 DSPy 模块
            pred = module(**context)
            # 将预测结果更新到上下文中
            for k, v in pred.items():
                context[k] = v

            # --- 步骤 3: 路由逻辑 (基于 Schema 对象) ---
            # 获取当前节点的流规则
            flow_rule = self.agent_dna_config.flow.get(current_node_name)

            # 如果没有流规则，则结束执行
            if not flow_rule:
                current_node_name = "end"

            # 判断流规则类型：分支流
            elif isinstance(flow_rule, BranchFlow):
                # 从上下文中获取分支决策变量
                val = context.get(flow_rule.source_var, "").strip().upper()
                # 根据变量值选择分支，如果没有匹配则使用默认分支
                next_node = flow_rule.branches.get(val, flow_rule.default)
                current_node_name = next_node

            # 流规则类型：顺序流
            else:
                # 顺序流直接跳转到下一个节点
                current_node_name = flow_rule.next

            # 增加步数计数器
            steps += 1

        # 返回最终上下文作为预测结果
        return dspy.Prediction(**context)
