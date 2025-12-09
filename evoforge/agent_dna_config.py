# -*- coding: utf-8 -*-
"""
EvoForgePlus 智能体 DNA 配置模块

这个模块定义了智能体 DNA 配置的数据模型，使用 Pydantic 进行数据验证和序列化。
智能体 DNA 配置描述了智能体的结构，包括节点定义、流规则和图完整性。

核心组件：
1. NodeConfig: 定义单个节点的配置，包括类型、签名、指令和工具
2. SequenceFlow: 定义顺序流转规则（A -> B）
3. BranchFlow: 定义条件分支流转规则（A -> B or C）
4. AgentDNAConfig: 完整的智能体 DNA 配置，包含图完整性验证

使用 Pydantic 的优势：
- 自动数据验证和类型检查
- 序列化和反序列化 JSON 数据
- 模型验证器确保配置的逻辑正确性
"""

from typing import List, Dict, Optional, Literal, Union
from pydantic import BaseModel, Field, model_validator, field_validator


# --- 1. 节点定义 ---

class NodeConfig(BaseModel):
    """
    节点配置类：定义智能体工作流中的单个节点
    
    智能体由多个节点组成，每个节点对应一个 DSPy 模块，执行特定的任务。
    节点之间通过流规则连接，形成工作流。
    
    属性:
        type (Literal["ChainOfThought", "ReAct", "Predict"]): DSPy 模块类型
            - ChainOfThought: 思维链模块，生成推理步骤
            - ReAct: 推理-行动模块，结合推理和工具调用
            - Predict: 基础预测模块，直接生成输出
        signature (str): 输入输出签名，定义节点的输入和输出变量
            格式: "input_var1, input_var2 -> output_var1, output_var2"
            示例: "question -> answer", "topic -> content, critique"
        instruction (str): 节点的系统提示（System Prompt），指导节点如何执行任务
            这是给 LLM 的指令，描述节点的职责和期望行为
        tools (List[str]): 该节点可使用的工具列表，仅对 ReAct 类型节点有效
            工具名称必须与 tools.py 中的 TOOL_REGISTRY 中的键匹配
    
    验证:
        signature 必须包含 "->" 分隔符，确保正确区分输入和输出
    """
    type: Literal["ChainOfThought", "ReAct", "Predict"] = Field(..., description="DSPy 模块类型")
    signature: str = Field(..., description="输入输出签名，如 'question -> answer'")
    instruction: str = Field(..., description="节点的 System Prompt")
    tools: List[str] = Field(default_factory=list, description="该节点可使用的工具列表")

    @field_validator('signature')
    def validate_signature(cls, v):
        """
        验证签名格式
        
        参数:
            v (str): 待验证的签名字符串
            
        返回:
            str: 验证通过的签名字符串
            
        异常:
            ValueError: 如果签名中不包含 "->" 分隔符
            
        说明:
            签名必须包含 "->" 来分隔输入和输出变量，这是 DSPy 签名的标准格式。
        """
        if "->" not in v:
            raise ValueError(f"Signature must contain '->'. Got: {v}")
        return v


# --- 2. 流转规则定义 ---

class SequenceFlow(BaseModel):
    """
    顺序流转规则类：定义普通的顺序跳转（A -> B）
    
    顺序流转是最简单的流转类型，当节点执行完毕后，无条件跳转到下一个指定节点。
    
    属性:
        next (str): 下一个节点的名称，必须是已定义的节点名称或 "end"（结束）
        
    注意:
        - JSON 配置中可以不写 type 字段，Pydantic 会根据是否有 next 字段自动识别为顺序流转
        - "end" 是一个特殊节点，表示工作流结束
    """
    next: str = Field(..., description="下一个节点的名称")


class BranchFlow(BaseModel):
    """
    条件分支流转规则类：定义条件分支跳转（A -> B or C）
    
    分支流转根据上下文变量的值决定下一步跳转到哪个节点。
    类似于编程语言中的 switch/case 语句。
    
    属性:
        type (Literal["branch"]): 流转类型标识，固定为 "branch"
        source_var (str): 用于判断的上下文变量名，智能体执行时该变量的值将决定分支选择
        branches (Dict[str, str]): 分支映射字典，键为变量值，值为目标节点名称
            示例: {"PASS": "next_node", "FAIL": "retry_node"}
        default (str): 默认分支，当变量值不匹配任何分支时使用的目标节点，默认为 "end"
    """
    type: Literal["branch"]
    source_var: str = Field(..., description="用于判断的上下文变量名")
    branches: Dict[str, str] = Field(..., description="值与下一节点的映射，如 {'PASS': 'node_b'}")
    default: str = Field(default="end", description="未命中任何分支时的默认去向")


# 使用 Union 让 Pydantic 自动判断是哪种流转
FlowRule = Union[BranchFlow, SequenceFlow]
"""
流转规则联合类型

Pydantic 会自动根据提供的字段判断是哪种流转规则：
- 如果包含 type="branch" 字段，则解析为 BranchFlow
- 如果包含 next 字段，则解析为 SequenceFlow

这种设计使得 JSON 配置更加简洁：
顺序流转: {"next": "node_b"}
分支流转: {"type": "branch", "source_var": "decision", "branches": {"YES": "node_b", "NO": "node_c"}}
"""


# --- 3. 根配置定义 ---

class AgentDNAConfig(BaseModel):
    """
    智能体 DNA 完整配置类
    
    这是智能体配置的根模型，包含了智能体的完整定义：
    - 智能体标识和版本
    - 起始节点
    - 所有节点定义
    - 节点之间的流转规则
    
    属性:
        agent_id (str): 智能体唯一标识符，用于区分不同的智能体配置
        version (Union[str, int]): 配置版本，用于版本控制和升级，默认为 1
        start_node (str): 起始节点名称，智能体工作流从这个节点开始执行
        nodes (Dict[str, NodeConfig]): 节点字典，键为节点名称，值为节点配置
        flow (Dict[str, FlowRule]): 流转规则字典，键为源节点名称，值为流转规则
        
    验证:
        通过模型验证器检查图的完整性，确保所有跳转目标都是已定义的节点。
    """
    agent_id: str
    version: Union[str, int] = 1
    start_node: str
    nodes: Dict[str, NodeConfig]
    flow: Dict[str, FlowRule]

    @model_validator(mode='after')
    def check_graph_integrity(self):
        """
        核心图完整性校验
        
        这个验证器确保智能体工作流图的连通性和正确性：
        1. 起始节点必须存在于节点定义中
        2. 所有流转规则指向的节点必须存在于节点定义中或为 "end"
        3. 防止悬空引用和无效跳转
        
        返回:
            AgentDNAConfig: 验证通过的配置实例
            
        异常:
            ValueError: 如果发现任何图完整性错误
            
        算法说明:
            - 收集所有有效节点名称（包括 "end" 特殊节点）
            - 检查起始节点是否存在
            - 遍历所有流转规则，检查跳转目标是否存在
            - 对于分支流转，检查所有分支目标和默认目标
        """
        # 获取所有定义的节点名称，加上 'end' 作为合法终点
        valid_nodes = set(self.nodes.keys())
        valid_nodes.add("end")

        # 1. 检查 start_node
        if self.start_node not in valid_nodes:
            raise ValueError(f"Start node '{self.start_node}' is not defined in 'nodes'.")

        # 2. 检查 flow 中的每一个跳转
        for node_name, rule in self.flow.items():
            # 确保定义流转规则的节点本身是存在的
            if node_name not in valid_nodes and node_name != "start":  # start 有时是隐式的，但在你的设计中是指向第一个node
                # 这里我们假设 flow key 必须是 nodes 里定义的
                pass

            if isinstance(rule, SequenceFlow):
                if rule.next not in valid_nodes:
                    raise ValueError(f"Node '{node_name}' points to undefined node '{rule.next}'")

            elif isinstance(rule, BranchFlow):
                # 检查所有分支目标
                for branch_key, target_node in rule.branches.items():
                    if target_node not in valid_nodes:
                        raise ValueError(
                            f"Node '{node_name}' branch '{branch_key}' points to undefined node '{target_node}'")
                # 检查默认目标
                if rule.default not in valid_nodes:
                    raise ValueError(f"Node '{node_name}' default branch points to undefined node '{rule.default}'")

        return self
