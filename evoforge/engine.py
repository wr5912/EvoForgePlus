import dspy
import yaml
import os
import importlib
from typing import Dict, Any, Type, Callable, Optional, List


# =============================================================================
# 1. 辅助工具: 动态加载器与解析器
# =============================================================================

class ToolResolver:
    """负责将 tools.yaml 中的字符串路径解析为可执行的 Python 函数"""

    @staticmethod
    def import_tool(path_str: str) -> Callable:
        """
        例如: "lib.math_utils.calculate_sum" -> 对应的函数对象
        """
        try:
            module_path, func_name = path_str.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, func_name)
        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(f"无法加载工具: {path_str}. 错误: {e}")


class DNALoader:
    """负责加载 YAML 配置文件并合并为一个完整的 Config 字典"""

    @staticmethod
    def load(entry_yaml_path: str) -> Dict[str, Any]:
        if not os.path.exists(entry_yaml_path):
            raise FileNotFoundError(f"找不到入口配置文件: {entry_yaml_path}")

        base_dir = os.path.dirname(entry_yaml_path)

        # 1. 加载主清单 (agent.yaml)
        with open(entry_yaml_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)

        full_config = {
            "metadata": main_config,  # 保存 id, version 等元数据
            "types": {},
            "signatures": {},
            "tools": {},
            "knowledge": {},
            "modules": {},
            "workflow": {}
        }

        # 2. 递归加载 includes
        includes = main_config.get("includes", {})

        # 定义映射关系: include_key -> config_key
        # 例如: includes 中的 'signatures' 文件内容加载到 full_config['signatures']
        section_map = {
            "types": "types",
            "signatures": "signatures",
            "tools": "tools",
            "knowledge": "knowledge",
            "modules": "modules",
            "workflow": "workflow"
        }

        for inc_key, rel_path in includes.items():
            target_section = section_map.get(inc_key)
            if not target_section:
                continue

            full_path = os.path.join(base_dir, rel_path)
            if not os.path.exists(full_path):
                print(f"Warning: Included file not found: {full_path}")
                continue

            with open(full_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)

            # 根据 YAML 文件结构合并数据
            # 假设子 YAML 文件的根键通常就是 section 名 (如 tools.yaml 里是以 tools: 开头)
            if content:
                # 如果文件内容包含根 key (如 tools: ...), 取其值；否则直接合并整个 content
                data_to_merge = content.get(target_section, content)

                # Flow 通常是一个嵌套字典，直接替换或更新
                if target_section == "workflow":
                    full_config["workflow"] = data_to_merge
                # 其他部分通常是字典列表，进行 update
                elif isinstance(full_config[target_section], dict) and isinstance(data_to_merge, dict):
                    full_config[target_section].update(data_to_merge)

        return full_config


class SignatureFactory:
    """动态创建 DSPy Signature 类"""

    @staticmethod
    def create(name: str, sig_config: Dict[str, Any]) -> Type[dspy.Signature]:
        # 1. 准备类属性 (Docstring 是关键)
        class_attrs = {
            "__doc__": sig_config.get("docstring", "").strip()
        }

        # 2. 动态添加 Inputs
        for field_name, meta in sig_config.get("inputs", {}).items():
            desc = meta.get("desc", "") if isinstance(meta, dict) else str(meta)
            class_attrs[field_name] = dspy.InputField(desc=desc)

        # 3. 动态添加 Outputs
        for field_name, meta in sig_config.get("outputs", {}).items():
            desc = meta.get("desc", "") if isinstance(meta, dict) else str(meta)
            class_attrs[field_name] = dspy.OutputField(desc=desc)

        # 4. 构造类
        return type(name, (dspy.Signature,), class_attrs)


# =============================================================================
# 2. 核心引擎: GraphAgent
# =============================================================================

class GraphAgent(dspy.Module):
    """
    基于 YAML 配置的动态图智能体执行器。

    它负责：
    1. 加载并组装 components (Signature + Tools -> Modules)
    2. 维护 Tool Registry
    3. 执行 workflow 定义的图逻辑
    """

    def __init__(self, agent_yaml_path: str):
        super().__init__()

        # --- 步骤 1: 加载完整配置 (Manifest -> All Layers) ---
        self.config = DNALoader.load(agent_yaml_path)

        # --- 步骤 2: 初始化资源 (Tools) ---
        # 将 tools.yaml 中的定义解析为实际的 Python 函数对象
        self.tool_registry = {}
        for tool_name, tool_cfg in self.config.get("tools", {}).items():
            path_str = tool_cfg.get("path")
            if path_str:
                try:
                    func = ToolResolver.import_tool(path_str)
                    # 可以在这里包装 docstring，如果 YAML 里有 desc
                    if "desc" in tool_cfg:
                        func.__doc__ = tool_cfg["desc"]
                    self.tool_registry[tool_name] = func
                except Exception as e:
                    print(f"Error loading tool '{tool_name}': {e}")

        # --- 步骤 3: 动态构建 Signatures ---
        self.sig_classes = {}
        for name, sig_cfg in self.config.get("signatures", {}).items():
            self.sig_classes[name] = SignatureFactory.create(name, sig_cfg)

        # --- 步骤 4: 实例化 Modules (Components Layer) ---
        self.modules_config = self.config.get("modules", {})

        for node_name, mod_cfg in self.modules_config.items():
            # 4.1 获取 Signature 类
            sig_name = mod_cfg.get("signature")
            if sig_name in self.sig_classes:
                signature = self.sig_classes[sig_name]
            else:
                # 容错：允许内联字符串定义 (e.g. "q -> a")
                signature = dspy.Signature(sig_name)
                signature.__doc__ = mod_cfg.get("instruction", "")

            # 4.2 根据类型实例化 DSPy 模块
            mod_type = mod_cfg.get("type", "Predict")

            if mod_type == 'ChainOfThought':
                module = dspy.ChainOfThought(signature)

            elif mod_type == 'ReAct':
                # 关键：从 registry 中解析工具
                tool_refs = mod_cfg.get("tools", [])
                tools_for_node = []
                for t_name in tool_refs:
                    if t_name in self.tool_registry:
                        tools_for_node.append(self.tool_registry[t_name])
                    else:
                        print(f"Warning: Module '{node_name}' refers to unknown tool '{t_name}'")

                module = dspy.ReAct(signature, tools=tools_for_node)

            elif mod_type == 'Predict':
                module = dspy.Predict(signature)

            else:
                raise ValueError(f"Unsupported module type: {mod_type}")

            # 4.3 注册为属性 (DSPy 优化器需要能访问到这些属性)
            self.__setattr__(node_name, module)

        # --- 步骤 5: 准备流程控制 ---
        self.flow_config = self.config.get("workflow", {})
        self.start_node = self.flow_config.get("start_node")
        self.rules = self.flow_config.get("rules", {})
        self.max_steps = 15

    def forward(self, **kwargs):
        """
        执行 workflow.yaml 定义的工作流
        """
        context = kwargs.copy()
        current_node_name = self.start_node
        steps = 0

        # 记录执行路径 (用于调试和优化)
        trace_path = []

        print(f"\n🚀 Agent Started. Input keys: {list(context.keys())}")

        while current_node_name != "end" and steps < self.max_steps:
            trace_path.append(current_node_name)

            # 1. 检查节点是否存在
            if not hasattr(self, current_node_name):
                print(f"Error: Node '{current_node_name}' not defined in modules.")
                break

            module = getattr(self, current_node_name)

            # 2. 执行模块
            print(f"👉 Step {steps}: Running [{current_node_name}]")
            try:
                # DSPy 会自动从 context 匹配参数
                prediction = module(**context)

                # 更新上下文
                for k, v in prediction.items():
                    context[k] = v
            except Exception as e:
                print(f"❌ Error executing node '{current_node_name}': {e}")
                break

            # 3. 路由逻辑 (Flow Control)
            rule = self.rules.get(current_node_name)

            if not rule:
                # 如果没有定义后续规则，默认结束
                current_node_name = "end"

            else:
                rule_type = rule.get("type", "sequence")  # 默认为顺序流

                # --- 分支流 (Branch) ---
                if rule_type == "branch":
                    source_var = rule.get("source_var")
                    val = str(context.get(source_var, "")).strip()

                    # 查找匹配的分支
                    branches = rule.get("branches", {})
                    # 简单匹配策略：完全匹配 或 包含匹配 (视业务需求而定)
                    # 这里使用包含匹配以提高鲁棒性 (LLM 输出可能包含标点)
                    next_node = rule.get("default", "end")

                    found_match = False
                    for key, target in branches.items():
                        if key.upper() in val.upper():  # 忽略大小写
                            next_node = target
                            found_match = True
                            print(f"   🔀 Branch: '{val}' matches '{key}' -> Goto {target}")
                            break

                    if not found_match:
                        print(f"   🔀 Branch: '{val}' no match -> Goto Default ({next_node})")

                    current_node_name = next_node

                # --- 顺序流 (Sequence) ---
                else:
                    # type: sequence
                    current_node_name = rule.get("next", "end")

            steps += 1

        # 将 trace 路径注入结果，方便外层分析
        context["_trace_path"] = trace_path

        if steps >= self.max_steps:
            print("⚠️ Max steps reached. Terminating.")

        return dspy.Prediction(**context)


# ==========================================
# 4. 使用示例
# ==========================================
if __name__ == "__main__":
    # 直接传入入口 yaml 文件路径
    agent = GraphAgent("DNA/agent.yaml")

    # 运行
    agent(user_query="My computer is broken", full_document_text="Manual...")