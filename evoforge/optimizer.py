"""
EvoForgePlus 优化器模块：双环进化 SOP 实现

这个模块实现了 EvoForgePlus 的双环进化优化策略，包括：
1. 内环优化（Prompt/Few-Shot 优化）：使用 BootstrapFewShot 优化智能体的提示和示例
2. 外环优化（架构变异）：使用元架构师智能体（MetaArchitect）修改智能体 DNA 配置

核心特点：
- 分层架构：清晰区分内环（Prompt/Few-Shot）和外环（JSON Mutation）优化
- 元架构师智能体：使用 DSPy 构建专门用于分析错误并修改 JSON 配置的智能体
- 诊断机制：在触发外环进化前，先对内环的失败案例进行汇总分析
- 容错与回滚：如果新生成的架构无法通过 Pydantic 校验，自动回滚或重试

该模块遵循生产级代码标准，具有完整的日志记录和错误处理机制。
"""

import dspy
import json
import logging
from typing import List, Callable, Tuple
from dspy.teleprompt import BootstrapFewShot
from pydantic import ValidationError

from evoforge.engine import GraphAgent
from evoforge.agent_dna_config import AgentDNAConfig

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EvoForge")


# ==============================================================================
# 1. 定义 Meta-Architect (元架构师)：负责外环进化
# ==============================================================================

class ArchitectureRefinerSignature(dspy.Signature):
    """
    元架构师智能体的签名定义
    
    这个智能体负责分析当前智能体配置和失败案例，然后修改 JSON 配置以解决性能瓶颈。
    
    可用变异策略：
    1. 节点拆分 (Splitting): 如果某节点任务过重，将其拆分为 Planner -> Executor
    2. 工具挂载 (Tooling): 如果涉及计算或搜索，将 ChainOfThought 改为 ReAct 并挂载工具
    3. 增加循环 (Looping): 如果质量不稳定，增加 Critic 节点和分支逻辑
    
    输入字段:
        current_dna_json: 当前的 Agent JSON 配置
        diagnosis_report: 性能评估报告和失败案例分析
        
    输出字段:
        refined_dna_json: 优化后的完整 Agent JSON 配置代码块
        mutation_reason: 修改原因的简短说明
        
    要求:
        - 输出必须是严格合法的 JSON
        - 严格遵守输入中描述的 Schema 格式
    """
    current_dna_json = dspy.InputField(desc="当前的 Agent JSON 配置")
    diagnosis_report = dspy.InputField(desc="性能评估报告和失败案例分析")

    refined_dna_json = dspy.OutputField(desc="优化后的完整 Agent JSON 配置代码块")
    mutation_reason = dspy.OutputField(desc="修改原因的简短说明")


class MetaArchitect(dspy.Module):
    """
    元架构师智能体类
    
    这个类使用 ChainOfThought 模块来构建一个能够思考并修改智能体架构的智能体。
    它接收当前 DNA 配置和诊断报告，然后输出优化后的配置和修改原因。
    """
    def __init__(self):
        """
        初始化元架构师智能体
        
        使用 ChainOfThought 模块，让架构师在修改前先进行思考。
        """
        super().__init__()
        # 使用 ChainOfThought 让架构师在修改前先思考
        self.prog = dspy.ChainOfThought(ArchitectureRefinerSignature)

    def forward(self, current_dna_json, diagnosis_report):
        """
        执行架构优化
        
        参数:
            current_dna_json (str): 当前 DNA 配置的 JSON 字符串
            diagnosis_report (str): 诊断报告字符串
            
        返回:
            dspy.Prediction: 包含优化后配置和修改原因的预测结果
        """
        return self.prog(current_dna_json=current_dna_json, diagnosis_report=diagnosis_report)


# ==============================================================================
# 2. EvoOptimizer (双环进化主控制器)
# ==============================================================================

class EvoOptimizer:
    """
    进化优化器主类
    
    这个类实现了双环进化策略，通过内环和外环的交替优化来提升智能体性能。
    
    属性:
        cur_agent_dna_config (AgentDNAConfig): 当前智能体 DNA 配置
        trainset (List[dspy.Example]): 训练数据集
        metric_func (Callable): 评估函数
        max_generations (int): 最大外环进化代数
        score_threshold (float): 目标分数，达到即停止
        meta_architect (MetaArchitect): 元架构师智能体实例
        history (list): 进化历史记录
    """
    def __init__(self,
                 agent_dna_config: AgentDNAConfig,
                 trainset: List[dspy.Example],
                 metric_func: Callable,
                 max_generations: int = 3,
                 score_threshold: float = 90.0):
        """
        初始化进化优化器
        
        参数:
            agent_dna_config (AgentDNAConfig): 初始 Agent DNA 配置
            trainset (List[dspy.Example]): 训练数据集，用于优化和评估
            metric_func (Callable): 评估函数，用于计算智能体得分
            max_generations (int): 最大外环进化代数，默认为3
            score_threshold (float): 目标分数阈值，达到即停止进化，默认为90.0
            
        初始化步骤:
            1. 保存配置和参数
            2. 初始化元架构师智能体
            3. 初始化历史记录列表
        """
        self.cur_agent_dna_config: AgentDNAConfig = agent_dna_config
        self.trainset = trainset
        self.metric_func = metric_func
        self.max_generations = max_generations
        self.score_threshold = score_threshold

        # 初始化 Meta-Agent
        self.meta_architect = MetaArchitect()

        # 历史记录
        self.history = []

    def evolve(self) -> Tuple[dspy.Module, AgentDNAConfig]:
        """
        [SOP 主流程] 执行双环进化
        
        进化流程:
            1. Stage 1: 初始化与验证 - 验证当前 DNA 配置并创建智能体实例
            2. Stage 2: 内环进化 - 使用 BootstrapFewShot 优化提示和少量示例
            3. Stage 3: 评估与诊断 - 评估优化后智能体的性能并生成诊断报告
            4. Stage 4: 外环进化 - 如果性能不足，使用元架构师修改 DNA 配置
            
        返回:
            Tuple[dspy.Module, AgentDNAConfig]: 优化后的智能体实例和最终的 DNA 配置
            
        注意:
            - 进化过程会持续直到达到目标分数或最大进化代数
            - 每代进化都会记录历史以便分析和调试
        """
        logger.info(">>> 🧬 EvoForge Evolution Started")

        optimized_agent = None

        for generation in range(self.max_generations):
            logger.info(f"\n========== Generation {generation} ==========")

            # --- Stage 1: 初始化与验证 ---
            try:
                agent = GraphAgent(self.cur_agent_dna_config)
                logger.info("✅ Generation DNA validated.")
            except ValidationError as e:
                logger.error(f"❌ Invalid DNA in generation {generation}: {e}")
                break

            # --- Stage 2: 内环进化 (Prompt/Few-Shot Optimization) ---
            optimized_agent = self._run_inner_loop(agent)

            # --- Stage 3: 评估与诊断 ---
            score, bad_cases = self._evaluate_agent(optimized_agent)
            logger.info(f"📊 Generation {generation} Score: {score:.2f}%")

            # 记录历史
            self.history.append({
                "gen": generation,
                "config": self.cur_agent_dna_config.copy(),
                "score": score
            })

            # 决策：是否达到目标？
            if score >= self.score_threshold:
                logger.info("🎉 Target Score Reached! Stopping evolution.")
                return optimized_agent, self.cur_agent_dna_config

            # --- Stage 4: 外环进化 (Architecture Mutation) ---
            if generation < self.max_generations - 1:
                logger.info("🔧 Score insufficient. Triggering Outer Loop (Mutation)...")
                new_config = self._run_outer_loop(score, bad_cases)
                if new_config:
                    self.cur_agent_dna_config = new_config
                else:
                    logger.warning("Mutation failed, stopping early.")
                    break

        logger.info("🏁 Evolution finished (Max generations reached).")
        return optimized_agent, self.cur_agent_dna_config

    def _run_inner_loop(self, agent) -> dspy.Module:
        """
        [SOP Stage 2] 内环：利用 BootstrapFewShot 优化 Prompt
        
        这个阶段使用 DSPy 的 BootstrapFewShot 方法优化智能体的提示和少量示例。
        
        参数:
            agent (dspy.Module): 当前代的智能体实例
            
        返回:
            dspy.Module: 优化后的智能体实例（如果优化失败则返回原智能体）
            
        注意:
            - max_bootstrapped_demos: 每个 predictor 最多生成的 few-shot 数量
            - max_labeled_demos: 从训练集直接采样的数量（设置为0表示不使用预标记示例）
        """
        logger.info("   [Inner Loop] Optimizing Prompts & Few-Shots...")

        # 配置 BootstrapFewShot
        # max_bootstrapped_demos: 每个 predictor 最多生成的 few-shot 数量
        # max_labeled_demos: 从训练集直接采样的数量
        teleprompter = BootstrapFewShot(
            metric=self.metric_func,
            max_bootstrapped_demos=2,
            max_labeled_demos=0
        )

        # 编译 (Compile)
        try:
            compiled_agent = teleprompter.compile(agent, trainset=self.trainset)
            return compiled_agent
        except Exception as e:
            logger.warning(f"   [Inner Loop] Optimization warning: {e}. Returning original agent.")
            return agent

    def _evaluate_agent(self, agent) -> Tuple[float, str]:
        """
        [SOP Stage 3] 评估并生成诊断报告
        
        这个阶段评估优化后智能体的性能，并生成详细的诊断报告用于外环进化。
        
        参数:
            agent (dspy.Module): 需要评估的智能体实例
            
        返回:
            Tuple[float, str]: 得分（百分比）和诊断报告字符串
            
        评估过程:
            1. 遍历训练集中的所有示例
            2. 使用智能体进行预测
            3. 使用评估函数检查预测是否正确
            4. 记录所有失败案例的详细信息
            5. 计算总体得分
            6. 生成包含得分和典型失败案例的诊断报告
        """
        logger.info("   [Evaluation] Running validation...")
        total = len(self.trainset)
        correct = 0
        bad_cases_log = []

        for ex in self.trainset:
            # 运行预测
            try:
                pred = agent(**ex.inputs())
                passed = self.metric_func(ex, pred, None)
                if passed:
                    correct += 1
                else:
                    # 记录失败案例用于 Meta-Agent 分析
                    case_info = f"Input: {ex.inputs()}\nExpected: {getattr(ex, 'answer', 'N/A')}\nGot: {getattr(pred, 'answer', 'N/A')}"
                    # 如果有 trace 路径，也记录下来
                    if hasattr(pred, '_trace_path'):
                        case_info += f"\nPath: {pred._trace_path}"
                    bad_cases_log.append(case_info)
            except Exception as e:
                bad_cases_log.append(f"Runtime Error: {e}")

        score = (correct / total) * 100 if total > 0 else 0

        # 生成诊断报告 summary
        diagnosis = f"Current Score: {score:.2f}%\nFailure Count: {len(bad_cases_log)}\n"
        if bad_cases_log:
            diagnosis += "Top 3 Bad Cases:\n" + "\n---\n".join(bad_cases_log[:3])

        return score, diagnosis

    def _run_outer_loop(self, current_score, diagnosis_report) -> AgentDNAConfig:
        """
        [SOP Stage 4] 外环：调用 Meta-Agent 修改 JSON
        
        这个阶段使用元架构师智能体来分析当前配置和诊断报告，然后生成优化后的配置。
        
        参数:
            current_score (float): 当前代智能体的得分
            diagnosis_report (str): 诊断报告字符串
            
        返回:
            AgentDNAConfig: 优化后的智能体 DNA 配置，如果失败则返回 None
            
        处理流程:
            1. 将当前 DNA 配置转换为 JSON 字符串
            2. 调用元架构师智能体生成优化后的配置
            3. 清理 LLM 可能输出的 Markdown 代码块
            4. 解析 JSON 并验证其是否符合 AgentDNAConfig 的 Pydantic 模型
            5. 返回验证通过的配置，或在出现错误时返回 None
        """
        logger.info("   [Outer Loop] Meta-Architect is redesigning the agent...")

        # 准备上下文
        current_dna_str = json.dumps(self.cur_agent_dna_config, indent=2, ensure_ascii=False)

        # 调用 Meta-Architect
        try:
            # 使用 MetaArchitect (ChainOfThought)
            prediction = self.meta_architect(
                current_dna_json=current_dna_str,
                diagnosis_report=diagnosis_report
            )

            logger.info(f"   [Outer Loop] Architect's Thought: {prediction.mutation_reason}")

            # 清洗并解析 JSON (防止 LLM 输出 Markdown 代码块)
            raw_json = prediction.refined_dna_json.strip()
            if raw_json.startswith("```"):
                raw_json = raw_json.strip("`").replace("json\n", "").replace("json", "")

            new_config = json.loads(raw_json)

            # [SOP Stage 1 Re-validation] 立即验证新生成的配置是否合法
            agent_dna_config = AgentDNAConfig(**new_config)
            logger.info("   [Outer Loop] Mutation successful & Validated.")
            return agent_dna_config

        except json.JSONDecodeError:
            logger.error("   [Outer Loop] Failed: Architect produced invalid JSON.")
            return None
        except ValidationError as e:
            logger.error(f"   [Outer Loop] Failed: New architecture violates Schema. {e}")
            return None
        except Exception as e:
            logger.error(f"   [Outer Loop] Unexpected error: {e}")
            return None
