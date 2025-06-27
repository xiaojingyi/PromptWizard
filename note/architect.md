# PromptWizard 项目架构分析报告

## 1. 项目概述

`PromptWizard` 是一个先进的、自动化的提示工程（Prompt Engineering）框架，其设计目标是通过程序化的方式系统性地优化大语言模型（LLM）的提示（Prompt）。

它旨在解决手动调试和优化提示时面临的巨大挑战，例如耗时、依赖直觉、难以复现等。通过引入一种由反馈驱动的自适应优化机制，`PromptWizard` 能够显著提升 LLM 在特定任务上的准确性和可靠性。

## 2. 核心功能与特性

该框架的核心竞争力在于其创新的、多层次的提示优化策略。

*   **指令优化 (Instruction Optimization)**: 框架的核心是一个迭代循环，它首先通过“变异”生成多种不同风格的候选指令，然后使用一小部分数据对它们进行“评分”，最后通过独特的“批判与精炼”机制，让 LLM 自我反思并改进表现不佳的指令。
*   **示例优化 (Example Optimization)**: 除了指令本身，框架还能优化用于上下文学习（In-Context Learning）的 few-shot 示例。它能够从数据集中筛选出最有效（或最具挑战性）的样本，甚至能生成全新的“合成示例”来弥补训练数据的不足。
*   **思维链生成 (Chain-of-Thought Generation)**: 为了提升模型在复杂推理任务上的表现，`PromptWizard` 可以自动为上下文示例生成详细的、分步的推理过程（即思维链，CoT）。
*   **反馈驱动机制 (Feedback-Driven Mechanism)**: 这是 `PromptWizard` 的灵魂。它并非简单地尝试不同的提示，而是让 LLM 对自己的输出（生成的指令或答案）进行“批判”，找出潜在的问题，然后基于这些批判性反馈来“精炼”和改进提示。
*   **配置驱动 (Configuration-Driven)**: 整个优化流程高度可配置。用户可以通过 `.yaml` 文件精确控制优化的各个方面（如迭代次数、批处理大小、模型选择等），并通过 `.env` 文件安全地管理 API 密钥和端点。

## 3. 项目结构分析

项目的代码结构清晰，职责分离明确，易于理解和扩展。

*   `promptwizard/`: 核心 Python 包。
    *   `glue/`: 框架的“粘合层”，是所有核心逻辑的所在地。
        *   `common/`: 存放整个项目可复用的通用模块，包括用于解析配置文件的**基类** (`base_classes.py`)、封装 LLM 调用的**LLM 管理器** (`llm_mgr.py`) 以及其他工具函数。
        *   `promptopt/`: 提示优化的主逻辑模块。
            *   `runner.py`: 命令行执行入口。
            *   `techniques/`: 存放具体的优化算法实现。目前核心是 `critique_n_refine`（批判与精炼）。这种结构使其易于在未来添加新的优化技术。
        *   `paramlogger/`: 一个轻量级的日志模块，用于在优化过程中记录详细的参数、中间结果和 LLM 调用日志，便于调试和分析。
*   `demos/`: 包含了多个真实世界数据集（如 `GSM8k`, `SVAMP`, `AQUARAT`）的端到端运行示例。这些 `demo.ipynb` notebook 是理解和学习如何使用该框架的最佳起点。
*   `docs/`: 存放项目文档、图片和静态网站资源。

## 4. 核心工作流程

`CritiqueNRefine` 技术是 `PromptWizard` 的核心，其工作流程可以通过以下图表来可视化：

```mermaid
graph TD
    A[开始: 输入基础指令和数据集] --> B{指令优化循环};
    B --> C[1. 指令变异: LLM生成多个候选指令];
    C --> D[2. 评分: 在部分数据集上评估候选指令];
    D --> E[3. 筛选: 选择Top-N个最佳指令];
    E --> F{是否需要精炼?};
    F -- 是 --> G[4. 批判与精炼: LLM分析错误并改进指令];
    G --> B;
    F -- 否/完成 --> H{示例优化};
    H --> I[1. 筛选/生成合成示例];
    I --> J[2. (可选)为示例生成CoT推理];
    J --> K[最终组装];
    K --> L[输出: 优化后的指令 + 示例 + 专家身份];

    subgraph "核心反馈循环"
        direction LR
        G
    end
```

这个流程清晰地展示了从一个简单的基础指令开始，如何通过多轮的“生成-评估-反馈-改进”循环，逐步演进出一个高度优化的、任务专属的复杂提示。

## 5. 关键类与抽象

*   `CritiqueNRefine`: 位于 `glue/promptopt/techniques/critique_n_refine/core_logic.py`，是核心优化算法的实现者，编排了上述工作流程中的所有步骤。
*   `DatasetSpecificProcessing`: 位于 `glue/promptopt/techniques/common_logic.py`。这是一个至关重要的抽象基类。用户若想在自己的数据集上使用 `PromptWizard`，需要继承这个类并实现两个核心方法：`extract_final_answer`（如何从 LLM 的输出中解析出最终答案）和 `access_answer`（如何评估解析出的答案是否正确）。这是框架可扩展性的关键所在。
*   `LLMConfig`, `SetupConfig`: 位于 `glue/common/base_classes.py`。这些 dataclass 直接映射自 `.yaml` 配置文件，使得配置的管理和使用非常方便和类型安全。
*   `CritiqueNRefinePromptPool`: 位于 `glue/promptopt/techniques/critique_n_refine/base_classes.py`。它从 `prompt_pool.yaml` 加载并存储了所有用于指导 LLM 进行“元任务”（如批判、精炼、生成风格等）的“元提示”，将提示内容与业务逻辑解耦。

## 6. 配置与扩展

*   **配置**:
    *   `promptopt_config.yaml`: 定义了优化过程的超参数，例如 `mutate_refine_iterations` (迭代次数), `few_shot_count` (上下文示例数量) 等。
    *   `.env`: 用于存储敏感信息，如 `OPENAI_API_KEY` 和 `AZURE_OPENAI_ENDPOINT`。
*   **扩展**:
    *   要支持一个新的自定义数据集，主要工作就是创建一个继承自 `DatasetSpecificProcessing` 的新类，并根据新数据集的格式和评估标准，实现其抽象方法。`demos/` 目录下的示例提供了清晰的参考。

## 7. 总结

`PromptWizard` 是一个设计精良、功能强大的提示工程自动化框架。它将复杂的提示优化过程抽象成一个结构化、可配置、可扩展的流程。其核心的“批判与精炼”反馈机制，充分利用了 LLM 自身的智能，实现了提示的自我进化。对于任何需要为特定任务构建高性能、高鲁棒性提示的团队或个人来说，`PromptWizard` 都是一个极具价值的工具。