# PromptWizard 详细操作手册

本手册旨在提供一个全面、详细的指南，指导用户如何利用 `PromptWizard` 框架为自定义任务和数据集优化提示（Prompts）。我们将以一个端到端的流程，讲解从环境设置到最终评估的每一个步骤。

## 1. 环境准备

在开始之前，请确保您的开发环境已正确设置。

### 1.1. 克隆并安装

首先，克隆 `PromptWizard` 的官方仓库，并安装必要的依赖。

```bash
# 1. 克隆仓库
git clone https://github.com/microsoft/PromptWizard.git
cd PromptWizard

# 2. 创建并激活 Python 虚拟环境 (推荐)
python -m venv venv
# Windows
# venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. 以可编辑模式安装项目
# 这将同时安装所有必需的依赖库
pip install -e .
```

### 1.2. 配置环境变量

`PromptWizard` 需要通过 API 与大语言模型（LLM）进行交互。您需要提供相应的凭证。

1.  在项目根目录下，复制或重命名一个 `.env.template` 文件为 `.env`。
2.  打开 `.env` 文件并填入您的凭证。

```dotenv
# .env 文件示例

# 首先，决定使用 OpenAI API 还是 Azure OpenAI 服务
# 设置为 "True" 使用 OpenAI, 设置为 "False" 使用 Azure
USE_OPENAI_API_KEY="True"

# --- 如果使用 OpenAI API (USE_OPENAI_API_KEY="True") ---
# 填入您的 OpenAI API 密钥和希望使用的模型名称
OPENAI_API_KEY="sk-..."
OPENAI_MODEL_NAME="gpt-4o"

# --- 如果使用 Azure OpenAI (USE_OPENAI_API_KEY="False") ---
# 填入您的 Azure OpenAI 服务凭证
AZURE_OPENAI_ENDPOINT="https://YOUR_ENDPOINT.openai.azure.com/"
OPENAI_API_VERSION="2024-02-01"
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="YOUR_DEPLOYMENT_NAME"
```

## 2. 核心概念: 数据集适配器

为了让 `PromptWizard` 能够理解和处理您的自定义数据集，您必须创建一个“数据集适配器”类。这个类继承自框架提供的 `DatasetSpecificProcessing` 抽象基类，并负责处理所有与特定数据集相关的逻辑。

这是您将编写的最重要的代码。

### `DatasetSpecificProcessing` 详解

这个类位于 [`promptwizard/glue/promptopt/techniques/common_logic.py`](promptwizard/glue/promptopt/techniques/common_logic.py:19)。您需要在一个执行脚本中（如 `demo.ipynb` 或 `run.py`）定义一个继承自它的子类，并实现以下关键方法：

#### a) `dataset_to_jsonl`

此方法负责将您的原始数据集（无论是来自 Hugging Face、本地文件还是数据库）转换为框架所需的标准 `.jsonl` 格式。

*   **输入**: 原始数据集。
*   **输出**: 一个 `.jsonl` 文件，其中每一行都是一个 JSON 对象。
*   **JSON 对象格式**:
    ```json
    {
      "question": "这是需要LLM回答的问题文本。",
      "answer": "这是带有详细推理过程的标准答案（思维链）。",
      "final_answer": "这是从标准答案中提取出的、用于精确评估的最终答案。"
    }
    ```

#### b) `extract_final_answer`

此方法定义了如何从 LLM 生成的、可能包含冗长解释的自由文本输出中，精确地提取出最终答案。

*   **输入**: LLM 的完整响应字符串。
*   **输出**: 提取出的最终答案字符串。
*   **实现技巧**:
    *   **使用分隔符**: 在您的提示中，要求 LLM 将最终答案包裹在特殊标签中，如 `<ANS_START>123<ANS_END>`。这样提取会变得非常简单。
    *   **正则表达式**: 对于数字、选项（A, B, C）等格式化答案，使用正则表达式是常用且高效的方法。

#### c) `access_answer` (可选，但强烈建议重写)

此方法负责比较模型提取出的答案和数据集中的标准答案，并返回一个布尔值表示是否正确。

*   **输入**: LLM 的输出，标准答案。
*   **输出**: `(is_correct: bool, predicted_answer: str)` 元组。
*   **为何要重写**: 默认实现只是简单的字符串比较。对于大多数任务，这远远不够。例如：
    *   **数学题**: 您需要将字符串 "1,024" 和 "1024.0" 都视为与 `1024` 相等。
    *   **分类题**: 您可能需要忽略大小写和多余的空格。
    *   **开放式问题**: 您可能需要使用另一个 LLM 作为“裁判”（LLM-as-a-Judge）来评估语义上的等价性。

## 3. 项目结构设置

为了保持代码的整洁和可维护性，我们建议您为每个新任务创建一个独立的目录，模仿 `demos/` 下的结构。

```
my_custom_task/
├── configs/
│   ├── promptopt_config.yaml  # 优化过程的超参数
│   └── setup_config.yaml      # 实验设置
├── data/
│   ├── train.jsonl            # 转换后的训练数据
│   └── test.jsonl             # 转换后的测试数据
├── .env                       # API 凭证
└── run.ipynb                  # 主执行脚本 (或 .py 文件)
```

## 4. 配置文件详解 (`promptopt_config.yaml`)

这个文件是 `PromptWizard` 的控制面板，定义了优化流程的所有超参数。

以下是一些最关键的参数：

*   `task_description`: (字符串) 对任务的总体描述，例如 "You are a mathematics expert. You will be given a mathematics problem which you need to solve"。这会成为最终提示的一部分。
*   `base_instruction`: (字符串) 优化的起点指令，例如 "Lets think step by step."。框架将在这个基础上进行变异和精炼。
*   `answer_format`: (字符串) 明确告知 LLM 应如何格式化其答案，以便您的 `extract_final_answer` 函数能够正确解析。
*   `mutate_refine_iterations`: (整数) 核心优化循环的迭代次数。建议值为 3 或 5。这是模型性能和计算成本之间的主要权衡点。
*   `few_shot_count`: (整数) 最终生成的提示中应包含多少个上下文学习示例。如果设为 0，则生成 Zero-Shot 提示。
*   `seen_set_size`: (整数) 从训练集中随机抽取多少个样本用于整个优化过程。建议值为 20-50。
*   `generate_reasoning`: (布尔值) 是否让 LLM 为上下文示例自动生成思维链（CoT）推理过程。默认为 `true`，强烈建议开启以提升性能。
*   `generate_expert_identity`: (布尔值) 是否让 LLM 自动生成一个“专家身份”（如“你是一位资深Python程序员”）作为 System Prompt。默认为 `true`。

## 5. 端到端执行流程

以下是一个典型的执行流程，整合了上述所有概念。

### 步骤 1: 编写主脚本 (`run.ipynb` 或 `run.py`)

```python
import os
import sys
from dotenv import load_dotenv
from datasets import load_dataset

# 添加项目根目录到 Python 路径
sys.path.insert(0, "../../")

from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing

# --- 1. 定义您的数据集适配器 ---
class MyTaskProcessor(DatasetSpecificProcessing):
    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any) -> None:
        # ... 实现您的数据转换逻辑 ...
        pass

    def extract_final_answer(self, answer: str) -> str:
        # ... 实现您的答案提取逻辑 ...
        pass
    
    # (可选) 重写 access_answer
    # def access_answer(self, llm_output: str, gt_answer: str) -> (bool, Any):
    #     # ... 实现您的评估逻辑 ...
    #     pass

# --- 2. 准备数据 ---
# 加载环境变量
load_dotenv(override=True)

# 实例化处理器
my_processor = MyTaskProcessor()

# 创建数据目录
if not os.path.exists("data"):
    os.mkdir("data")

# 加载原始数据并转换为 .jsonl 格式
# (这里以 Hugging Face 为例)
raw_dataset = load_dataset("your/dataset-name")
my_processor.dataset_to_jsonl("data/train.jsonl", dataset=raw_dataset['train'])
my_processor.dataset_to_jsonl("data/test.jsonl", dataset=raw_dataset['test'])

# --- 3. 设置路径并初始化 PromptWizard ---
train_file = "data/train.jsonl"
test_file = "data/test.jsonl"
promptopt_config_path = "configs/promptopt_config.yaml"
setup_config_path = "configs/setup_config.yaml"

# 初始化 GluePromptOpt
gp = GluePromptOpt(promptopt_config_path,
                   setup_config_path,
                   train_file,
                   my_processor)

# --- 4. 运行优化 ---
# 这是核心调用，通过参数控制不同的优化场景
# 场景1: 标准流程，使用训练数据优化指令和示例
best_prompt, expert_profile = gp.get_best_prompt(use_examples=True)

# 场景2: Zero-shot，只优化指令，不生成示例
# best_prompt, expert_profile = gp.get_best_prompt(run_without_train_examples=True)

# 场景3: 无训练数据，让LLM自己生成合成数据
# best_prompt, expert_profile = gp.get_best_prompt(generate_synthetic_examples=True)

print("--- OPTIMIZED PROMPT ---")
print(best_prompt)
print("\n--- EXPERT PROFILE ---")
print(expert_profile)

# --- 5. 评估结果 ---
# 将优化好的 prompt 和 profile 应用于评估器
gp.EXPERT_PROFILE = expert_profile
gp.BEST_PROMPT = best_prompt

# 在测试集上运行评估
accuracy = gp.evaluate(test_file)
print(f"\nFinal Accuracy on Test Set: {accuracy}")
```

### 步骤 2: 运行脚本

在您的终端中，导航到 `my_custom_task/` 目录并运行您的脚本。

```bash
python run.py
# 或者在 VS Code 中运行 .ipynb 文件
```

`PromptWizard` 将开始执行多轮优化，并打印详细的日志。完成后，您将获得优化后的提示和其在测试集上的性能表现。

## 6. 总结与最佳实践

*   **从 `demos` 开始**: 在开始您自己的项目前，请务必完整地运行并理解 `demos/` 目录下的一个示例（如 `gsm8k`）。这是最快的学习路径。
*   **迭代式开发**: 先实现一个基础版本的 `DatasetSpecificProcessing`，让整个流程跑通。然后再逐步完善答案提取和评估逻辑。
*   **关注 `answer_format`**: `answer_format` 指令和 `extract_final_answer` 函数必须紧密配合。这是决定评估准确性的关键。
*   **成本与效果的权衡**: `mutate_refine_iterations` 是影响性能和 API 调用成本最直接的参数，请根据您的预算和需求进行调整。

希望这份手册能帮助您成功地使用 `PromptWizard` 提升您的 LLM 应用性能。