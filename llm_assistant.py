"""
LLM assistant for experiment design using OpenAI function calling.
"""

import os
import json
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

import stats_functions as sf

load_dotenv()

# Tool definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_sample_size",
            "description": "Calculate required sample size per group for an A/B test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "baseline_rate": {"type": "number", "description": "Baseline conversion rate as decimal (0.10 = 10%)"},
                    "mde": {"type": "number", "description": "Minimum detectable effect, absolute (0.02 = 2pp)"},
                    "alpha": {"type": "number", "description": "Significance level", "default": 0.05},
                    "power": {"type": "number", "description": "Statistical power", "default": 0.80}
                },
                "required": ["baseline_rate", "mde"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_power",
            "description": "Calculate statistical power given sample size and effect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "baseline_rate": {"type": "number", "description": "Baseline rate as decimal"},
                    "effect_size": {"type": "number", "description": "Effect size (absolute)"},
                    "sample_size": {"type": "integer", "description": "Sample size per group"},
                    "alpha": {"type": "number", "description": "Significance level", "default": 0.05}
                },
                "required": ["baseline_rate", "effect_size", "sample_size"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_mde",
            "description": "Calculate minimum detectable effect for a given sample size.",
            "parameters": {
                "type": "object",
                "properties": {
                    "baseline_rate": {"type": "number", "description": "Baseline rate as decimal"},
                    "sample_size": {"type": "integer", "description": "Sample size per group"},
                    "alpha": {"type": "number", "description": "Significance level", "default": 0.05},
                    "power": {"type": "number", "description": "Statistical power", "default": 0.80}
                },
                "required": ["baseline_rate", "sample_size"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_cuped_sample_size",
            "description": "Calculate sample size with CUPED variance reduction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "baseline_rate": {"type": "number", "description": "Baseline rate as decimal"},
                    "mde": {"type": "number", "description": "MDE (absolute)"},
                    "variance_reduction": {"type": "number", "description": "Variance reduction (0-1, e.g. 0.3 = 30%)"},
                    "alpha": {"type": "number", "description": "Significance level", "default": 0.05},
                    "power": {"type": "number", "description": "Statistical power", "default": 0.80}
                },
                "required": ["baseline_rate", "mde", "variance_reduction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_experiment_duration",
            "description": "Estimate experiment duration based on sample size and traffic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_size_per_group": {"type": "integer", "description": "Required sample size per group"},
                    "daily_visitors": {"type": "integer", "description": "Daily visitors available"},
                    "traffic_allocation": {"type": "number", "description": "Traffic fraction (0-1)", "default": 1.0}
                },
                "required": ["sample_size_per_group", "daily_visitors"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an experimentation assistant. Help users design A/B tests.

When users describe their scenario, use the tools to calculate sample sizes, power, MDE, or duration.

Guidelines:
- Convert percentages to decimals (10% → 0.10)
- Handle relative effects ("20% lift on 10% baseline" = 0.02 absolute)
- Default to 80% power, 5% significance if not specified
- Ask for missing info (baseline, effect, traffic)
- Warn about peeking and multiple comparisons

Be concise. Format numbers clearly (e.g., "3,847 users per group")."""


FUNCTION_MAP = {
    "calculate_sample_size": sf.calculate_sample_size,
    "calculate_power": sf.calculate_power,
    "calculate_mde": sf.calculate_mde,
    "calculate_cuped_sample_size": sf.calculate_cuped_sample_size,
    "calculate_experiment_duration": sf.calculate_experiment_duration,
}


class ExperimentAssistant:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=self.api_key)
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name not in FUNCTION_MAP:
            return json.dumps({"error": f"Unknown function: {tool_name}"})

        try:
            func = FUNCTION_MAP[tool_name]
            result = func(**arguments)

            if tool_name == "calculate_sample_size":
                return json.dumps({
                    "sample_size_per_group": result,
                    "total_sample_size": result * 2,
                    "parameters": arguments
                })
            elif tool_name == "calculate_power":
                return json.dumps({
                    "statistical_power": round(result, 4),
                    "power_percentage": f"{result * 100:.1f}%",
                    "parameters": arguments
                })
            elif tool_name == "calculate_mde":
                return json.dumps({
                    "mde_absolute": round(result, 4),
                    "mde_percentage_points": f"{result * 100:.2f} pp",
                    "relative_to_baseline": f"{(result / arguments['baseline_rate']) * 100:.1f}%" if arguments.get('baseline_rate', 0) > 0 else "N/A",
                    "parameters": arguments
                })
            elif tool_name == "calculate_cuped_sample_size":
                original = sf.calculate_sample_size(
                    arguments['baseline_rate'],
                    arguments['mde'],
                    arguments.get('alpha', 0.05),
                    arguments.get('power', 0.80)
                )
                return json.dumps({
                    "cuped_sample_size_per_group": result,
                    "original_sample_size_per_group": original,
                    "sample_size_reduction": f"{((original - result) / original) * 100:.1f}%",
                    "parameters": arguments
                })
            else:
                return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": str(e)})

    def chat(self, user_message: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.conversation_history,
            tools=TOOLS,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message

        # Handle tool calls
        while assistant_message.tool_calls:
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            for tool_call in assistant_message.tool_calls:
                result = self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversation_history,
                tools=TOOLS,
                tool_choice="auto"
            )
            assistant_message = response.choices[0].message

        final_response = assistant_message.content or ""
        self.conversation_history.append({"role": "assistant", "content": final_response})
        return final_response

    def reset_conversation(self):
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]


if __name__ == "__main__":
    assistant = ExperimentAssistant()

    # Quick test
    queries = [
        "Our conversion is 3.2%, we want to detect a 10% relative lift. Sample size?",
        "We have 5000 visitors/day. How long to run it?",
    ]
    for q in queries:
        print(f"\n> {q}")
        print(assistant.chat(q))
