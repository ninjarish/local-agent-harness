import unittest

from local_agent.agent import Agent, _required_tool, parse_action
from local_agent.tools import Tool, ToolRegistry


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, prompt):
        return next(self.responses)


class AgentTests(unittest.TestCase):
    def test_docx_path_separator_is_not_treated_as_arithmetic(self):
        prompt = (
            'can you summarize file at the location - "C:\\Users\\me\\Downloads\\'
            'Power_Platform_Environment_Migration_Guide.docx"'
        )
        tool = _required_tool(
            prompt,
            ["calculate", "read_text", "read_document"],
        )
        self.assertEqual(tool, "read_document")

    def test_parse_action_accepts_fenced_json(self):
        action = parse_action(
            '```json\n{"type":"tool_call","name":"ping","arguments":{}}\n```'
        )
        self.assertEqual(action["name"], "ping")

    def test_plain_text_is_rejected(self):
        action = parse_action("A direct answer")
        self.assertEqual(action, {"type": "invalid", "content": "A direct answer"})

    def test_agent_executes_tool_and_returns_followup(self):
        registry = ToolRegistry()
        registry.register(
            Tool("double", "Double a number", {}, lambda number: number * 2)
        )
        model = SequenceModel(
            [
                '{"type":"tool_call","name":"double","arguments":{"number":21}}',
                '{"type":"final","content":"The answer is 42."}',
            ]
        )
        agent = Agent(model=model, tools=registry, confirm=lambda tool, args: True)
        self.assertEqual(agent.run("Double 21"), "The answer is 42.")

    def test_agent_rejects_final_until_explicit_tool_runs(self):
        registry = ToolRegistry()
        calls = []
        registry.register(
            Tool(
                "calculate",
                "Calculate",
                {},
                lambda expression: calls.append(expression) or 42,
            )
        )
        model = SequenceModel(
            [
                '{"type":"final","content":"41"}',
                '{"type":"tool_call","name":"calculate","arguments":{"expression":"21 * 2"}}',
                '{"type":"final","content":"42"}',
            ]
        )
        agent = Agent(model=model, tools=registry, confirm=lambda tool, args: True)
        self.assertEqual(
            agent.run("Use calculate for twenty-one times two"),
            "42",
        )
        self.assertEqual(calls, ["21 * 2"])

    def test_agent_returns_authoritative_calculator_result_immediately(self):
        registry = ToolRegistry()
        registry.register(
            Tool("calculate", "Calculate", {}, lambda expression: 9876)
        )
        model = SequenceModel(
            [
                '{"type":"tool_call","name":"calculate","arguments":{"expression":"21 * 2"}}',
            ]
        )
        agent = Agent(model=model, tools=registry, confirm=lambda tool, args: True)
        self.assertEqual(
            agent.run("Use calculate for the requested product"),
            "9876",
        )

    def test_agent_routes_numeric_arithmetic_without_model_planning(self):
        registry = ToolRegistry()
        expressions = []
        registry.register(
            Tool(
                "calculate",
                "Calculate",
                {},
                lambda expression: expressions.append(expression) or 12360,
            )
        )
        model = SequenceModel([])
        agent = Agent(model=model, tools=registry, confirm=lambda tool, args: True)
        self.assertEqual(
            agent.run("Compute (287 * 43) + 19."),
            "12360",
        )
        self.assertEqual(expressions, ["(287 * 43) + 19"])


if __name__ == "__main__":
    unittest.main()
