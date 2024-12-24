import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import numpy as np
from sympy import symbols, diff, integrate, sympify, SympifyError, Matrix, Eq
import networkx as nx
from typing import Dict, List, Any, Tuple, Union, Optional
import time
import random
import json
import os
import re

class ConceptNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ConceptNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class MathematicalConcept:
    def __init__(self, name: str, description: str, prerequisites: List[str]):
        self.name = name
        self.description = description
        self.prerequisites = prerequisites
        self.examples = []
        self.proofs = []
        self.related_concepts = {}

class MathTrainerGPU:
    def __init__(self):
        """
        A class for training and reasoning about mathematical concepts.
        Provides basic arithmetic, calculus, matrix, statistics, and trig operations,
        as well as concept learning and theorem proving functionalities.
        """
        # State and memory
        self.variables = {}
        self.attention_weights = {}
        self.optimization_log = []
        self.functions = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "undefined",
            "power": lambda x, y: x ** y,
        }

        # GPT-2 pipeline for text generation
        self.text_generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)

        # Extend functions
        self.functions |= {
            "matrix_multiply": lambda x, y: np.matmul(x, y),
            "inverse": lambda x: np.linalg.inv(x),
            "mean": lambda x: np.mean(x),
            "std": lambda x: np.std(x),
            "sin": lambda x: np.sin(x),
            "cos": lambda x: np.cos(x),
            "tan": lambda x: np.tan(x),
        }
        self.history = []
        self.max_history = 1000

        # Complex functionalities
        self.concept_graph = nx.DiGraph()
        self.concepts = {}
        self.neural_net = ConceptNetwork(100, 256, 100).cuda() if torch.cuda.is_available() else ConceptNetwork(100, 256, 100)
        self.optimizer = optim.Adam(self.neural_net.parameters())
        self.concept_embeddings = {}

        self.functions |= {
            "prove": self.prove_theorem,
            "learn_concept": self.learn_new_concept,
            "relate_concepts": self.relate_concepts,
            "verify_proof": self.verify_proof,
        }

        # Data preprocessors
        self.data_preprocessors = {
            "matrix": self.preprocess_matrix,
            "vector": self.preprocess_vector,
            "equation": self.preprocess_equation
        }

        self.functions |= {
            "mix": self.mix_data,
            "transform": self.transform_data,
            "validate": self.validate_data,
        }

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load a BERT classifier (not a text generator) to classify instructions
        try:
            self.instruction_parser = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
            print("BERT model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load BERT model: {e}")
            self.instruction_parser = None

        self.knowledge_base = {}
        self.model_state = {}
        self.current_context = None
        self.instruction_cache = {}

        self.brain_state_dir = "brain_states"
        os.makedirs(self.brain_state_dir, exist_ok=True)

        self.repair_attempts = {}
        self.max_repair_attempts = 3
        self.repair_history = []
        self.repair_strategies = {
            "syntax_error": self.repair_syntax,
            "type_error": self.repair_type_mismatch,
            "value_error": self.repair_value_error,
            "attribute_error": self.repair_attribute_error,
            "general_error": self.repair_using_bert
        }

        self.goals = []
        self.active_goal = None

        self.creative_strategies = {
            "analogical_reasoning": self.analogical_reasoning,
            "divergent_thinking": self.divergent_thinking,
            "metacognition": self.metacognition,
        }

        # Cache for classification results
        self.classification_cache = {}

    def set_goal(self, goal_description: str):
        goal = {
            "description": goal_description,
            "subgoals": [],
            "progress": 0.0,
            "completed": False
        }
        self.goals.append(goal)
        self.active_goal = goal
        return f"New goal set: {goal_description}"

    def plan_achievement(self):
        if not self.active_goal:
            return "No active goal to plan for."
        # Use GPT-2 for text generation
        # Since we no longer attempt to use BERT for generation, we can do something simple
        plan_prompt = f"Plan steps to achieve the goal: {self.active_goal['description']}"
        responses = self.text_generator(plan_prompt, max_length=150, num_return_sequences=1)
        generated = responses[0]['generated_text']
        steps = generated.strip().split('\n')
        self.active_goal['subgoals'] = [step.strip() for step in steps if step.strip()]
        return f"Planned steps: {self.active_goal['subgoals']}"

    def execute_plan(self):
        if not self.active_goal or not self.active_goal['subgoals']:
            return "No plan to execute."
        results = []
        for step in self.active_goal['subgoals']:
            result = self.teach(step)
            results.append(result)
            self.active_goal['progress'] += 1 / len(self.active_goal['subgoals'])
        self.active_goal['completed'] = True
        return f"Goal '{self.active_goal['description']}' achieved with results: {results}"

    def analogical_reasoning(self, problem_description: str):
        # Find similar problems in history
        similar_problems = [h for h in self.history if self.similar_strings(h['instruction'], problem_description)]
        if similar_problems:
            analogous_solution = similar_problems[-1]['result']
            return f"Applying analogous solution: {analogous_solution}"
        else:
            return "No analogous solution found."

    def divergent_thinking(self, problem_description: str):
        prompt = f"List different ways to solve the problem: {problem_description}"
        responses = self.text_generator(prompt, max_length=100, num_return_sequences=3)
        solutions = [r['generated_text'] for r in responses]
        return solutions

    def metacognition(self):
        reflections = []
        for entry in self.history[-5:]:
            reflections.append(f"Instruction: {entry['instruction']}, Result: {entry['result']}")
        return "Recent reflections:\n" + '\n'.join(reflections)

    def teach(self, instruction):
        start_time = time.time()
        try:
            # Determine complexity based on classification
            is_complex = self.is_complex_instruction(instruction)
            if is_complex:
                # If instruction is complex, plan steps and execute them
                plan = self.plan_instruction(instruction)
                results = []
                for step in plan:
                    result = self.parse_instruction(step)
                    results.append(result)
                combined_result = self.combine_results(results)
                elapsed_time = time.time() - start_time
                self.update_attention(instruction, elapsed_time)
                self.add_to_history(instruction, combined_result)
                return {
                    "instruction": instruction,
                    "result": combined_result,
                    "time": elapsed_time
                }
            else:
                # Simple instruction handling
                result = self.parse_instruction(instruction)
                # If result is problematic, try creative strategies
                if not result or "Error" in str(result):
                    creative_result = self.apply_creative_strategies(instruction)
                    if creative_result:
                        result = creative_result
                elapsed_time = time.time() - start_time
                self.update_attention(instruction, elapsed_time)
                self.add_to_history(instruction, result)
                return {
                    "instruction": instruction,
                    "result": result,
                    "time": elapsed_time
                }
        except Exception as e:
            # Attempt self-repair on failure
            repair_result = self.attempt_self_repair(instruction, e)
            elapsed_time = time.time() - start_time
            return {
                "instruction": instruction,
                "result": repair_result,
                "time": elapsed_time,
                "repaired": True
            }

    def apply_creative_strategies(self, instruction):
        analogical_result = self.creative_strategies["analogical_reasoning"](instruction)
        if analogical_result != "No analogous solution found.":
            return analogical_result
        divergent_solutions = self.creative_strategies["divergent_thinking"](instruction)
        if divergent_solutions:
            return f"Divergent solutions: {divergent_solutions}"
        reflection = self.creative_strategies["metacognition"]()
        return reflection

    def classify_instruction(self, instruction: str) -> int:
        """
        Classify the instruction using the BERT classifier.
        Returns an integer label representing the class.
        This is a placeholder - you can define your own label schema.
        """
        if self.instruction_parser is None:
            return 0  # Default to a neutral class if model not available

        if instruction in self.classification_cache:
            return self.classification_cache[instruction]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = self.tokenizer(instruction, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = self.instruction_parser(**inputs)
        label = torch.argmax(outputs.logits, dim=1).item()
        self.classification_cache[instruction] = label
        return label

    def is_complex_instruction(self, instruction):
        """
        Determine if an instruction is complex.
        Heuristics:
        - Length of instruction
        - Classification label: certain labels indicate complexity
        - Key phrases
        """
        label = self.classify_instruction(instruction)
        # Simple heuristic: If label is above a certain threshold or instruction is long
        if len(instruction.split()) > 5 or "and then" in instruction:
            return True
        # You can refine the logic based on label values (e.g. certain classes = complex)
        return False

    def plan_instruction(self, instruction):
        # Use GPT-2 model to generate steps
        # Simple approach: ask GPT-2 to break down the instruction
        prompt = f"Break down this instruction into a step-by-step plan:\n{instruction}"
        response = self.text_generator(prompt, max_length=512, num_return_sequences=1)
        steps_text = response[0]['generated_text']
        steps = steps_text.split('\n')
        return [step.strip() for step in steps if step.strip()]

    def combine_results(self, results):
        return "\n".join(str(result) for result in results)

    def ingest_research_paper(self, paper_text):
        headings = re.findall(r'\n([A-Z][A-Za-z ]+)\n', paper_text)
        for heading in headings:
            concept = heading.strip()
            if concept not in self.knowledge_base:
                self.knowledge_base[concept] = paper_text.count(concept)
        return f"Extracted concepts: {', '.join(headings)}"

    def parse_instruction(self, instruction):
        # Check cache
        if instruction in self.instruction_cache:
            return self.instruction_cache[instruction]

        instruction = instruction.strip()

        try:
            # Route to appropriate handlers based on instruction content
            if instruction.startswith("let"):
                result = self.assign_variable(instruction)
            elif instruction.startswith("matrix"):
                result = self.handle_matrix_operation(instruction)
            elif any(op in instruction for op in ["add", "subtract", "multiply", "divide", "power"]):
                result = self.compute_with_torch(instruction)
            elif instruction.startswith("find the derivative"):
                result = self.handle_derivative(instruction)
            elif instruction.startswith("find the integral"):
                result = self.handle_integral(instruction)
            elif instruction.startswith("determine the eigenvalues"):
                result = self.handle_eigenvalues(instruction)
            elif instruction.startswith("determine the eigenvectors"):
                result = self.handle_eigenvectors(instruction)
            elif instruction.startswith("reset all"):
                result = self.reset_all()
            elif instruction.startswith("stats"):
                result = self.handle_statistics(instruction)
            elif instruction.startswith("trig"):
                result = self.handle_trigonometry(instruction)
            elif instruction.startswith("save state"):
                result = self.save_state(instruction.split()[-1])
            elif instruction.startswith("load state"):
                result = self.load_state(instruction.split()[-1])
            elif instruction.startswith("learn"):
                parts = instruction.split("|")
                result = self.learn_new_concept(parts[1], parts[2], parts[3].split(","))
            elif instruction.startswith("relate"):
                _, concept1, concept2, relationship = instruction.split("|")
                result = self.relate_concepts(concept1, concept2, relationship)
            elif instruction.startswith("prove"):
                _, theorem = instruction.split("|")
                result = self.prove_theorem(theorem)
            elif instruction.startswith("ingest paper"):
                paper_path = instruction[len("ingest paper "):].strip()
                try:
                    with open(paper_path, 'r', encoding='utf-8') as f:
                        paper_text = f.read()
                    extraction_result = self.ingest_research_paper(paper_text)
                    result = f"Ingested research paper from {paper_path}. {extraction_result}"
                except Exception as e:
                    result = f"Error ingesting paper: {e}"
            elif instruction.startswith("save brain"):
                name = instruction[len("save brain "):].strip()
                result = self.save_brain_state(name)
            elif instruction.startswith("load brain"):
                name = instruction[len("load brain "):].strip()
                result = self.load_brain_state(name)
            else:
                result = "Unknown instruction"

            self.instruction_cache[instruction] = result
            return result
        except Exception as e:
            error_msg = f"Error processing instruction: {str(e)}"
            self.instruction_cache[instruction] = error_msg
            return error_msg

    def assign_variable(self, instruction):
        _, var, _, value = instruction.split()
        self.variables[var] = float(value)
        return f"Variable {var} set to {value}"

    def compute_with_torch(self, instruction):
        op, rest = instruction.split(" ", 1)
        operands = [o.strip() for o in rest.split("and")]
        if len(operands) != 2:
            return "Invalid operands"
        a_val, b_val = float(operands[0]), float(operands[1])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.tensor(a_val, device=device)
        b = torch.tensor(b_val, device=device)
        if op == "add":
            return (a + b).item()
        elif op == "subtract":
            return (a - b).item()
        elif op == "multiply":
            return (a * b).item()
        elif op == "divide":
            return (a / b).item() if b.item() != 0 else "undefined"
        elif op == "power":
            try:
                return pow(a.item(), b.item())
            except OverflowError:
                return "Result is too large to compute"

    def handle_derivative(self, instruction):
        try:
            _, _, expression = instruction.partition("of ")
            x = symbols("x")
            expression = expression.strip().replace('\xa0', ' ')
            expr = sympify(expression)
            return diff(expr, x)
        except SympifyError as e:
            return f"Invalid expression for derivative: {str(e)}"
        except Exception as e:
            return f"Error in derivative calculation: {str(e)}"

    def handle_integral(self, instruction):
        _, _, expression = instruction.partition("of ")
        x = symbols("x")
        try:
            expr = sympify(expression)
            return integrate(expr, x)
        except SympifyError:
            return "Invalid expression for integral"

    def handle_eigenvalues(self, instruction):
        try:
            _, _, matrix = instruction.partition("of ")
            matrix = matrix.strip().replace('\xa0', ' ')
            mat = Matrix(eval(matrix))
            return mat.eigenvals()
        except SyntaxError:
            return "Invalid matrix format"
        except Exception as e:
            return f"Error calculating eigenvalues: {str(e)}"

    def handle_eigenvectors(self, instruction):
        _, _, matrix = instruction.partition("of ")
        try:
            mat = Matrix(eval(matrix))
            return mat.eigenvects()
        except Exception as e:
            return f"Error: {e}"

    def handle_matrix_operation(self, instruction):
        parts = instruction.split(maxsplit=2)
        if len(parts) < 3:
            return "Invalid matrix instruction format"

        operation = parts[1]
        try:
            if operation == "inverse":
                matrix_str = parts[2]
                matrix_str = matrix_str.replace(' ', '')
                matrix = np.array(eval(matrix_str), dtype=float)
                return np.linalg.inv(matrix)
            elif operation == "multiply":
                return self._handle_matrix_multiplication(parts)
        except Exception as e:
            return f"Matrix operation error: {str(e)}"

    def _handle_matrix_multiplication(self, parts):
        matrices_str = parts[2].replace(' ', '')
        bracket_count = 0
        split_index = 0
        for i, char in enumerate(matrices_str):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    split_index = i + 1
                    break

        if split_index == 0 or split_index >= len(matrices_str):
            return "Invalid matrix format"

        matrix1_str = matrices_str[:split_index]
        matrix2_str = matrices_str[split_index:]
        matrix1 = np.array(eval(matrix1_str), dtype=float)
        matrix2 = np.array(eval(matrix2_str), dtype=float)
        if matrix1.shape[1] != matrix2.shape[0]:
            return f"Matrix dimensions incompatible: {matrix1.shape} and {matrix2.shape}"
        return np.matmul(matrix1, matrix2)

    def handle_statistics(self, instruction):
        parts = instruction.split()
        if len(parts) < 3:
            return "Invalid statistics instruction"
        operation = parts[1]
        data = [float(x) for x in parts[2:]]
        if operation == "mean":
            return np.mean(data)
        elif operation == "std":
            return np.std(data)
        return "Unknown statistical operation"

    def handle_trigonometry(self, instruction):
        parts = instruction.split()
        operation = parts[1]
        angle = float(parts[2])
        if operation == "sin":
            return np.sin(angle)
        elif operation == "cos":
            return np.cos(angle)
        elif operation == "tan":
            return np.tan(angle)
        return "Unknown trigonometric operation"

    def reset_all(self):
        self.variables.clear()
        return "All variables reset"

    def update_attention(self, instruction, time_taken):
        if instruction not in self.attention_weights:
            self.attention_weights[instruction] = {"time": time_taken, "count": 1}
        else:
            self.attention_weights[instruction]["time"] += time_taken
            self.attention_weights[instruction]["count"] += 1

    def add_to_history(self, instruction, result):
        self.history.append({
            "timestamp": time.time(),
            "instruction": instruction,
            "result": result
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def save_state(self, filename):
        state = {
            "variables": self.variables,
            "attention_weights": self.attention_weights,
            "history": self.history
        }
        with open(f"{filename}.json", "w") as f:
            json.dump(state, f)
        return f"State saved to {filename}.json"

    def load_state(self, filename):
        try:
            with open(f"{filename}.json", "r") as f:
                state = json.load(f)
            self.variables = state["variables"]
            self.attention_weights = state["attention_weights"]
            self.history = state["history"]
            return f"State loaded from {filename}.json"
        except FileNotFoundError:
            return "State file not found"

    def save_brain_state(self, name: str) -> str:
        try:
            state_path = os.path.join(self.brain_state_dir, f"{name}")
            os.makedirs(state_path, exist_ok=True)

            torch.save({
                'model_state_dict': self.neural_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(state_path, 'neural_net.pt'))

            nx.write_gpickle(self.concept_graph, os.path.join(state_path, 'concept_graph.gpickle'))

            state_data = {
                'knowledge_base': self.knowledge_base,
                'concepts': {nm: self.__concept_to_dict(concept)
                             for nm, concept in self.concepts.items()},
                'instruction_cache': self.instruction_cache,
                'model_state': self.model_state,
                'attention_weights': self.attention_weights,
                'variables': self.variables,
                'concept_embeddings': {k: v.tolist() if isinstance(v, torch.Tensor) else v
                                       for k, v in self.concept_embeddings.items()}
            }

            with open(os.path.join(state_path, 'state.json'), 'w') as f:
                json.dump(state_data, f)

            return f"Brain state saved to {state_path}"
        except Exception as e:
            return f"Error saving brain state: {str(e)}"

    def load_brain_state(self, name: str) -> str:
        try:
            state_path = os.path.join(self.brain_state_dir, f"{name}")

            checkpoint = torch.load(os.path.join(state_path, 'neural_net.pt'))
            self.neural_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.concept_graph = nx.read_gpickle(os.path.join(state_path, 'concept_graph.gpickle'))

            with open(os.path.join(state_path, 'state.json'), 'r') as f:
                state_data = json.load(f)

            self.knowledge_base = state_data['knowledge_base']
            self.concepts = {nm: self.__dict_to_concept(cdict)
                             for nm, cdict in state_data['concepts'].items()}
            self.instruction_cache = state_data['instruction_cache']
            self.model_state = state_data['model_state']
            self.attention_weights = state_data['attention_weights']
            self.variables = state_data['variables']
            self.concept_embeddings = {k: torch.tensor(v) if isinstance(v, list)
                                       else v for k, v in state_data['concept_embeddings'].items()}

            return f"Brain state loaded from {state_path}"
        except Exception as e:
            return f"Error loading brain state: {str(e)}"

    def __concept_to_dict(self, concept: MathematicalConcept) -> dict:
        return {
            'name': concept.name,
            'description': concept.description,
            'prerequisites': concept.prerequisites,
            'examples': concept.examples,
            'proofs': concept.proofs,
            'related_concepts': concept.related_concepts
        }

    def __dict_to_concept(self, concept_dict: dict) -> MathematicalConcept:
        concept = MathematicalConcept(
            concept_dict['name'],
            concept_dict['description'],
            concept_dict['prerequisites']
        )
        concept.examples = concept_dict['examples']
        concept.proofs = concept_dict['proofs']
        concept.related_concepts = concept_dict['related_concepts']
        return concept

    def prove_theorem(self, theorem: str) -> str:
        is_proven, proof_details = self.prove_theorem_logic(theorem)
        if is_proven:
            return f"Theorem proved:\n{proof_details}"
        else:
            return f"Theorem could not be proved:\n{proof_details}"

    def prove_theorem_logic(self, theorem: str) -> Tuple[bool, str]:
        try:
            if "==" in theorem:
                left, right = theorem.split("==")
                left = sympify(left.strip())
                right = sympify(right.strip())
                theorem_expr = left - right
            else:
                theorem_expr = sympify(theorem)

            simplified = theorem_expr.simplify()
            expanded = theorem_expr.expand()
            factored = theorem_expr.factor()

            if 0 in [simplified, expanded, factored]:
                proof_steps = [
                    f"Original: {theorem}",
                    f"Simplified: {simplified}",
                    f"Expanded: {expanded}",
                    f"Factored: {factored}",
                    "Theorem is true (equals zero in some form)",
                ]
                return True, "\n".join(proof_steps)

            return False, "Could not prove theorem"
        except Exception as e:
            return False, f"Error in proof attempt: {str(e)}"

    def verify_proof(self, theorem: str, proof_steps: List[str]) -> bool:
        try:
            current_state = sympify(theorem)
            for step in proof_steps:
                step_result = sympify(step)
                if not current_state.equals(step_result):
                    return False
                current_state = step_result
            return True
        except Exception:
            return False

    def mix_data(self, data1: Union[np.ndarray, list], data2: Union[np.ndarray, list],
                 method: str = "concatenate") -> Union[np.ndarray, list]:
        try:
            d1 = np.array(data1)
            d2 = np.array(data2)

            if method == "concatenate":
                return np.concatenate([d1, d2])
            elif method == "stack":
                return np.vstack([d1, d2])
            elif method == "interleave":
                return np.array([x for pair in zip(d1, d2) for x in pair])
            else:
                raise ValueError(f"Unknown mixing method: {method}")
        except Exception as e:
            return f"Error mixing data: {str(e)}"

    def learn_new_concept(self, name: str, description: str, prerequisites: List[str]) -> str:
        concept = MathematicalConcept(name, description, prerequisites)
        self.concepts[name] = concept
        self.concept_graph.add_node(name)
        for prereq in prerequisites:
            if prereq in self.concepts:
                self.concept_graph.add_edge(prereq, name)
        return f"Learned new concept: {name}"

    def transform_data(self, data: Union[np.ndarray, list], transformation: str) -> Union[np.ndarray, list]:
        try:
            arr = np.array(data)
            if transformation == "normalize":
                return (arr - np.mean(arr)) / np.std(arr)
            elif transformation == "scale":
                return arr * 2
            else:
                raise ValueError(f"Unknown transformation method: {transformation}")
        except Exception as e:
            return f"Error transforming data: {str(e)}"

    def validate_data(self, data: Union[np.ndarray, list], data_type: str) -> bool:
        try:
            if data_type == "matrix":
                mat = np.array(data)
                return mat.ndim == 2 and mat.shape[0] == mat.shape[1]
            elif data_type == "vector":
                vec = np.array(data)
                return vec.ndim == 1
            elif data_type == "equation":
                expr = sympify(data)
                return isinstance(expr, (Eq, type(sympify('x'))))
            else:
                raise ValueError(f"Unknown data type for validation: {data_type}")
        except Exception:
            return False

    def relate_concepts(self, concept1: str, concept2: str, relationship: str) -> bool:
        try:
            if concept1 not in self.concepts or concept2 not in self.concepts:
                return False

            self.concept_graph.add_edge(concept1, concept2, relationship=relationship)
            self.concepts[concept1].related_concepts[concept2] = relationship
            self.concepts[concept2].related_concepts[concept1] = relationship

            return True
        except Exception as e:
            print(f"Error relating concepts: {str(e)}")
            return False

    def attempt_self_repair(self, instruction: str, error: Exception) -> str:
        error_type = type(error).__name__
        error_msg = str(error)

        if instruction in self.repair_attempts:
            if self.repair_attempts[instruction] >= self.max_repair_attempts:
                return f"Max repair attempts reached. Last error: {error_msg}"
            self.repair_attempts[instruction] += 1
        else:
            self.repair_attempts[instruction] = 1

        self.repair_history.append({
            "instruction": instruction,
            "error_type": error_type,
            "error_message": error_msg,
            "timestamp": time.time()
        })

        if error_type in self.repair_strategies:
            repair_func = self.repair_strategies[error_type]
        else:
            repair_func = self.repair_strategies["general_error"]

        try:
            repaired_instruction = repair_func(instruction, error)
            if repaired_instruction:
                result = self.parse_instruction(repaired_instruction)
                return f"Self-repaired and executed: {result}"
            else:
                return f"Could not repair instruction. Error: {error_msg}"
        except Exception as repair_error:
            return f"Repair failed: {str(repair_error)}"

    def repair_syntax(self, instruction: str, error: Exception) -> Optional[str]:
        # Use GPT-2 to suggest syntax fixes
        prompt = f"Fix syntax: {instruction}\nError: {str(error)}"
        suggestion = self.text_generator(prompt, max_length=100)[0]["generated_text"]
        return suggestion.strip()

    def repair_type_mismatch(self, instruction: str, error: Exception) -> Optional[str]:
        error_msg = str(error)
        if "float" in error_msg and "str" in error_msg:
            # Try removing quotes around numbers
            modified = re.sub(r'["\'](\d+\.?\d*)["\']', r'\1', instruction)
            return modified
        return None

    def repair_value_error(self, instruction: str, error: Exception) -> Optional[str]:
        error_msg = str(error)
        if "division by zero" in error_msg:
            return instruction.replace("divide", "divide_safe")
        elif "invalid literal" in error_msg:
            cleaned = re.sub(r'[^\w\s+\-*/().]', '', instruction)
            return cleaned
        return None

    def repair_attribute_error(self, instruction: str, error: Exception) -> Optional[str]:
        error_msg = str(error)
        # Attempt to guess correct attribute
        missing_attr = error_msg.split("'")[1] if "'" in error_msg else None
        if not missing_attr:
            return None
        for concept_name, concept in self.concepts.items():
            if concept_name.lower() in instruction.lower():
                valid_attrs = dir(concept)
                similar_attrs = [attr for attr in valid_attrs if self.similar_strings(attr, missing_attr)]
                if similar_attrs:
                    return instruction.replace(missing_attr, similar_attrs[0])
        return None

    def repair_using_bert(self, instruction: str, error: Exception) -> Optional[str]:
        # Use GPT-2 for suggestions (not BERT)
        repair_prompt = f"Instruction: {instruction}\nError: {str(error)}\nSuggest repair:"
        suggestion = self.text_generator(repair_prompt, max_length=150)[0]["generated_text"]
        # No validation logic here, just return suggestion
        return suggestion.strip()

    def similar_strings(self, s1: str, s2: str) -> bool:
        if not s1 or not s2:
            return False
        if abs(len(s1) - len(s2)) > 2:
            return False
        return sum(a != b for a, b in zip(s1, s2)) <= 2

    def generate_dynamic_instructions(self, count=50):
        instructions = []
        operations = ["add", "subtract", "multiply", "divide", "power"]
        for _ in range(count):
            op = random.choice(operations)
            a, b = random.randint(1, 100), random.randint(1, 100)
            instructions.append(f"{op} {a} and {b}")

        calculus_ops = [
            "find the derivative of x**2 + 3*x",
            "find the integral of sin(x)",
            "determine the eigenvalues of [[2, -1], [-1, 2]]",
            "determine the eigenvectors of [[2, -1], [-1, 2]]"
        ]

        matrix_ops = [
            "matrix multiply [[1,2],[3,4]] [[5,6],[7,8]]",
            "matrix inverse [[1,0],[0,1]]"
        ]

        stats_ops = [
            "stats mean 1 2 3 4 5",
            "stats std 1 2 3 4 5"
        ]

        trig_ops = [
            "trig sin 0.5",
            "trig cos 0.5",
            "trig tan 0.5"
        ]

        concept_ops = [
            "learn|Linear Algebra|Study of linear equations and functions|calculus,matrices",
            "learn|Group Theory|Study of algebraic structures|linear algebra",
            "relate|Linear Algebra|Group Theory|fundamental basis",
            "prove|x**2 - 1 == (x+1)*(x-1)"
        ]

        return instructions + calculus_ops + matrix_ops + stats_ops + trig_ops + concept_ops

if __name__ == "__main__":
    trainer = MathTrainerGPU()

    generated_instructions = trainer.generate_dynamic_instructions()

    for instruction in generated_instructions:
        output = trainer.teach(instruction)
        print(f"Instruction: {instruction}\nResult: {output['result']} (Processed in {output['time']:.4f}s)\n")
