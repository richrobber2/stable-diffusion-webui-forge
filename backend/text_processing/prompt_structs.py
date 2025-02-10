from dataclasses import dataclass

@dataclass
class PromptConfig:
    prompt: str
    width: int = 1024
    height: int = 1024
    is_negative_prompt: bool = False

@dataclass
class TextualPrompt:
    text: str
    width: int = 1024
    height: int = 1024
    is_negative_prompt: bool = False
