# -*- coding: utf-8 -*-
"""
LaViT prompt template utilities.
Aligned with the VisualMindTraining project, used for teacher feature extraction
and evaluation.
"""

from typing import List, Optional


def build_lavit_messages(
    image_path: str,
    question: str,
    system_message: Optional[str] = None
) -> List[dict]:
    """
    Build a standard LaViT message list (aligned with evaluation.py).

    Args:
        image_path: Path to the image.
        question: Question text.
        system_message: Optional system message (LaViT usually does not use one).

    Returns:
        List[dict]: Messages in OpenAI-style chat format.
    """
    messages = []
    
    # By default LaViT does not use a system message; pass it explicitly if needed.
    if system_message:
        messages.append({
            "role": "system",
            "content": system_message
        })
    
    # User message: image + question
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {
                "type": "text",
                "text": question
            },
        ],
    })
    
    return messages


def build_lavit_teacher_generation_messages(
    image_path: str,
    question: str,
    system_message: Optional[str] = None
) -> List[dict]:
    """
    Build messages for teacher model generation (single-shot full answer).

    Args:
        image_path: Path to the image.
        question: Question text.
        system_message: Optional system message.

    Returns:
        List[dict]: Messages in OpenAI-style chat format.
    """
    # Same as the base version since LaViT uses the standard format.
    return build_lavit_messages(image_path, question, system_message)


def build_lavit_stepwise_messages(
    image_path: str,
    question: str,
    previous_steps: List[str],
    system_message: Optional[str] = None
) -> List[dict]:
    """
    Build messages for step-wise reasoning (used to extract per-step features).

    Args:
        image_path: Path to the image.
        question: Question text.
        previous_steps: List of previously generated steps.
        system_message: Optional system message.

    Returns:
        List[dict]: Messages in OpenAI-style chat format.
    """
    messages = []
    
    if system_message:
        messages.append({
            "role": "system",
            "content": system_message
        })
    
    # User message
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {
                "type": "text",
                "text": question
            },
        ],
    })
    
    # If previous steps exist, append them as an assistant reply.
    if previous_steps:
        assistant_response = "\n".join(previous_steps)
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })
    
    return messages


def apply_lavit_chat_template(
    processor,
    messages: List[dict],
    add_generation_prompt: bool = True
) -> str:
    """
    Apply the chat template (wrapper around ``processor.apply_chat_template``).

    Args:
        processor: AutoProcessor instance.
        messages: Message list.
        add_generation_prompt: Whether to append a generation prompt (default True).

    Returns:
        str: Formatted text prompt.
    """
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )


# Compatibility: provide the same interface as the older utils.py
def build_prompt_with_chat_template(
    processor,
    question: str,
    steps_prefix: List[str],
    force_T: int = None,
    image_path: str = None
) -> str:
    """
    Backward-compatible interface using the LaViT standard format.

    Note: ``force_T`` is kept for compatibility but is not used, since LaViT
    does not force the number of reasoning steps in the prompt.

    Args:
        processor: AutoProcessor instance.
        question: Question text.
        steps_prefix: Previously generated steps.
        force_T: Deprecated step count parameter.
        image_path: Image path (if None, an image placeholder is used).

    Returns:
        str: Formatted prompt string.
    """
    # Note: this function includes an image placeholder in the user content
    # so that ``processor(text=prompt, images=image)`` can work correctly.
    messages = []
    
    # User message: image placeholder + question
    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},  # Image placeholder; processor will insert image tokens.
            {"type": "text", "text": question}
        ]
    })
    
    if steps_prefix:
        messages.append({
            "role": "assistant",
            "content": "\n".join(steps_prefix)
        })
    
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=bool(not steps_prefix)
    )


def build_teacher_generation_prompt(
    processor,
    question: str,
    force_T: int = None,
    image_path: str = None
) -> str:
    """
    Backward-compatible interface to build prompts for teacher generation.

    Args:
        processor: AutoProcessor instance.
        question: Question text.
        force_T: Deprecated step count parameter.
        image_path: Image path.

    Returns:
        str: Formatted prompt string.
    """
    # User message: image placeholder + question so that
    # ``processor(text=prompt, images=image)`` can handle images correctly.
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},  # Image placeholder; processor will insert image tokens.
            {"type": "text", "text": question}
        ]
    }]
    
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# Backward compatibility aliases for older call sites
build_lvr_messages = build_lavit_messages
build_lvr_teacher_generation_messages = build_lavit_teacher_generation_messages
build_lvr_stepwise_messages = build_lavit_stepwise_messages
apply_lvr_chat_template = apply_lavit_chat_template
