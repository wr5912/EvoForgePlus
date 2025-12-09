# -*- coding: utf-8 -*-
import dspy
import litellm

class LiteLLMAdapter(dspy.LM):
    def __init__(self, model_name, **kwargs):
        super().__init__(model=model_name)
        self.kwargs = kwargs
        self.history = []

    def __call__(self, prompt, **kwargs):
        params = {**self.kwargs, **kwargs}
        messages = [{"role": "user", "content": prompt}]
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                **params
            )
            content = response.choices[0].message.content
            self.history.append({"prompt": prompt, "response": content})
            return [content]
        except Exception as e:
            print(f"LLM Error: {e}")
            return ["Error"]

    def inspect_history(self, n=1):
        for i in range(min(n, len(self.history))):
            print(f"\n--- Call {len(self.history)-i} ---\n{self.history[-(i+1)]['response']}")