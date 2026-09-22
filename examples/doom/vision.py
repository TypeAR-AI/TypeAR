"""Multi-image Qwen VL adapter for the Doom example (sequential or batch execution)."""
from typellm_sglang import SGLangClient, SGLangError


class DoomVisionClient(SGLangClient):
    images: list[str] | None = None

    def render_chat(self, messages, *, add_generation_prompt):
        if not self.images:
            raise SGLangError("Set a screenshot before making a decision")
        messages = [dict(message) for message in messages]
        first = next(message for message in messages if message["role"] == "user")
        content = []
        for index in range(len(self.images)):
            label = "CURRENT" if index == len(self.images) - 1 else "PREVIOUS"
            content.extend([
                {"type": "text", "text": f"Image {index + 1} / {label}:"},
                {"type": "image"},
            ])
        first["content"] = content + [{"type": "text", "text": first["content"]}]
        prompt = super().render_chat(messages, add_generation_prompt=add_generation_prompt)
        if prompt.count("<|image_pad|>") != len(self.images):
            raise SGLangError("This demo requires a Qwen VL image chat template")
        return prompt

    def _request(self, path, payload=None, **kwargs):
        if path == "/generate":
            if not self.images:
                raise SGLangError("Missing screenshot")
            text = payload.get("text")
            if isinstance(text, list):
                images = [list(self.images) for _ in text]
            elif isinstance(text, str):
                images = list(self.images)
            else:
                raise SGLangError("Expected text or a batch of texts")
            payload = {**payload, "image_data": images}
        return super()._request(path, payload, **kwargs)
