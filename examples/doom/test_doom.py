import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import buttons, decision_context, history_image, png_data, questions
import base64
import io
import numpy as np
from PIL import Image
from vision import DoomVisionClient
from typellm_sglang import SGLangClient, SGLangError


class DoomTests(unittest.TestCase):
    def test_action_history_window(self):
        import json
        history = [{"step": i, "action": {"turn": "left", "shoot": False, "duration": 2}}
                   for i in range(1, 400)]
        context = decision_context(history, 400, None, [398, 399, 400])
        actions = json.loads(context.rsplit("\n", 1)[1])
        self.assertEqual([a["step"] for a in actions], list(range(390, 400)))
        self.assertTrue(decision_context(history, 400, None, [400], 0).endswith("\n[]"))

    def test_full_action_history_and_recent_images(self):
        frame = png_data(np.zeros((480, 640, 3), dtype=np.uint8))
        history = [{"step": i, "action": {"turn": "left", "shoot": False, "duration": 2}, "image": frame}
                   for i in range(1, 6)]
        images, steps = history_image(history, frame, 6, 3)
        self.assertEqual(steps, [4, 5, 6])
        self.assertEqual(images, [frame, frame, frame])
        for image in images:
            with Image.open(io.BytesIO(base64.b64decode(image))) as original:
                self.assertEqual(original.size, (640, 480))
        context = decision_context(history, 6, 8, steps)
        self.assertIn('"step":1', context)
        self.assertIn('"step":5', context)
        self.assertIn('"duration":2', context)
        self.assertNotIn("Each previous action lasted", context)
        self.assertNotIn(frame, context)
        self.assertEqual(history_image([], frame, 1, 3)[1], [1])
        self.assertEqual(history_image(history, frame, 6, 1)[1], [6])

    def test_actions(self):
        self.assertEqual(buttons({"move": "left", "shoot": True, "duration": 8}, "basic"), [1, 0, 1])
        self.assertEqual(buttons({"turn": "right", "shoot": False, "duration": 2}, "defend_the_center"), [0, 1, 0])
        with self.assertRaises(ValueError):
            buttons({"move": "left", "shoot": "false", "duration": 1}, "basic")

    def test_all_eight_durations_are_allowed(self):
        self.assertEqual(questions("defend_the_center")["duration"]["enum"], list(range(1, 9)))
        for duration in range(1, 9):
            self.assertEqual(buttons({"turn": "left", "shoot": False, "duration": duration}, "defend_the_center"), [1, 0, 0])

    def test_reject_invalid_duration(self):
        for duration in (True, 0, 9, 2.0, "2"):
            with self.assertRaises(ValueError):
                buttons({"turn": "left", "shoot": False, "duration": duration}, "defend_the_center")

    def test_image_attached_to_each_generate_not_tokenize(self):
        client = DoomVisionClient()
        client.images = ["FRAME_ONE"]
        with patch.object(SGLangClient, "_request", return_value={}) as request:
            client._request("/generate", {"text": "first"})
            self.assertEqual(request.call_args.args[1]["image_data"], ["FRAME_ONE"])
            client.images = ["FRAME_ONE", "FRAME_TWO", "FRAME_THREE"]
            client._request("/generate", {"text": "second"})
            self.assertEqual(request.call_args.args[1]["image_data"], ["FRAME_ONE", "FRAME_TWO", "FRAME_THREE"])
            client._request("/generate", {"text": ["turn", "shoot", "duration"]})
            self.assertEqual(request.call_args.args[1]["image_data"],
                             [["FRAME_ONE", "FRAME_TWO", "FRAME_THREE"]] * 3)
            client._request("/v1/tokenize", {"prompt": "A"})
            self.assertNotIn("image_data", request.call_args.args[1])

    def test_image_only_in_first_user_turn_and_history_preserved(self):
        client = DoomVisionClient()
        client.images = ["PNG1", "PNG2", "PNG3"]
        messages = [{"role": "user", "content": "view"},
                    {"role": "assistant", "content": "A"},
                    {"role": "user", "content": "shoot?"}]
        with patch.object(SGLangClient, "render_chat", return_value="<|image_pad|>" * 3) as render:
            client.render_chat(messages, add_generation_prompt=True)
            rendered = render.call_args.args[0]
            self.assertEqual(sum(block["type"] == "image" for block in rendered[0]["content"]), 3)
            self.assertIn("CURRENT", rendered[0]["content"][-3]["text"])
            self.assertEqual(rendered[2], messages[2])
            self.assertEqual(messages[0]["content"], "view")
        with patch.object(SGLangClient, "render_chat", return_value="text only"):
            with self.assertRaises(SGLangError):
                client.render_chat(messages, add_generation_prompt=True)


if __name__ == "__main__":
    unittest.main()
