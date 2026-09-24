"""Render an example's demo.gif from its recorded result; no model is called."""
import argparse
import json
import runpy
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
FONTS = HERE.parents[2] / 'typellm-site' / 'assets'
W, H = 1200, 675
BG, FG, MUTED, LINE = '#0d0d0f', '#ececec', '#8d8d96', '#23232a'
TYPE_COLORS = {'string': '#7f9bff', 'integer': '#e0b86a', 'number': '#e0b86a', 'boolean': '#6fcf97'}


def font(name, size):
    return ImageFont.truetype(str(FONTS / f'JetBrainsMono-{name}.ttf'), size)


REG, BOLD, SMALL, TITLE = font('Regular', 17), font('SemiBold', 17), font('Regular', 13), font('SemiBold', 22)


def literal(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return json.dumps(value, ensure_ascii=False) if isinstance(value, str) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', nargs='?', type=Path, default=HERE,
                        help='folder with receipt.jpg, receipt.py and gif_config.py')
    example = parser.parse_args().example.resolve()
    config = runpy.run_path(str(example / 'gif_config.py'))
    CROP, SOURCES = config['CROP'], config['SOURCES']
    questions = runpy.run_path(str(example / 'receipt.py'))['QUESTIONS']
    result = {name: value for name, value in
              json.loads((example / config['RESULT']).read_text())['result'].items()
              if name not in config['HIDE']}
    types = {name: spec['type'] for name, spec in questions.items()}
    DERIVED = [name for name in result if 'depends_on' in questions[name]]

    scale = (H - 90) / (CROP[3] - CROP[1])
    photo = Image.open(example / 'receipt.jpg').crop(CROP)
    photo = photo.resize((round(photo.width * scale), round(photo.height * scale)), Image.LANCZOS)
    px, py = 40, 60

    def to_canvas(box):
        x0, y0, x1, y1 = box
        return (px + (x0 - CROP[0]) * scale, py + (y0 - CROP[1]) * scale,
                px + (x1 - CROP[0]) * scale, py + (y1 - CROP[1]) * scale)

    left = px + photo.width + 36
    names = [n for n in result if n not in DERIVED] + DERIVED

    def frame(progress, active):
        img = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((40, 22), 'TypeLLM', font=TITLE, fill=FG)
        d.rectangle((40 + d.textlength('TypeLLM', font=TITLE) + 5, 26, 40 + d.textlength('TypeLLM', font=TITLE) + 12, 48), fill='#7f9bff')
        d.text((left, 28), 'client.generate(images=[receipt], questions=...)', font=SMALL, fill=MUTED)
        img.paste(photo, (px, py))
        # With many fields active, the line boxes already cover the composite ones.
        shown = [n for n in active if len(active) == 1 or n not in config['COMPOSITE']]
        for box in {box for name in shown for box in SOURCES[name]}:
            d.rounded_rectangle(to_canvas(box), radius=4, outline='#7f9bff', width=3)
        y = py + 4
        for name in names:
            if name == names[0]:
                d.text((left, y - 2), 'layer 1 · no dependencies', font=SMALL, fill=MUTED)
                y += 24
            if name == DERIVED[0]:
                y += 8
                d.line((left, y, W - 40, y), fill=LINE)
                d.text((left, y + 6), 'layer 2 · depends_on layer 1', font=SMALL, fill=MUTED)
                y += 28
            if name not in progress:
                y += 29
                continue
            kind = types[name]
            d.text((left, y), name, font=REG, fill=FG)
            d.text((left + 205, y + 3), kind, font=SMALL, fill=TYPE_COLORS[kind])
            text = literal(result[name])
            if progress[name] is not None:
                text = text[:progress[name]] + '▌'
            if name == 'expense_note':
                for i, part in enumerate(textwrap.wrap(text, 26)[:4]):
                    d.text((left + 280, y + i * 22), part, font=BOLD, fill=TYPE_COLORS[kind])
            else:
                d.text((left + 280, y), text, font=BOLD, fill=TYPE_COLORS[kind])
            y += 29
        d.text((left, H - 34), f'Qwen3.8-27B · one generate() call', font=SMALL, fill=MUTED)
        return img

    frames, durations = [frame({}, [])], [800]
    done = {}
    for layer in (names[:-len(DERIVED)], DERIVED):
        texts = {name: literal(result[name]) for name in layer}
        steps = 12
        for step in range(steps + 1):
            # Every field in the layer types at once, each at its own pace.
            progress = {**done, **{name: min(len(t), -(-len(t) * step // steps)) if step < steps else None
                                   for name, t in texts.items()}}
            frames.append(frame(progress, layer)); durations.append(70)
        done.update({name: None for name in layer})
        durations[-1] = 500  # Short pause: layer 2 starts right after layer 1.
    frames.append(frame(done, [])); durations.append(3500)

    # Give every accent its own palette entry; median cut drops small clusters.
    sample = frames[-1].copy()
    swatches = ImageDraw.Draw(sample)
    for i, color in enumerate([FG, MUTED, LINE, '#7f9bff', *TYPE_COLORS.values()]):
        swatches.rectangle((i * 40, 0, i * 40 + 39, 40), fill=color)
    palette = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out = example / 'demo.gif'
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(out, len(frames), 'frames', round(sum(durations) / 1000, 1), 's', out.stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
