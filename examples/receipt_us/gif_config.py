"""Where each field is printed on receipt.jpg, for ../receipt/render_gif.py."""
CROP = (0, 0, 534, 700)
ROW = {1: (52, 276, 392, 295), 2: (52, 297, 392, 316), 3: (52, 316, 392, 337)}
AMOUNT = {'subtotal': (195, 402, 392, 422), 'tax': (195, 424, 392, 446), 'total': (100, 465, 410, 493)}
GUESTS = (170, 168, 240, 188)
SOURCES = {
    'store': [(52, 20, 170, 40)], 'date': [(52, 122, 130, 142)],
    'first_item': [ROW[1]], 'second_item': [ROW[2]], 'third_item': [ROW[3]],
    'table': [(190, 188, 340, 212)], 'item_lines': [(52, 276, 392, 337)], 'guests': [GUESTS],
    **{name: [box] for name, box in AMOUNT.items()},
    'currency': [(52, 60, 240, 82)], 'tips_accepted': [(70, 535, 420, 630)],
    'tax_rate_percent': [AMOUNT['subtotal'], AMOUNT['tax']],
    'per_person': [AMOUNT['total'], GUESTS],
}
COMPOSITE = {'item_lines'}
RESULT = 'result.json'
# expense_note hit the string-prompt bug in this run; see README.
HIDE = {'expense_note'}
