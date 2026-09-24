"""Where each field is printed on receipt.jpg, for render_gif.py."""
CROP = (85, 300, 765, 1060)
ROW = {1: (95, 548, 740, 584), 2: (95, 590, 740, 627), 3: (95, 632, 740, 670)}
AMOUNT = {name: (245, top, 745, top + 36) for name, top in
          [('subtotal', 764), ('discount', 804), ('total', 850), ('cash', 890), ('change', 940)]}
SOURCES = {
    'first_item': [ROW[1]], 'second_item': [ROW[2]], 'third_item': [ROW[3]],
    'item_lines': [(95, 548, 740, 670)], 'first_item_quantity': [(450, 548, 505, 584)],
    **{name: [box] for name, box in AMOUNT.items()},
    'paid_in_cash': [AMOUNT['cash']],
    'discount_percent': [AMOUNT['subtotal'], AMOUNT['discount']],
    'change_is_correct': [AMOUNT['total'], AMOUNT['cash'], AMOUNT['change']],
    'expense_note': [ROW[1], ROW[2], ROW[3], AMOUNT['total']],
}
# Boxes that overlap the line boxes; drawn only when their field is alone.
COMPOSITE = {'item_lines', 'first_item_quantity'}
RESULT = 'result_thinking.json'
HIDE = set()
