"""Diagnostic intervention only; production prompt remains unchanged."""
import json
import random
from pathlib import Path
from _thinking_eval import ThinkingClient
from typellm_runtime import Choice, _decode_numeric

c=ThinkingClient(model='qwen3.8-27b',tokenizer='RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead',timeout=180)
rows=[]
for i,expected in enumerate([24.5]*6+[-24.5]*2+[0.0,-85.25]):
    d=Choice(question='What is the invoice total?',choices={},name='total',syntax='Number',numeric_type='number')
    content=d.opening_text().replace('Return only the signed number answer.',
        "Return only a JSON number. For zero or positive values, start with a digit and never use '+'. Use '-' only for negative values. Do not use markdown.")
    messages=[{'role':'user','content':f'Invoice: total GBP {expected}, paid in full.'},{'role':'user','content':content}]
    prefix=c.render_chat(messages,add_generation_prompt=True)
    free=c._request('/generate',{'text':prefix,'sampling_params':{'max_new_tokens':32,'temperature':0}})
    value,_,text=_decode_numeric(c,prefix,d,'argmax',1,random.Random(0),32)
    row={'trial':i,'expected':expected,'free_answer':free.get('text'),'value':value,'correct':value==expected,'thinking_trace':c.traces[-1]}
    rows.append(row);print(json.dumps(row),flush=True)
Path('evals/thinking_prompt_probe.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
