"""Debug script to run pipeline headlessly with proper interrupt handling."""
import os, logging

logging.basicConfig(level=logging.INFO)
os.chdir(r'E:\Inter OS Data\ProjectVitae')

from langgraph.types import Command
from project_vitae.config import load_config
from project_vitae.graph import build_graph
from project_vitae.models import SessionState

cfg = load_config()
graph = build_graph(cfg, 'test-py3')

session_state = SessionState(
    session_name='test-py3',
    github_urls=['https://github.com/AKAASH297/VLM-Powered-OCR-ML-Diagram-Detection-Pipeline'],
    job_description='Senior software engineer with Python and ML',
)

config = {'configurable': {'thread_id': 'test-py3'}}


def _resume_value(state, next_nodes):
    for node in next_nodes or []:
        if node == 'filter_pause':
            sel = state.filter_proposal.selected if state.filter_proposal else []
            return {'action': 'confirm', 'selected': sel}
        elif node == 'review_pause':
            approved = [s.id for s in state.sections]
            return {'action': 'proceed', 'approved_ids': approved}
        elif node == 'compile_pause':
            return {'action': 'dismiss'}
    return {}


first = True
while True:
    stream = graph.stream(
        session_state.model_dump() if first else Command(resume=_resume_value(session_state, next_nodes)),
        config,
        stream_mode='updates',
    )
    first = False
    try:
        for event in stream:
            for node, output in event.items():
                if isinstance(output, dict):
                    if 'filter_proposal' in output and output['filter_proposal']:
                        fp = output['filter_proposal']
                        print(f'Filter: selected={fp.selected}')
                    if 'type' in output:
                        print(f'Interrupt: {output["type"]}')
    except Exception as e:
        import traceback; traceback.print_exc()
        break

    snap = graph.get_state(config)
    next_nodes = snap.next
    if not next_nodes:
        print('Pipeline completed')
        break
    print(f'Pending: {next_nodes}')
    session_state = SessionState.model_validate(snap.values)
