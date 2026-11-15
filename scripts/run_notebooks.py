"""Execute notebooks with isolated Jupyter caches and preserve frozen split files.

Example: python scripts/run_notebooks.py 02 04
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in (ROOT / 'data/processed/splits').glob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('numbers', nargs='+', choices=['01', '02', '03', '04'])
    args = parser.parse_args()
    audit_path = ROOT / 'results/tables/notebook_execution_audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    with tempfile.TemporaryDirectory(prefix='ml_jupyter_') as tmp:
        for key in ['IPYTHONDIR', 'JUPYTER_CONFIG_DIR', 'JUPYTER_RUNTIME_DIR', 'MPLCONFIGDIR']:
            p = Path(tmp) / key.lower()
            p.mkdir()
            os.environ[key] = str(p)
        os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['OPENBLAS_NUM_THREADS'] = '2'
        for number in args.numbers:
            path = next((ROOT / 'notebooks').glob(f'{number}_*.ipynb'))
            before = snapshot()
            started = time.perf_counter()
            print(f'RUN {path.name}', flush=True)
            nb = nbformat.read(path, as_version=4)
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    cell.outputs = []
                    cell.execution_count = None
            client = NotebookClient(nb, timeout=2400, kernel_name='python3',
                resources={'metadata': {'path': str(ROOT)}})
            def completed(cell, cell_index, **kwargs):
                if cell.cell_type == 'code':
                    print(f'  cell {cell_index} complete', flush=True)
            client.on_cell_executed = completed
            client.execute()
            assert before == snapshot(), 'Frozen splits changed during execution'
            nbformat.validate(nb)
            nbformat.write(nb, path)
            record = dict(seconds=round(time.perf_counter()-started, 2),
                code_cells=sum(c.cell_type == 'code' for c in nb.cells),
                errors=sum(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code' for o in c.outputs),
                notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                frozen_splits_unchanged=True)
            audit[path.name] = record
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
            print(f'DONE {path.name}: {record}', flush=True)


if __name__ == '__main__':
    main()
