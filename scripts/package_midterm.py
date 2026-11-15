"""Build a submission ZIP from an explicit allowlist, excluding customer data."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    allowed = {
        'src': {'.py'}, 'tests': {'.py'}, 'scripts': {'.py'},
        'notebooks': {'.ipynb'}, 'docs': {'.md'},
        'results': {'.csv', '.json', '.png'}, 'report': {'.md', '.html'},
        'slides': {'.md', '.html'},
    }
    files = [ROOT / name for name in ['README.md', 'DATA_GUIDE.txt', 'requirements.txt', 'requirements-tested.txt']]
    for folder, extensions in allowed.items():
        files.extend(p for p in (ROOT/folder).rglob('*')
                     if p.is_file() and p.suffix in extensions and
                     not any(part.startswith('.') or part == '__pycache__' for part in p.relative_to(ROOT).parts))
    # No per-customer predictions, even if someone later copies them into results.
    files = sorted(set(p for p in files if 'predictions' not in p.name and 'scores.csv' not in p.name))
    output = ROOT / 'dist'
    output.mkdir(exist_ok=True)
    archive = output / ('midterm_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.zip')
    manifest = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr('PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any(n.startswith(('data/', '.git/', '.venv/')) or n.endswith(('.joblib', '.pkl')) for n in z.namelist())
        assert all(sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    print(f'Created {archive}\nFiles: {len(files)}; no raw/processed customer data or trained models.')


if __name__ == '__main__':
    main()
