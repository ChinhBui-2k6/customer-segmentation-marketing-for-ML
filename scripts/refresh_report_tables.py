"""Refresh numeric Markdown tables directly from exported experiment results."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'results/tables'


def markdown(frame):
    def value(x):
        if pd.isna(x):
            return ''
        if isinstance(x, float):
            return f'{x:.4f}'
        return str(x).replace('|', '/')
    rows = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows += ['| ' + ' | '.join(value(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def main():
    report = ROOT / 'report/midterm_report.md'
    text = report.read_text(encoding='utf-8')
    metrics = pd.read_csv(TABLES / 'classification/classification_metrics.csv')
    selected = metrics.query("variant == 'tuned' and partition == 'validation'").copy()
    names = {'gaussian_nb': 'Gaussian NB', 'svm_rbf': 'SVM RBF', 'encoded': '37 chiều', 'pca': 'PCA 12'}
    for col in ['model', 'space']:
        selected[col] = selected[col].replace(names)
    gaps = pd.read_csv(TABLES / 'classification/classification_train_validation_gaps.csv')
    gaps = gaps.query("variant == 'tuned' and split == '80_20'").copy()
    for col in ['model', 'space']:
        gaps[col] = gaps[col].replace(names)
    frames = {
        'splits': pd.read_csv(TABLES/'data_split_summary.csv')[['partition','rows','response_0','response_1']],
        'pca': pd.read_csv(TABLES/'pca_experiment_summary.csv')[['split','retained_components','retained_variance_ratio','components_for_90pct','components_for_95pct']],
        'reduction': pd.read_csv(TABLES/'pca_kpca_comparison.csv')[['method','partition','rows','components','trustworthiness_10']],
        'clusters': pd.read_csv(TABLES/'kmeans_80_20_encoded_train_summary.csv')[['cluster','rows','response_1','response_rate']],
        'classification80': selected.query("split == '80_20'")[['model','space','precision','recall','f1','pr_auc','average_precision','fp','fn']],
        'classification_all': selected[['split','model','space','f1','average_precision','roc_auc']],
        'gaps': gaps[['model','space','train_average_precision','average_precision','train_minus_validation_average_precision']],
    }
    for key, frame in frames.items():
        start, end = f'<!-- table:{key} -->', f'<!-- end:{key} -->'
        assert text.count(start)==text.count(end)==1
        a, b = text.index(start)+len(start), text.index(end)
        text = text[:a]+'\n'+markdown(frame)+'\n'+text[b:]
    report.write_text(text, encoding='utf-8')
    print(f'Updated {len(frames)} tables: {report}')


if __name__ == '__main__':
    main()
