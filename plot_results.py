"""Plot authentic saved measurements, not paper benchmark figures."""
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parent
    data=json.loads((root/'results/benchmark.json').read_text())
    summaries=data['summary']
    for task in ('fixed','content','content_ood'):
        rows=[s for s in summaries if s['task']==task]
        names=[x['model'].replace('_',' ') for x in rows]
        values=[x['accuracy_mean']*100 for x in rows]
        errors=[x['accuracy_std']*100 for x in rows]
        plt.figure(figsize=(9,4.5))
        plt.bar(names,values,yerr=errors,capsize=4)
        plt.ylim(0,105)
        plt.ylabel('Held-out accuracy (%)')
        plt.title(f'{task.title()} token retrieval: independent synthetic benchmark')
        plt.xticks(rotation=15)
        plt.tight_layout()
        path=root/f'figures/{task}_accuracy.png'
        path.parent.mkdir(exist_ok=True)
        plt.savefig(path,dpi=160)
        plt.close()
        print(path)

if __name__=='__main__':
    main()
