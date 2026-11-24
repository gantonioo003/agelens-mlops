"""Create aggregate-only training EDA and a fully executed reproducibility notebook."""

import argparse
import asyncio
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

from scripts.audit_data import validate_manifests
from scripts.prepare_assets import ROOT, sha256

PALETTE = ["#007F73", "#D46243", "#3E69A2", "#A67C00", "#784C85", "#657A46"]


def build_figures():
    output = ROOT / "reports/task04"
    summary = json.loads((output / "audit_summary.json").read_text())
    manifest_dir = ROOT / "data/manifests"
    for name, digest in summary["manifest_sha256"].items():
        if sha256(manifest_dir / name) != digest:
            raise ValueError(f"Frozen manifest changed: {name}")
    selected = pd.read_csv(manifest_dir / "selected.csv")
    validate_manifests(selected, 15000)
    train = selected[selected.split == "train"].copy()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    findings = []

    def save(number, name, figure, title, interpretation):
        figure.tight_layout()
        filename = f"{number:02d}_{name}.png"
        figure.savefig(figures / filename, dpi=160, bbox_inches="tight")
        plt.close(figure)
        findings.append(
            {
                "number": number,
                "title": title,
                "filename": filename,
                "interpretation": interpretation,
            }
        )

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(train.age, bins=np.arange(0, 106, 5), color=PALETTE[0], edgecolor="white")
    ax.set(
        xlabel="Idade anotada (anos)",
        ylabel="Imagens de treino",
        title=f"Distribuicao etaria do treino (n={len(train):,})",
    )
    ax.axvline(18, color=PALETTE[1], linestyle="--", label="18 anos")
    ax.legend()
    minority = int((train.age < 18).sum())
    save(
        1,
        "idades_treino",
        fig,
        "Distribuicao de idade",
        f"O treino tem {len(train)} imagens, idade media {train.age.mean():.2f}, "
        f"mediana {train.age.median():.0f} e intervalo {train.age.min()} a {train.age.max()} anos. "
        f"Menores de 18 somam {minority} ({100 * minority / len(train):.1f}%). "
        "A distribuicao nao e uniforme; MAE global pode ocultar erros nas idades jovens. "
        "Idades extremas validas foram mantidas dentro do recorte, sem remocao por serem raras.",
    )

    thresholds = [12, 13, 15, 16, 18, 21]
    minors = np.array([(train.age < threshold).sum() for threshold in thresholds])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    positions = np.arange(len(thresholds))
    ax.bar(positions, minors, label="Menor que T", color=PALETTE[0])
    ax.bar(positions, len(train) - minors, bottom=minors, label="Idade >= T", color=PALETTE[1])
    ax.set(
        xticks=positions,
        xticklabels=thresholds,
        xlabel="Limiar T em anos",
        ylabel="Imagens de treino",
        title="Balanceamento das tarefas binarias no treino",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    percentages = ", ".join(
        f"<{threshold}: {100 * count / len(train):.1f}%"
        for threshold, count in zip(thresholds, minors, strict=True)
    )
    save(
        2,
        "balanceamento_tarefas",
        fig,
        "Balanceamento por limiar",
        f"Proporcoes positivas no treino: {percentages}. A classe positiva muda com T; "
        "as seis tarefas nao sao independentes. A amostragem por idade do protocolo sera "
        "aplicada apenas no treino. Validacao e teste nao recebem sobreamostragem nem pesos de treino.",
    )

    demographic = pd.crosstab(train.race, train.gender).reindex(
        index=range(5), columns=[0, 1], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(9, 4.2))
    demographic.plot.bar(ax=ax, color=PALETTE[:2], rot=0)
    ax.set(
        xlabel="Codigo de categoria demografica UTKFace (0 a 4)",
        ylabel="Imagens de treino",
        title="Representacao das anotacoes demograficas no treino",
    )
    ax.legend(["Codigo de genero 0", "Codigo de genero 1"])
    largest = train.race.value_counts().idxmax()
    smallest = train.race.value_counts().idxmin()
    save(
        3,
        "demografia_treino",
        fig,
        "Representacao demografica anotada",
        f"A categoria {largest} tem {int((train.race == largest).sum())} registros e a categoria "
        f"{smallest} tem {int((train.race == smallest).sum())}. Os codigos descrevem os rotulos "
        "binarios/de categorias do dataset, nao identidade de genero ou etnia verificada. "
        "Nao entram como atributos preditores. O desbalanceamento exige cautela ao generalizar; "
        "esta figura nao mede equidade do modelo, pois nenhum modelo foi treinado.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, field, label, color in zip(
        axes,
        ["brightness", "contrast", "sharpness"],
        ["Brilho medio (0-255)", "Contraste (desvio-padrao)", "log10(1 + variancia Laplaciana)"],
        PALETTE[:3],
        strict=True,
    ):
        values = np.log10(1 + train[field]) if field == "sharpness" else train[field]
        ax.hist(values, bins=30, color=color, edgecolor="white")
        ax.set(xlabel=label, ylabel="Imagens de treino")
    fig.suptitle("Qualidade visual medida em escala cinza padronizada 64 x 64", y=1.04)
    save(
        4,
        "qualidade_treino",
        fig,
        "Variacao de qualidade visual",
        f"Medianas de brilho {train.brightness.median():.1f}, contraste {train.contrast.median():.1f} "
        f"e variancia Laplaciana {train.sharpness.median():.1f}. As medidas usam o mesmo tamanho "
        "64 x 64, para comparar imagens em condicoes consistentes. Elas nao provam qualidade facial "
        "ou idade correta. Imagens escuras, claras ou pouco nitidas nao foram excluidas por limiar "
        "arbitrario, evitando facilitar artificialmente o problema.",
    )

    sizes = train.groupby(["width", "height"]).size().sort_values(ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar([f"{width} x {height}" for width, height in sizes.index], sizes, color=PALETTE[2])
    ax.set(
        xlabel="Resolucao apos orientacao EXIF",
        ylabel="Imagens de treino",
        title="Resolucao dos arquivos de treino",
    )
    save(
        5,
        "resolucao_treino",
        fig,
        "Formato e resolucao",
        f"Foram observadas {train[['width', 'height']].drop_duplicates().shape[0]} resolucoes no treino "
        f"e modos originais {train.original_mode.value_counts().to_dict()}. A conversao RGB e a "
        "orientacao EXIF sao deterministicas. O FaRL ainda exigira seu preprocessamento oficial "
        "224 x 224; aumentar uma imagem pequena nao recupera detalhes perdidos.",
    )

    reasons = {
        "selected": "Selecionadas",
        "outside_sample_cap": "Fora do teto amostral",
        "conflicting_exact_duplicate_labels": "Conflito em pixels identicos",
        "exact_pixel_duplicate": "Copia exata redundante",
        "outside_age_scope": "Idade fora de 0-101",
        "invalid_label": "Nome/rotulo inconsistente",
        "invalid_image": "Imagem invalida",
    }
    counts = summary["exclusion_counts"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.barh([reasons[key] for key in counts], list(counts.values()), color=PALETTE[0])
    ax.bar_label(bars, padding=4)
    ax.set(
        xlabel="Arquivos (categorias mutuamente exclusivas)", title="Destino dos arquivos auditados"
    )
    ax.set_xlim(0, max(counts.values()) * 1.18)
    ax.invert_yaxis()
    save(
        6,
        "auditoria_destinos",
        fig,
        "Fluxo de exclusao e amostragem",
        f"Todos os {summary['files_audited']} arquivos foram auditados; {summary['images_decoded']} "
        f"decodificaram. Foram excluidos {counts.get('invalid_label', 0)} nomes inconsistentes, "
        f"{counts.get('outside_age_scope', 0)} idades fora do recorte, "
        f"{counts.get('exact_pixel_duplicate', 0)} copias exatas redundantes e "
        f"{counts.get('conflicting_exact_duplicate_labels', 0)} imagens de grupos exatos com rotulos "
        f"conflitantes. Outros {counts.get('outside_sample_cap', 0)} registros elegiveis ficaram fora "
        "somente pelo teto de custo, nao por erro de modelo. As exclusoes conservadoras podem "
        "alterar a distribuicao e devem constar como limitacao da reproducao reduzida.",
    )

    splits = ["train", "val_select", "val_calibration", "test"]
    supports = pd.DataFrame(
        {
            split: [summary["support"][str(t)][split]["minor"] for t in thresholds]
            for split in splits
        },
        index=thresholds,
    )
    fig, ax = plt.subplots(figsize=(9, 4.2))
    heat = ax.imshow(supports.values, cmap="YlGnBu", aspect="auto")
    ax.set(
        xticks=range(4),
        xticklabels=splits,
        yticks=range(6),
        yticklabels=thresholds,
        xlabel="Particao congelada",
        ylabel="Limiar T",
        title="Suporte de menores por tarefa e particao",
    )
    for row in range(6):
        for col in range(4):
            value = int(supports.iloc[row, col])
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > supports.to_numpy().max() * 0.55 else "black",
            )
    fig.colorbar(heat, ax=ax, label="Imagens com idade < T")
    weakest = int(supports["val_calibration"].min())
    test_weakest = int(supports["test"].min())
    save(
        7,
        "suporte_splits",
        fig,
        "Suporte e precisao do alvo de recall",
        f"O menor suporte positivo e {weakest} na calibracao e {test_weakest} no teste. "
        "Um unico falso adulto muda a FNR em 1/n; recall de 0,99 nao pode ser tratado como garantia. "
        "A tabela usa somente contagens de suporte previstas para auditar a particao, nao escolhas "
        "de modelo pelo teste. Dependencia por pessoa pode reduzir o tamanho efetivo da amostra; "
        "a separacao por identidade nao foi comprovada.",
    )

    fig, ax = plt.subplots(figsize=(9, 4.2))
    table = train.groupby(["cv_fold", "stratum"]).size().unstack(fill_value=0)
    table.div(table.sum(axis=1), axis=0).T.plot.bar(ax=ax, color=PALETTE[:3], rot=45)
    boundaries = json.loads((ROOT / "configs/protocol-v1.json").read_text())["splits"][
        "fallback_age_boundaries"
    ]
    ax.set_xticklabels(
        [f"{boundaries[int(value)]}-{boundaries[int(value) + 1] - 1}" for value in table.columns]
    )
    ax.set(
        xlabel="Estrato etario em anos (limites inclusivos)",
        ylabel="Fracao do fold",
        title="Distribuicao dos tres folds internos de treino",
    )
    ax.legend(title="Fold")
    fold_sizes = train.cv_fold.value_counts().sort_index().to_dict()
    save(
        8,
        "folds_treino",
        fig,
        "Validacao cruzada sem romper grupos",
        f"Tamanhos dos folds internos: {fold_sizes}. Todos sao subconjuntos exclusivos do treino "
        "e grupos de duplicatas nao cruzam folds. Pequenas diferencas entre as proporcoes sao "
        "esperadas ao preservar grupos inteiros. Normalizacao, amostragem e tuning futuros "
        "deverao ser ajustados dentro de cada fold, sem usar seus exemplos de avaliacao.",
    )

    (output / "eda_findings.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    train.age.describe().rename("value").to_csv(output / "train_age_statistics.csv")
    demographic.to_csv(output / "train_demographic_counts.csv")
    supports.to_csv(output / "split_minor_support.csv")
    train[["brightness", "contrast", "sharpness"]].describe().to_csv(
        output / "train_quality_statistics.csv"
    )
    report = [
        "# AgeLens | Task 04",
        "",
        "## Auditoria integral e EDA",
        "",
        f"Arquivos auditados: {summary['files_audited']}. Imagens selecionadas: {summary['selected']}.",
        f"Particoes: {summary['split_sizes']}.",
        "",
        "Auditoria tecnica nao confirma idade documental, identidade ou consentimento das pessoas.",
        "Nao houve treinamento. A EDA de atributos usa apenas o treino; os demais splits aparecem somente em verificacoes de suporte e integridade.",
        "",
        "## Referencia e duplicatas",
        "",
        f"Metadados de referencia: {summary['reference']['eligible_matches']} de {summary['reference']['eligible_total']} elegiveis correspondem; {summary['reference']['cross_partition_similarity_groups']} grupos conservadores cruzariam as particoes originais. Aplicado o fallback predefinido, com grupos indivisiveis.",
        "Correspondencia de metadados nao prova identidade. Grupos perceptuais sao precaucao contra copias proximas, nao reconhecimento facial nem verificacao humana exaustiva.",
        "",
    ]
    for finding in findings:
        report.extend(
            [
                f"## {finding['number']}. {finding['title']}",
                "",
                f"![{finding['title']}](../reports/task04/figures/{finding['filename']})",
                "",
                finding["interpretation"],
                "",
            ]
        )
    report.extend(
        [
            "## Preprocessamento definido",
            "",
            "Nomes incompletos e grupos exatos com rotulos conflitantes ficam em quarentena no manifesto local. Nao imputar idade-alvo, genero ou categoria demografica. Todos os arquivos sao preservados; exclusao significa nao selecionar para o experimento.",
            "Manter extremos de qualidade e idades validas no recorte. Conversao RGB, orientacao EXIF e preprocessamento FaRL sao deterministas. Escalonamento KNN e balanceamento etario serao ajustados apenas no treino/fold, nunca globalmente.",
            "",
            "## Congelamento e limitacoes",
            "",
            "Os manifests privados incluem caminhos relativos, hashes de arquivo/pixels, grupo conservador, split e fold. Os hashes publicados permitem detectar alteracoes, mas nao tornam o split subject-exclusive.",
            "Teste reservado: nenhuma predicao, tuning ou avaliacao de modelo foi produzida. A proxima etapa e extrair features FaRL conforme esses manifests congelados.",
            "",
        ]
    )
    (ROOT / "docs/task-04-auditoria-eda.md").write_text("\n".join(report), encoding="utf-8")
    return findings


def build_notebook(execute=True):
    findings = build_figures()
    cells = [
        nbformat.v4.new_markdown_cell(
            "# AgeLens - Task 04: auditoria integral e EDA\n\n"
            "Auditoria tecnica de todas as imagens, tratamento de duplicatas, splits agrupados e "
            "EDA interpretada. Nenhum modelo e treinado. Rostos e manifests individuais permanecem locais."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport sys, json, platform\n"
            "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'configs/audit-v1.json').exists())\n"
            "sys.path.insert(0, str(ROOT))\n"
            "assert Path(sys.prefix).resolve() == (ROOT / '.venv').resolve(), 'Use o ambiente AgeLens'\n"
            "print('Python', platform.python_version())\n"
            "from scripts.audit_data import run_audit\nfrom scripts.build_eda import build_figures\n"
            "from IPython.display import display, Image, Markdown\n"
            "before = json.loads((ROOT / 'reports/task04/audit_summary.json').read_text())['manifest_sha256']\n"
            "summary = run_audit(workers=4)\n"
            "assert before == summary['manifest_sha256'], 'Manifests mudaram ao repetir a auditoria'\n"
            "print('Auditoria repetida integralmente; manifests identicos e invariantes aprovadas.')\n"
            "print(json.dumps({k:summary[k] for k in ['files_audited','images_decoded','selected','exclusion_counts','reference','split_sizes']}, indent=2))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Metodo e limites\n\n"
            "Verificacao de bytes contra o arquivo original, decodificacao integral, SHA-256 de arquivo e "
            "pixels RGB, rotulos do nome, duplicatas exatas e candidatas perceptuais (dHash + erro RGB). "
            "Conflitos exatos sao excluidos; similaridade so agrupa particoes. Nao se verificou identidade "
            "ou idade real por inspecao documental. O split de referencia e usado apenas se compativel; "
            "o fallback foi registrado antes dos experimentos. Nenhuma imagem facial e exibida."
        ),
        nbformat.v4.new_code_cell(
            "findings = build_figures()\nprint(f'{len(findings)} graficos agregados gerados com interpretacoes.')"
        ),
    ]
    for index, finding in enumerate(findings):
        cells.extend(
            [
                nbformat.v4.new_markdown_cell(f"## {finding['number']}. {finding['title']}"),
                nbformat.v4.new_code_cell(
                    f"finding = findings[{index}]\n"
                    "display(Image(filename=str(ROOT / 'reports/task04/figures' / finding['filename'])))\n"
                    "display(Markdown(finding['interpretation']))"
                ),
            ]
        )
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Conclusao e proximos passos\n\n"
            "Ha desbalanceamento etario/demografico e rotulos conflitantes em imagens identicas. "
            "O recorte preserva grupos e e limitado a pesquisa neste dataset. Nenhuma metrica de modelo "
            "foi estimada. Preprocessamento: RGB/EXIF deterministico, nao imputar alvos, nao excluir por "
            "brilho/nitidez, escalonar e balancear somente dentro do treino/fold.\n\n"
            "Proxima task: extracao congelada de features, sem alterar os manifests nem consultar o teste. "
            "Os resultados nao justificam verificacao etaria real ou controle de acesso."
        )
    )
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (AgeLens)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    path = ROOT / "notebooks/04_auditoria_eda.ipynb"
    if execute:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        locations = {
            "IPYTHONDIR": ROOT / "runtime/ipython",
            "JUPYTER_RUNTIME_DIR": ROOT / "runtime/jupyter",
        }
        previous = {key: os.environ.get(key) for key in locations}
        try:
            for key, directory in locations.items():
                directory.mkdir(parents=True, exist_ok=True)
                os.environ[key] = str(directory)
            NotebookClient(
                notebook,
                timeout=1200,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if any(
            output.output_type == "error"
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
        ):
            raise AssertionError("Notebook has error outputs")
    nbformat.write(notebook, path)
    print(f"Notebook {'executed and saved' if execute else 'created'}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    build_notebook(execute=not args.no_execute)
