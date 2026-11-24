"""Publish executed aggregate diagnostics, never feature vectors or personal image rows."""

import asyncio
import os
import sys

import nbformat
from nbclient import NotebookClient

from scripts.prepare_assets import ROOT


def main():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# AgeLens - Task 05: features FaRL congeladas\n\n"
                "Representacoes FP32 de 512 dimensoes, com L2 por imagem e preprocessamento oficial. "
                "Sem treinamento, ajuste de scaler/PCA, estimativa de idade ou alteracao dos splits. "
                "O cache privado e gerado pela CLI e somente consumido por este notebook."
            ),
            nbformat.v4.new_code_cell(
                "import json\nimport sys\nfrom pathlib import Path\n\n"
                "import numpy as np\nimport pandas as pd\nfrom IPython.display import display\n\n"
                "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'configs/features-v1.json').exists())\n"
                "assert Path(sys.prefix).resolve() == (ROOT / '.venv').resolve()\n"
                "sys.path.insert(0, str(ROOT))\n"
                "from agelens.cache_access import load_active_cache\n\n"
                "directory, features, rows, stats, metadata = load_active_cache(ROOT)\n"
                "verification = json.loads((ROOT / 'reports/task05/verification.json').read_text())\n"
                "assert verification['completed'] and not verification['trained']\n"
                "assert verification['cache_fingerprint'] == metadata['fingerprint']\n"
                "print(json.dumps(stats, indent=2))"
            ),
            nbformat.v4.new_markdown_cell(
                "## Correspondencia por particao\n\n"
                "Os indices sao derivados dos manifests congelados. Os tres folds sao exclusivos do treino; "
                "-1 indica que a particao nao participa da validacao cruzada. Nenhum dado e redistribuido."
            ),
            nbformat.v4.new_code_cell(
                "summary = []\n"
                "for name in ['train','val_select','val_calibration','test']:\n"
                "    indices = np.load(directory / f'indices_{name}.npy', allow_pickle=False)\n"
                "    assert rows.iloc[indices]['split'].eq(name).all()\n"
                "    summary.append({'particao':name, 'imagens':len(indices), 'dimensoes':features.shape[1], "
                "'folds':sorted(map(int, rows.iloc[indices].cv_fold.unique()))})\n"
                "display(pd.DataFrame(summary))"
            ),
            nbformat.v4.new_markdown_cell(
                "## Verificacao independente\n\n"
                "O comprovante registra integridade de todos os arquivos/imagens selecionadas e recomputacao "
                "de quatro imagens por particao apos recarregar o checkpoint. Diferencas entre inferencia "
                "individual e em lote sao avaliadas com tolerancia FP32 explicita, nao igualdade artificial."
            ),
            nbformat.v4.new_code_cell(
                "keys = ['samples','feature_shape','dtype','finite','l2_normalized',"
                "'all_selected_image_hashes_verified','all_cache_file_hashes_verified',"
                "'independent_recomputed_rows','recomputation_max_absolute_error','recomputation_tolerance',"
                "'single_image_repeats_bitwise_equal','encoder_state_unchanged','all_parameters_frozen',"
                "'gradients_absent','unit_tests_passed','trained']\n"
                "print(json.dumps({key:verification[key] for key in keys}, indent=2))"
            ),
            nbformat.v4.new_markdown_cell(
                "## Limites e proxima etapa\n\n"
                "Features faciais permanecem fora do Git. O cache nao comprova idade real, identidade ou "
                "desempenho de classificacao. O risco de identidade repetida entre fotos diferentes continua "
                "sendo uma limitacao dos metadados. A task 06 preparara MLflow; os modelos e suas metricas "
                "serao trabalhados somente nas etapas de treinamento autorizadas."
            ),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (AgeLens)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
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
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert all(
        output.output_type != "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )
    path = ROOT / "notebooks/05_features_farl.ipynb"
    nbformat.write(notebook, path)
    print("Feature diagnostics notebook executed and saved.", flush=True)


if __name__ == "__main__":
    main()
