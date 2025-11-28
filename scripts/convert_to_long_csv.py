import os
import sys
import argparse
import pandas as pd

def _sniff_sep(path):
    try:
        with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            lines = [next(f) for _ in range(5)]
    except Exception:
        try:
            with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                lines = [next(f) for _ in range(5)]
        except Exception:
            lines = []
    text = "\n".join(lines)
    return ';' if text.count(';') >= text.count(',') else ','

def _try_read_csv(path, sep, encoding, relax=False):
    try:
        if relax:
            return pd.read_csv(path, skipinitialspace=True, encoding=encoding, sep=sep, engine='python', on_bad_lines='skip')
        return pd.read_csv(path, skipinitialspace=True, encoding=encoding, sep=sep, engine='python')
    except Exception:
        return None

def load_input(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        try:
            return pd.read_excel(path)
        except Exception:
            return None
    sep = _sniff_sep(path)
    df = _try_read_csv(path, sep, 'utf-8-sig')
    if df is None or df.shape[1] == 1:
        df = _try_read_csv(path, (';' if sep == ',' else ','), 'utf-8-sig')
    if df is None or df.shape[1] == 1:
        df = _try_read_csv(path, sep, 'latin-1')
    if df is None or df.shape[1] == 1:
        df = _try_read_csv(path, (';' if sep == ',' else ','), 'latin-1')
    if df is None:
        df = _try_read_csv(path, sep, 'utf-8-sig', relax=True)
        if df is None or df.shape[1] == 1:
            df = _try_read_csv(path, (';' if sep == ',' else ','), 'utf-8-sig', relax=True)
    return df

def normalize_headers(df):
    df.columns = df.columns.str.strip().str.replace('"', '', regex=False)
    return df

def convert_df(df, asignatura_label=None, curso_override=None):
    df = normalize_headers(df.copy())
    cols = {c.lower(): c for c in df.columns}
    def has(name):
        return name in df.columns
    def get_case(name):
        return cols.get(name.lower(), name)
    if 'Nombre' not in df.columns and 'nombre' in cols:
        df.rename(columns={cols['nombre']: 'Nombre'}, inplace=True)
    curso_col = get_case('curso')
    if 'curso' not in df.columns and 'Curso' in df.columns:
        df.rename(columns={'Curso': 'curso'}, inplace=True)
        curso_col = 'curso'
    if not has('curso') and curso_override:
        df['curso'] = str(curso_override)
        curso_col = 'curso'
    nota_col = get_case('Nota')
    if not has('Nota') and 'Promedio' in df.columns:
        df.rename(columns={'Promedio': 'Nota'}, inplace=True)
        nota_col = 'Nota'
    if not has('Nota') and 'promedio' in cols:
        df.rename(columns={cols['promedio']: 'Nota'}, inplace=True)
        nota_col = 'Nota'
    if asignatura_label:
        df['Asignatura'] = str(asignatura_label)
    else:
        df['Asignatura'] = 'Promedio Periodo'
    if nota_col in df.columns:
        df['Nota'] = df['Nota'].astype(str).str.replace(',', '.', regex=False)
        df['Nota'] = pd.to_numeric(df['Nota'], errors='coerce')
    out_cols = ['Nombre', 'curso', 'Asignatura', 'Nota']
    for c in out_cols:
        if c not in df.columns:
            df[c] = ''
    return df[out_cols]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Ruta al archivo CSV/Excel origen')
    parser.add_argument('-o', '--output', help='Ruta del CSV de salida')
    parser.add_argument('--asignatura', help='Etiqueta para la columna Asignatura', default='Promedio Periodo')
    parser.add_argument('--curso', help='Curso a usar si el archivo no lo tiene (ej. "1° basico A")')
    parser.add_argument('--sep', help='Separador de salida', default=';')
    args = parser.parse_args()
    df = load_input(args.input)
    if df is None or df.empty:
        print('Error: no se pudo leer el archivo de entrada o está vacío.')
        sys.exit(1)
    converted = convert_df(df, asignatura_label=args.asignatura, curso_override=args.curso)
    base = os.path.splitext(os.path.basename(args.input))[0]
    out = args.output or os.path.join(os.path.dirname(args.input), base + '_convertido.csv')
    converted.to_csv(out, index=False, sep=args.sep, encoding='utf-8-sig')
    print(out)

if __name__ == '__main__':
    main()
