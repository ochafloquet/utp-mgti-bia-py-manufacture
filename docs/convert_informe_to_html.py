"""
Conversor de Informe Markdown → HTML → PDF
UTP MGTI BIA — Proyecto Final Manufactura

Dependencias:
    pip install markdown weasyprint

Ejecutar:
    python docs/convert_informe_to_html.py
"""
import sys
from pathlib import Path


def convert_md_to_html(md_path: Path, html_path: Path, title: str = "Informe") -> Path:
    """Convierte un archivo Markdown a HTML con estilos académicos."""
    try:
        import markdown
    except ImportError:
        print("  Instalando markdown...")
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'markdown'], check=True)
        import markdown

    with open(md_path, 'r', encoding='utf-8') as f:
        contenido_md = f.read()

    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc', 'nl2br'])
    cuerpo_html = md.convert(contenido_md)

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Open+Sans:wght@400;600&display=swap');

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: 'Open Sans', Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.7;
        color: #2c3e50;
        background: #fff;
        max-width: 900px;
        margin: 0 auto;
        padding: 2cm 2.5cm;
    }
    h1 {
        font-family: 'Merriweather', serif;
        font-size: 20pt;
        color: #1a252f;
        border-bottom: 3px solid #2980b9;
        padding-bottom: 10px;
        margin: 30px 0 20px;
        page-break-before: always;
    }
    h1:first-of-type { page-break-before: avoid; }
    h2 {
        font-size: 15pt;
        color: #2980b9;
        margin: 25px 0 12px;
        border-left: 4px solid #2980b9;
        padding-left: 10px;
    }
    h3 {
        font-size: 12pt;
        color: #34495e;
        margin: 20px 0 8px;
        font-weight: 600;
    }
    p { margin: 0 0 10px; text-align: justify; }
    ul, ol { margin: 8px 0 12px 25px; }
    li { margin-bottom: 4px; }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0 20px;
        font-size: 10pt;
    }
    th {
        background: #2980b9;
        color: white;
        padding: 8px 10px;
        text-align: left;
        font-weight: 600;
    }
    td {
        padding: 7px 10px;
        border-bottom: 1px solid #ecf0f1;
    }
    tr:nth-child(even) td { background: #f8f9fa; }
    code {
        background: #f4f6f8;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 9pt;
    }
    pre {
        background: #1e272e;
        color: #dfe6e9;
        padding: 15px;
        border-radius: 6px;
        overflow-x: auto;
        margin: 15px 0;
    }
    pre code { background: none; color: inherit; padding: 0; }
    blockquote {
        border-left: 4px solid #2980b9;
        background: #eaf4fd;
        padding: 10px 15px;
        margin: 15px 0;
        border-radius: 0 4px 4px 0;
    }
    strong { color: #1a252f; }
    .portada {
        text-align: center;
        padding: 60px 0;
        border-bottom: 2px solid #2980b9;
        margin-bottom: 40px;
    }
    @media print {
        body { padding: 1cm 1.5cm; }
        h1 { page-break-before: always; }
        h1:first-of-type { page-break-before: avoid; }
        table { page-break-inside: avoid; }
    }
    """

    html_completo = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{css}</style>
</head>
<body>
{cuerpo_html}
</body>
</html>
"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_completo)

    print(f"  ✓ HTML generado: {html_path}")
    return html_path


def convert_html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Convierte HTML a PDF usando weasyprint."""
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        print(f"  ✓ PDF generado: {pdf_path}")
        return True
    except ImportError:
        print("  ⚠  weasyprint no instalado. Instalar con: pip install weasyprint")
        print("     Alternativa: Abrir el HTML en el navegador y usar Ctrl+P → Guardar como PDF")
        return False
    except Exception as e:
        print(f"  ⚠  Error al generar PDF: {e}")
        print("     Usa el archivo HTML para imprimir como PDF desde el navegador.")
        return False


if __name__ == '__main__':
    BASE_DIR = Path(__file__).parent
    DOCS_DIR = BASE_DIR

    md_path  = DOCS_DIR / 'INFORME_MANUFACTURA.md'
    html_path = DOCS_DIR / 'INFORME_MANUFACTURA.html'
    pdf_path  = DOCS_DIR / 'INFORME_MANUFACTURA.pdf'

    print("=" * 60)
    print("  Conversor de Informe MD → HTML → PDF")
    print("  UTP MGTI BIA — Manufactura")
    print("=" * 60)

    if not md_path.exists():
        print(f"  ✗ No se encontró: {md_path}")
        sys.exit(1)

    print(f"\n  Convirtiendo {md_path.name} → HTML...")
    convert_md_to_html(md_path, html_path, "Informe Manufactura — UTP BIA")

    print(f"\n  Intentando generar PDF...")
    success = convert_html_to_pdf(html_path, pdf_path)

    print("\n" + "=" * 60)
    if success:
        print(f"  ✓ Archivos generados:")
        print(f"    - {html_path}")
        print(f"    - {pdf_path}")
    else:
        print(f"  ✓ HTML generado: {html_path}")
        print(f"  → Para PDF: Abre el HTML en Chrome/Edge → Ctrl+P → Guardar como PDF")
    print("=" * 60)
